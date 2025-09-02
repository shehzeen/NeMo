# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import random
import tempfile
import numpy as np

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torchaudio
from lightning import LightningModule
from omegaconf import DictConfig, OmegaConf
from peft import PeftModel
from torch import Tensor, nn
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import Replicate, Shard
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    PrepareModuleInput,
    RowwiseParallel,
    SequenceParallel,
    loss_parallel,
    parallelize_module,
)
from transformers import DynamicCache

from nemo.collections.asr.models import EncDecSpeakerLabelModel

from transformers import AutoModelForCausalLM

from nemo.collections.audio.parts.utils.resampling import resample
from nemo.core.classes.module import NeuralModule
from nemo.collections.common.tokenizers import AutoTokenizer
from nemo.collections.nlp.parts.nlp_overrides import NLPSaveRestoreConnector
from nemo.collections.speechlm2.data.utils import get_pad_id
from nemo.collections.speechlm2.models.duplex_s2s_model import tokens_to_str
from nemo.collections.speechlm2.parts.hf_hub import HFHubMixin
from nemo.collections.speechlm2.parts.lora import maybe_install_lora
from nemo.collections.speechlm2.parts.metrics.asr_bleu import ASRBLEU
from nemo.collections.speechlm2.parts.metrics.bleu import BLEU
from nemo.collections.speechlm2.parts.metrics.intelligibility import Intelligibility
from nemo.collections.speechlm2.parts.metrics.results_logger import ResultsLogger
from nemo.collections.speechlm2.parts.metrics.secs import SECS
from nemo.collections.speechlm2.parts.metrics.token_accuracy import TokenAccuracy
from nemo.collections.speechlm2.parts.optim_setup import configure_optimizers, is_frozen
from nemo.collections.speechlm2.parts.precision import fp32_precision
from nemo.collections.speechlm2.parts.pretrained import (
    load_pretrained_hf,
    set_model_dict_for_partial_init,
    setup_speech_encoder,
)
from nemo.core.neural_types import AudioSignal, LabelsType, LengthsType, NeuralType
from nemo.utils import logging

from nemo.collections.tts.modules import transformer_2501
from nemo.collections.tts.modules.mimi_codec_modules import ReshapeTransformerEncoder

from nemo.collections.speechlm2.modules.cfm import MatchaTTSCFM
from types import SimpleNamespace


from nemo.collections.speechlm2.modules.rvq_ear_tts_model import RVQEARTTSModel, RVQEARTTSConfig
from nemo.collections.speechlm2.modules.rvq_ear_tts_vae import RVQVAEModel


def replace_control_speech_codes(speech_codes: torch.Tensor, control_codes: torch.Tensor) -> torch.Tensor:
    """
    Replaces control codes (speech BOS, EOS, etc) in `speech_codes` with the first frame which is
    assumed to consist of 'valid' codes representing silence.
    """
    return torch.where(torch.isin(speech_codes, control_codes), speech_codes[:, :1], speech_codes)


def get_mask_from_lengths(
    lengths: torch.Tensor = None,
    x: torch.Tensor = None,
    pad_to_factor: int = None
) -> torch.Tensor:
    """Constructs binary mask from a 1D torch tensor of input lengths
    Args:
        lengths: torch.tensor (torch.tensor): 1D tensor with lengths
        x: torch.tensor = tensor to be used on, last dimension is for mask
    Returns:
        mask (torch.tensor): num_sequences x max_length binary tensor
    """
    if lengths is None:
        assert x is not None
        return torch.ones(x.shape[-1], dtype=torch.bool, device=x.device)
    else:
        if x is None:
            max_len = torch.max(lengths)
        else:
            max_len = x.shape[-1]

    if pad_to_factor is not None:
        with fp32_precision():
            max_len = torch.ceil(max_len / pad_to_factor) * pad_to_factor

    ids = torch.arange(0, max_len, device=lengths.device, dtype=lengths.dtype)
    mask = ids < lengths.unsqueeze(1)
    return mask



def setup_rvq_audio_codec(model):
    """
    Sets up an ``AudioCodecModel``, initializing it from pretrained weights.
    The result is assigned to ``model.audio_codec`` attribute.

    Includes a workaround for PTL auto-downcasting the codec model to bf16 with bf16-true precision.
    """
    if hasattr(model, "audio_codec") and next(model.audio_codec.parameters()).dtype == torch.float:
        return  # skip if already set up and has the right dtype
    with fp32_precision():
        model.audio_codec = RVQVAEModel.from_pretrained(model.cfg.pretrained_ae_dir).eval().to(model.device)
    for p in model.audio_codec.parameters():
        p.requires_grad = False


class DuplexEARTTS(LightningModule, HFHubMixin):
    def __init__(self, cfg: dict) -> None:
        assert isinstance(cfg, dict), (
            "You must pass the config to ContextAwareMagpieTTS as a Python dict to support hyperparameter serialization "
            f"in PTL checkpoints (we got: '{type(cfg)=}')."
        )
        super().__init__()
        self.save_hyperparameters()
        # convert dict to config
        cfg = DictConfig(cfg)
        self.cfg = cfg.model
        self.target_sample_rate = cfg.data.target_sample_rate
        self.source_sample_rate = cfg.data.source_sample_rate

        self.validation_save_path = os.path.join(cfg.exp_manager.explicit_log_dir, "validation_logs")

        # move back text channel by x, in inference it advance the text channel prediction by x frames
        self.advance_text_channel_by = self.cfg.get("advance_text_channel_by", None)

        # tts general configs
        self.num_delay_tokens = self.cfg.get("num_delay_tokens", 1) # delay between text input and speech output


        # Load ForCausalLM
        self.language_model = self._load_language_model(self.cfg)
        self.embed_tokens = self._load_embed_tokens(self.cfg)

        # codec configs
        setup_rvq_audio_codec(self)

        # compute target fps
        self.target_fps = self.target_sample_rate / self.audio_codec.config.wav_to_token_ratio

        # compute source fps
        self.source_fps = self.source_sample_rate / (
            self.source_sample_rate * cfg.data.frame_length
        )  # conver frame rate in fps
        self.source_samples_per_frame = int(self.source_sample_rate//self.source_fps)
        self.target_samples_per_frame = self.audio_codec.config.wav_to_token_ratio
        # instanciate eartts model
        self.tts_model = self._load_tts_model(self.cfg)
        self._codebook_size = self.tts_model.config.codebook_size

        # Load tokenizer
        self.tokenizer = AutoTokenizer(self.cfg.pretrained_lm_name, use_fast=True)
        if 'Qwen2.5' in self.cfg.pretrained_lm_name:
            # For Qwen, '<|im_start|>' is a common choice for a BOS token.
            # You can check your tokenizer's vocabulary for the best candidate.
            logging.warning("Tokenizer does not have a `bos_token`. Setting it to '<|im_start|>'.")
            self.tokenizer.bos_token = '<|im_start|>'
            self.tokenizer.eos_token = '<|im_end|>'

        # delete llm because we use it only to get the  embbeding tokens
        del self.language_model
        # cached for quicker audio decoding
        self.register_buffer(
            "_control_codes",
            torch.tensor([self.speech_bos_id, self.speech_eos_id, self.speech_pad_id], device=self.device),
        )

        self._use_fsdp = False
        self._use_tp = False

    def _load_embed_tokens(self, cfg) -> nn.Embedding:
        """Load token embedding layer for RVQ-EAR-TTS."""
        if self.language_model:
            assert callable(self.language_model.get_input_embeddings)
            embed_tokens: nn.Embedding = self.language_model.get_input_embeddings()
        else:
            embed_tokens_state_dict = torch.load(
                cfg.pretrained_lm_embedding_path, map_location="cpu", weights_only=True
            )

            # Create token embedding layer
            vocab_size, hidden_size = embed_tokens_state_dict["weight"].size()
            embed_tokens = nn.Embedding(vocab_size, hidden_size, dtype=torch.bfloat16)
            embed_tokens.load_state_dict(embed_tokens_state_dict)
        return embed_tokens

    def _load_tts_model(self, cfg) -> nn.Module:
        """Load TTS model for RVQ-EAR-TTS."""
        audio_codec = self.audio_codec
        tts_model = RVQEARTTSModel.from_pretrained(cfg.pretrained_tts_model, RVQEARTTSConfig(**cfg.tts_config))
        assert callable(tts_model.set_rvq_embs)
        tts_model.set_rvq_embs(torch.stack([x.detach() for x in audio_codec.prvq.mus_list], 0))
        return tts_model

    def _load_language_model(self, cfg):
        """Load language model for RVQ-EAR-TTS."""
        if cfg.pretrained_lm_name:
            language_model = load_pretrained_hf(self.cfg.pretrained_lm_name, pretrained_weights=True).eval()
        else:
            language_model = None
        return language_model


    def setup_speaker_encoder(self):
        with fp32_precision():
            self.speaker_encoder = EncDecSpeakerLabelModel.from_pretrained(model_name=self.speaker_encoder_model_name)

        # freeze the pretrained speaker encoder
        self.speaker_encoder.eval()
        self.speaker_encoder.freeze()

        for p in self.speaker_encoder.parameters():
            p.requires_grad = False

    def init_model_from_another_checkpoint(self, checkpoint_path):
        if checkpoint_path is not None:
            if '.nemo' in checkpoint_path:
                with tempfile.TemporaryDirectory() as tmpdir:
                    NLPSaveRestoreConnector._unpack_nemo_file(checkpoint_path, tmpdir)
                    checkpoint_path = f"{tmpdir}/model_weights.ckpt"
                    checkpoint_state = torch.load(checkpoint_path, map_location='cpu')
            else:
                checkpoint_state = torch.load(checkpoint_path, weights_only=False, map_location='cpu')['state_dict']

            checkpoint_state = set_model_dict_for_partial_init(checkpoint_state, self.state_dict())
            self.load_state_dict(checkpoint_state, strict=True)

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def speech_vocab_size(self):
        """Return the size of the audio codec codebook including extra speech BOS and EOS tokens."""
        if self.use_local_transformer and self.local_transformer_type == "nar": # add extra token for mask
            return self._codebook_size + 4
        return self._codebook_size + 3

    @property
    def speech_bos_id(self) -> int:
        """Indicates start of utterance generation (not start of inference!)."""
        if self.cfg.get("custom_speech_bos_id", None):
            return self.cfg.get("custom_speech_bos_id")
        return self._codebook_size + 2

    @property
    def speech_eos_id(self) -> int:
        """Indicates end of utterance generation."""
        if self.cfg.get("custom_speech_eos_id", None):
            return self.cfg.get("custom_speech_eos_id")
        return self._codebook_size + 1

    @property
    def speech_pad_id(self) -> int:
        """Indicates start of inference (the very first frame)."""
        if self.cfg.get("custom_speech_pad_id", None):
            return self.cfg.get("custom_speech_pad_id")
        return self._codebook_size

    @property
    def text_vocab_size(self):
        """Return the size of the text tokenizer."""
        return self.tokenizer.vocab_size

    @property
    def text_bos_id(self) -> int:
        return self.tokenizer.bos_id

    @property
    def text_zstts_task_id(self) -> int:
        return self.tokenizer.text_to_ids("<|box_start|>") # uses <|box_start|> special token as zstts task id token

    @property
    def text_cont_task_id(self) -> int:
        return self.tokenizer.text_to_ids("<|object_ref_start|>") # uses <|object_ref_start|> special token as cont task id token

    @property
    def text_eos_id(self) -> int:
        return self.tokenizer.eos_id

    @property
    def text_pad_id(self) -> int:
        """
        Text pad ID is used as a 'blank' for frames when the model is not speaking
        and for frames where the model is speaking but has already predicted the
        entire text channel's content.

        Example:

            flow:         |---user---||-------assistant--------||-user-|
            text channel:  0000000000  1xxxxxxx0000000000000002  000000

        Where 0 indicates PAD ID, 1 indicates BOS ID, 2 indacates EOS ID,
        and x indicates tokens corresponding to actual text

        """
        return get_pad_id(self.tokenizer)

    def forward(
        self,
        input_embeds: Tensor,
        cache=None,
        seq_mask=None,
        force_disable_cfg=False,
    ) -> dict[str, Tensor]:
        """
        Separated text and speech prediction:
            - Speech prediction is achieved by a independent AR decoder based on last_hidden_state + audio tokens
            - For KV-cache:
                (1) llm cache depends on input cache is None or Not
                (2) speech_generation cache relys on reset_input_and_kv_cache function.
        """

        if self.cfg_unconditional_prob and not force_disable_cfg:
            if self.training:
                # if training drop the "text" conditioning in a percentage of batch
                if torch.rand(1).item() < self.cfg_unconditional_prob:
                    # make the whole batch zeros to the unconditional model
                    input_embeds = torch.zeros_like(input_embeds)
            elif self.cfg_scale is not None and not self.training:
                # if inference or evaluation create a zero tensor for decoder input and concatenate it to compute unconditional logits
                input_embeds_zeros = torch.zeros_like(input_embeds)
                input_embeds = torch.cat([input_embeds, input_embeds_zeros], dim=0)
                # duplicate mask to match the new shape
                if seq_mask is not None:
                    seq_mask = torch.cat([seq_mask, seq_mask], dim=0)

        out = self.tts_model(
            inputs_embeds=input_embeds,
            attention_mask=seq_mask,
            past_key_values=cache, use_cache=cache is not None, return_dict=True
        )
        B, T = input_embeds.shape[:2]

        # get logits
        logits = self.final_proj(out.last_hidden_state)  # (B, T', num_codebooks * _codebook_size)

        text_logits = None
        if self.use_text_loss:
            text_logits = self.text_head(out.last_hidden_state)

        # if using cfg and it is in inference or evaluation mix unconditional and coditional logits
        if self.cfg_scale is not None and self.cfg_unconditional_prob and not self.training and not force_disable_cfg:
            batch_size = logits.size(0) // 2
            cond_logits = logits[:batch_size]
            uncond_logits = logits[batch_size:]
            logits = (1 - self.cfg_scale) * uncond_logits + self.cfg_scale * cond_logits

            if self.use_text_loss:
                text_cond_logits = text_logits[:batch_size]
                text_uncond_logits = text_logits[batch_size:]
                text_logits = (1 - self.cfg_scale) * text_uncond_logits + self.cfg_scale * text_cond_logits

        ans = {
            "logits": logits,
            "backbone_out": out.last_hidden_state,
            "text_logits": text_logits,
        }
        if cache is not None:
            ans["cache"] = out["past_key_values"]

        return ans

    def pad_audio_codes_to_factor(self, audio_codes: torch.Tensor, downsampling_factor: int = 1, pad_token: int = 0):
        """
        Pads the time dimension of the audio codes to a multiple of the downsampling factor.
        Args:
            audio_codes (torch.Tensor): B, C, T
            downsampling_factor (int): The factor to downsample by.
            pad_token (int): The token ID to pad with.
        Returns:
            B, C, T_padded
        """
        audio_codes = audio_codes.transpose(1, 2)
        T = audio_codes.size(2)
        with fp32_precision():
            T_padded = (torch.ceil(torch.tensor(T / downsampling_factor)) * downsampling_factor).int().item()

        if T_padded > T:
            padding = pad_token * torch.ones(audio_codes.size(0), audio_codes.size(1), T_padded - T, device=audio_codes.device, dtype=audio_codes.dtype)
            audio_codes = torch.cat([audio_codes, padding], dim=2)
        return audio_codes.transpose(1, 2)

    def pad_audio_to_factor(self, audio, audio_len, samples_per_frame, downsampling_factor: int = 1):
        """
        Zero pad the end of the audio so that we do not have a partial end frame.
        The output will be zero-padded to have an integer number of frames of
        length `samples_per_frame * downsampling_factor`.

        Args:
            audio: input time-domain signal (B, T)
            audio_len: valid length for each example in the batch (B,)
            samples_per_frame: number of samples per frame
            downsampling_factor: how much each frame is downsampled in later processing

        Returns:
            padded_audio: Padded time-domain signal (B, T')
            padded_len: Adjusted valid lengths (B,)
        """
        with fp32_precision():
            total_factor = samples_per_frame * downsampling_factor
            padded_len = total_factor * torch.ceil(audio_len / total_factor).int()
            max_len = padded_len.max().int().item()
            num_padding = max_len - audio.shape[1]
            padded_audio = F.pad(audio, (0, num_padding))
        return padded_audio, padded_len

    def prepare_inputs(self, batch: dict):
        """
        """
        # check if audios has the same batch size
        assert batch["source_audio"].size(0) == batch["target_audio"].size(0)
        assert batch["target_first_turn_audio"].size(0) == batch["target_audio"].size(0)

        target_audio = batch["target_audio"]
        target_audio_lens = batch["target_audio_lens"]
        input_text_tokens = batch["input_text_tokens"]
        audio_mask = batch["audio_mask"]
        desc_mask = batch["desc_mask"]
        aligned_attention_mask = batch["aligned_attention_mask"]
        aligned_position_ids = batch["aligned_position_ids"]

        # extract target audio codes
        with fp32_precision(), torch.no_grad():
            target_audio, target_audio_lens = self.pad_audio_to_factor(target_audio, target_audio_lens, self.target_samples_per_frame, 1)
            target_codes, target_codes_lens = self.audio_codec.encode(
                target_audio.unsqueeze(1), target_audio_lens
            )


        # ToDo: consider use the source audio
        """
        # resample source audio if needed
        if self.source_sample_rate != self.target_sample_rate:
            source_audio = resample(source_audio, self.source_sample_rate, self.target_sample_rate)
            with fp32_precision():
                source_audio_lens = (source_audio_lens * (self.target_sample_rate/self.source_sample_rate)).to(lengths.dtype)
        # ToDo: Add a transformer encoder to help the model to better extract contextual information, replace the code bellow with it
        # extract embedding for context audios
        with fp32_precision(), torch.no_grad():
            source_audio, source_audio_lens = self.pad_audio_to_factor(source_audio, source_audio_lens, self.target_samples_per_frame, 1)
            source_codes, source_codes_lens = self.audio_codec.encode(
                source_audio.unsqueeze(1), source_audio_lens
            )
            source_codes = source_codes.transpose(1, 2)  # (B, K, T) -> (B, T, K)
        """
        with fp32_precision():
            if (diff := input_text_tokens.shape[1] - ((target_codes.shape[1]))) < 0:
                # add extra frames on text channels if needed
                input_text_tokens = torch.cat(
                    [
                        input_text_tokens,
                        (
                            torch.ones(target_codes.shape[0], abs(diff), device=target_codes.device) * self.text_pad_id
                        ).to(torch.long),
                    ],
                    dim=-1,
                )
                audio_mask = torch.cat(
                    [
                        audio_mask,
                        (
                            torch.zeros(target_codes.shape[0], abs(diff), device=target_codes.device, dtype=audio_mask.dtype)
                        ).to(torch.long),
                    ],
                    dim=-1,
                )
                desc_mask = torch.cat(
                    [
                        desc_mask,
                        (
                            torch.zeros(target_codes.shape[0], abs(diff), device=target_codes.device, dtype=desc_mask.dtype)
                        ).to(torch.long),
                    ],
                    dim=-1,
                )
                pad_mask = torch.zeros(
                    aligned_attention_mask.shape[0],     # batch size
                    aligned_attention_mask.shape[1],     # head dim (usually 1)
                    aligned_attention_mask.shape[2],     # seq_len (rows)
                    abs(diff),                                # columns to pad
                    device=target_codes.device,
                    dtype=aligned_attention_mask.dtype
                )
                aligned_attention_mask = torch.cat([aligned_attention_mask, pad_mask], dim=-1)

            elif diff > 0:
                input_text_tokens = input_text_tokens[:, : target_codes.shape[1]]
                audio_mask = audio_mask[:, : target_codes.shape[1]]
                desc_mask = desc_mask[:, : target_codes.shape[1]]


        # set the pad token when there is desc as in https://gitlab-master.nvidia.com/jaehyeonk/easy-ar-tts/-/blame/simple-bq/scripts/train_tts_with_rvqvae.py#L69
        target_codes_aligned = torch.where(
            desc_mask.unsqueeze(-1),                    # (B, T, 1) for broadcasting
            torch.full_like(target_codes, self.speech_pad_id),  # fill with pad id
            target_codes
        )

        B, T = input_text_tokens.shape

        # ToDo: handle BOS and EOS as duplex
        """

        # Add BOS and EOS on speech channel considering
        bos_indices = (input_text_tokens == self.text_bos_id).nonzero(as_tuple=False)  # [N_bos, 2]
        eos_indices = (input_text_tokens == self.text_eos_id).nonzero(as_tuple=False)  # [N_eos, 2]

        # Modify target_codes_aligned in-place at aligned positions
        if bos_indices.numel() > 0:
            target_codes_aligned[bos_indices[:, 0], bos_indices[:, 1]] = self.speech_bos_id

        if eos_indices.numel() > 0:
            target_codes_aligned[eos_indices[:, 0], eos_indices[:, 1]] = self.speech_eos_id
        """

        # shift text tokens as done in https://gitlab-master.nvidia.com/jaehyeonk/easy-ar-tts/-/blob/simple-bq/scripts/train_tts_with_rvqvae.py#L118
        subword_ids = F.pad(input_text_tokens[:, 1:], [0, 1])
        subword_mask = F.pad(audio_mask[:, 1:], [0, 1]) # use audio_mask as subword_mask to be able to support duplex training

        if self._use_tp:
            tp_world_size = self.device_mesh["tensor_parallel"].size()
            if (remainder := (input_text_tokens.shape[1] - 1) % tp_world_size) != 0:
                input_text_tokens = input_text_tokens[:, :-remainder]
                target_codes_aligned = target_codes_aligned[:, :-remainder]
                target_codes_aligned = target_codes_aligned[:, :-remainder]
                audio_mask = audio_mask[:, :-remainder]
                desc_mask = desc_mask[:, :-remainder]
                subword_ids = subword_ids[:, :-remainder]
                subword_mask = subword_mask[:, :-remainder]
    

        # debug samples:
        if (
            self.cfg.get("debug_dataloader_audios_path", None)
            and self.training
        ):
            def write_wave(one_audio_signal, file_name, sr=None):
                import numpy as np
                import soundfile as sf

                one_audio_signal = one_audio_signal.cpu().numpy()
                one_audio_signal = one_audio_signal.astype(np.float32)
                if sr is None:
                    sr = self.target_sample_rate
                # one_audio_signal = np.clip(one_audio_signal, -1.0, 1.0)
                sf.write(file_name, one_audio_signal, sr)

            # encode and decode the audio
            with fp32_precision(), torch.no_grad():
                lengths = torch.tensor([batch["target_audio"].shape[1]] * batch["target_audio"].shape[0]).to(
                    self.device
                )
                # reconstruct wav
                print("target_codes_aligned:", target_codes_aligned.shape)
                target_codes_aligned_ = replace_control_speech_codes(target_codes, self._control_codes)
                print(self._control_codes, target_codes_aligned_.shape)
                with fp32_precision(), torch.no_grad():
                    lengths = torch.tensor([target_codes_aligned_.shape[1]] * target_codes_aligned_.shape[0]).to(
                        self.device
                    )
                    print(target_codes_aligned_.max(), target_codes.max())
                    reconstructed_audio_from_tokens, _ = self.audio_codec.decode(
                        target_codes_aligned_, lengths
                    )
                    reconstructed_audio_from_tokens = reconstructed_audio_from_tokens.squeeze(1)
                    print(reconstructed_audio_from_tokens.shape, batch["target_audio"].shape)

            for i in range(target_codes_aligned_.shape[0]):
                write_wave(
                    batch["target_audio"][i],
                    os.path.join(self.cfg.get("debug_dataloader_audios_path"), f"target_audio_{i}.wav"),
                    sr=self.target_sample_rate,
                )
                write_wave(
                    batch["target_first_turn_audio"][i],
                    os.path.join(self.cfg.get("debug_dataloader_audios_path"), f"speaker_ref_{i}.wav"),
                    sr=self.target_sample_rate,
                )
                write_wave(
                    batch["source_audio"][i],
                    os.path.join(self.cfg.get("debug_dataloader_audios_path"), f"source_audio_{i}.wav"),
                    sr=self.source_sample_rate,
                )

                write_wave(
                    reconstructed_audio_from_tokens[i],
                    os.path.join(
                        self.cfg.get("debug_dataloader_audios_path"), f"target_audio_reconstructed_from_tokens_{i}.wav"
                    ),
                    sr=self.target_sample_rate,
                )


            print(
                "target labels from dataloader decoded:",
                tokens_to_str(
                    batch["input_text_tokens"][-1:],
                    target_codes_lens,
                    tokenizer=self.tokenizer,
                    pad_id=self.text_pad_id,
                ),
            )
            print(batch["formatter"])
            if target_codes_aligned_.shape[0] > 1:
                exit()

        return {
            "code": target_codes_aligned,
            "audio_mask": audio_mask,
            "attention_mask": aligned_attention_mask,
            "position_ids": aligned_position_ids,
            "subword_ids": subword_ids,
            "subword_mask": subword_mask,
            "context_hidden_state": None, # ToDo: implement external llm latent
            "output_lens": target_codes_lens - 1,
        }

    def training_step(self, batch: dict, batch_idx: int):
        for m in (self.tts_model, ):
            if is_frozen(m):
                m.eval()

        inputs = self.prepare_inputs(batch)

        """forward_outputs = self(
            inputs["input_embeds"],
            seq_mask=inputs["seq_mask"]
        )

        codebook_loss, loss_mask = self.compute_loss(forward_outputs["logits"], inputs["target_codes_aligned"],  inputs["output_lens"], loss_scale=inputs["loss_scale"])
        """

        codebook_loss = 0.0
        loss = 0.0
        num_frames = inputs["input_lens"].sum()
        B, T = inputs["input_embeds"].shape[:2]
        ans = {
            "loss": loss,
            "learning_rate": (
                torch.as_tensor(self.trainer.optimizers[0].param_groups[0]['lr'] if self._trainer is not None else 0)
            ),
            "codebook_loss": codebook_loss,
            "batch_size": B,
            "sequence_length": T,
            "num_frames": num_frames.to(torch.float32),  # avoid warning
            "padding_ratio": num_frames / (B * T),
        }

        self.log_dict(ans, on_step=True)
        return ans

    def on_train_epoch_start(self) -> None:
        setup_rvq_audio_codec(self)  # potentially reloads the audio codec to make sure it's in fp32

    def on_validation_epoch_start(self) -> None:
        self.on_train_epoch_start()
        self.results_logger = ResultsLogger(self.validation_save_path).reset()
        self.asr_bleu = ASRBLEU(self.cfg.scoring_asr).reset()
        self.intelligibility = Intelligibility(self.cfg.scoring_asr, reuse_asr_hyps=True).reset()
        self.secs = SECS(self.cfg.get("scoring_se", "titanet_large")).reset()

    def on_validation_epoch_end(self, prefix="val") -> None:
        asr_bleu = self.asr_bleu.compute()
        for k, m in asr_bleu.items():
            self.log(f"{prefix}_{k}", m.to(self.device), on_epoch=True, sync_dist=True)
        cer_wer = self.intelligibility.compute()
        for k, m in cer_wer.items():
            self.log(f"{prefix}_{k}", m.to(self.device), on_epoch=True, sync_dist=True)
        secs = self.secs.compute()
        for k, m in secs.items():
            self.log(f"{prefix}_{k}", m.to(self.device), on_epoch=True, sync_dist=True)

    def validation_step(self, batch: dict, batch_idx: int):

        for name, dataset_batch in batch.items():
            if dataset_batch is None:
                continue  # some dataset is exhausted

            results = self.offline_inference(
                dataset_batch["source_audio"],
                dataset_batch["source_audio_lens"],
                speaker_audio=dataset_batch["target_first_turn_audio"],
                speaker_audio_lens=dataset_batch["target_first_turn_audio_lens"],
                text_tokens=dataset_batch["input_text_tokens"],
                formatter=dataset_batch["formatter"][0],
            )

            results["tf_audio_pred"] = self.get_teacher_force_inference_audio(dataset_batch)

            with fp32_precision():  # resample is fragile to bfloat16 default dtype
                metric_audio_pred = results["audio"]
                metric_audio_pred_lens = results["audio_len"]

                # resample audio to the asr sampling rate
                metric_audio_pred = resample(metric_audio_pred, self.target_sample_rate, 16000)
                metric_audio_pred_lens = (metric_audio_pred_lens / self.target_sample_rate * 16000).to(torch.long)

                asr_hyps = self.asr_bleu.update(
                    name=name,
                    refs=dataset_batch["target_texts"],
                    pred_audio=metric_audio_pred,
                    pred_audio_lens=metric_audio_pred_lens,
                )

                self.intelligibility.update(
                    name=name,
                    refs=dataset_batch["target_texts"],
                    pred_audio=metric_audio_pred,
                    pred_audio_lens=metric_audio_pred_lens,
                    asr_hyps=asr_hyps,
                )

                self.secs.update(
                    name=name,
                    target_audio=resample(dataset_batch["target_audio"], self.target_sample_rate, 16000),
                    target_audio_lens=(dataset_batch["target_audio_lens"] / self.target_sample_rate * 16000).to(torch.long),
                    pred_audio=resample(results["audio"], self.target_sample_rate, 16000),
                    pred_audio_lens=(results["audio_len"] / self.target_sample_rate * 16000).to(torch.long),
                )

                self.results_logger.update(
                    name=name,
                    refs=dataset_batch["target_texts"],
                    hyps=results["text"],
                    asr_hyps=asr_hyps,
                    samples_id=dataset_batch['sample_id'],
                    pred_audio=results["audio"],
                    pred_audio_tf=results["tf_audio_pred"],
                    pre_audio_trimmed=results["trimmed_audio"],
                    pred_audio_sr=self.target_sample_rate,
                    user_audio=dataset_batch["source_audio"],
                    user_audio_sr=self.source_sample_rate,
                    eou_pred=None,
                    fps=self.target_fps,
                    results=results if self.cfg.get("dump_tokens_text", False) else None,
                    tokenizer=self.tokenizer,
                )

    def on_test_epoch_start(self) -> None:
        return self.on_validation_epoch_start()

    def on_test_epoch_end(self) -> None:
        return self.on_validation_epoch_end(prefix="test")

    def test_step(self, *args, **kwargs):
        return self.validation_step(*args, **kwargs)

    def _get_text_pad_embedding(self) -> torch.Tensor:
        """
        Remove the audio codec embedding for the beginning of AR decoding.
        """
        text_bos = torch.full((1,), fill_value=self.text_pad_id, device=self.device)
        input_embeds = self.embed_text_tokens(text_bos)
        return text_bos, input_embeds

    @torch.no_grad()
    def offline_inference(
        self,
        source_audio: torch.Tensor,
        source_audio_lens: torch.Tensor,
        speaker_audio: torch.Tensor,
        speaker_audio_lens: torch.Tensor,
        text_tokens: torch.Tensor,
        decode_audio: bool = True,
        formatter: str = "",
    ) -> dict[str, torch.Tensor]:
        """
        Autoregressive prediction.

        Args:
            input_signal: a batch of waveforms with shape (B, T) with source sampling rate.
            input_signal_lens: example lengths as number of samples of shape (B,).
            decode_audio: bool, whether to decode audio codes to waveform.

        Returns:
            A dict with keys:
                * "text": generated text, de-tokenized to strings, properly skipping text_pad_id; list of length B.
                * "tokens_text": generated text tokens of shape (B, T2).
                * "tokens_audio": generated audio codes of shape (B, T2, K) where `K=num_codebooks`.
                * "tokens_len" output lengths as number of tokens of shape (B,).
                * "audio": generated waveform of shape (B, T3) (`decode_audio=True`).
                * "audio_len" output lengths as number of waveform samples of shape (B,) (when `decode_audio=True`).
        """


        # gen_audio_codes B, T=?, C=8, F=2
        ans = {
        }

        
        return ans

    def backward(self, *args, **kwargs):
        with loss_parallel():
            super().backward(*args, **kwargs)

    def configure_optimizers(self):
        return configure_optimizers(self)

    @property
    def oomptimizer_schema(self) -> dict:
        """
        Return a typing schema for optimal batch size calibration for various
        sequence lengths using OOMptimizer.
        """
        return {
            "cls": dict,
            "inputs": [
                {"name": "source_audio", "type": NeuralType(("B", "T"), AudioSignal()), "seq_length": "input"},
                {"name": "source_audio_lens", "type": NeuralType(("B",), LengthsType()), "seq_length": "input"},
                {"name": "target_audio", "type": NeuralType(("B", "T"), AudioSignal()), "seq_length": "input"},
                {"name": "target_audio_lens", "type": NeuralType(("B",), LengthsType()), "seq_length": "input"},
                {
                    "name": "input_text_tokens",
                    "type": NeuralType(("B", "T"), LabelsType()),
                    "seq_length": "output",
                    "vocab_size": self.tokenizer.vocab_size,
                },
            ],
        }

    def configure_model(self) -> None:
        # TODO(pzelasko): refactor into separate module re-usable across models
        device_mesh = self.device_mesh
        if device_mesh is None:
            return

        llm = self.tts_model.backbone
        if isinstance(llm, PeftModel):
            llm = llm.base_model.model

        if (tp_mesh := device_mesh["tensor_parallel"]).size() > 1:
            self._use_tp = True

            plan = {
                "layers.0": PrepareModuleInput(
                    input_layouts=(Replicate(),),  # , None)
                    desired_input_layouts=(Shard(1),),  # , None)
                    use_local_output=True,
                ),
                "norm": SequenceParallel(),
            }
            parallelize_module(llm, tp_mesh, plan)

            for transformer_block in llm.layers:
                plan = {
                    "input_layernorm": SequenceParallel(),
                    "self_attn.q_proj": ColwiseParallel(),
                    "self_attn.k_proj": ColwiseParallel(),
                    "self_attn.v_proj": ColwiseParallel(),
                    "self_attn.o_proj": RowwiseParallel(output_layouts=Shard(1)),
                    "post_attention_layernorm": SequenceParallel(),
                    "mlp": PrepareModuleInput(
                        input_layouts=(Shard(1),),
                        desired_input_layouts=(Replicate(),),
                    ),
                    "mlp.gate_proj": ColwiseParallel(),
                    "mlp.up_proj": ColwiseParallel(),
                    "mlp.down_proj": RowwiseParallel(output_layouts=Shard(1)),
                    # "pre_feedforward_layernorm": SequenceParallel(),
                    # "post_feedforward_layernorm": SequenceParallel(),
                }

                # Adjust attention module to use the local number of heads
                attn_layer = transformer_block.self_attn
                for attr in ("num_heads", "num_key_value_heads", "hidden_size"):
                    val = getattr(attn_layer, attr)
                    if val % tp_mesh.size() != 0:
                        logging.warning(
                            f"attn_layer.{attr}={val} is not divisible by {tp_mesh.size()=}: "
                            f"set a different tensor parallelism size to avoid errors."
                        )
                    setattr(attn_layer, attr, val // tp_mesh.size())

                parallelize_module(transformer_block, tp_mesh, plan)

            for m in (self.tts_model.mog_head, self.tts_model.embed_subword, self.tts_model.embed_context, self.tts_model.embed_code, self.tts_model.null_emb, self.tts_model.bos_emb, self.tts_model.lm_head):
                parallelize_module(
                    m,
                    tp_mesh,
                    ColwiseParallel(
                        input_layouts=Shard(1),
                        output_layouts=Shard(-1),
                        use_local_output=False,
                    ),
                )

        if (dp_mesh := device_mesh["data_parallel"]).size() > 1:
            assert dp_mesh.ndim == 1
            self._use_fsdp = True

            fsdp_config = {"mesh": dp_mesh}

            for idx, layer in enumerate(llm.layers):
                llm.layers[idx] = fully_shard(layer, **fsdp_config)
            self.embed_text_tokens = fully_shard(self.embed_text_tokens, **fsdp_config)
            # self.tts_model = fully_shard(self.tts_model, **fsdp_config)
            self.tts_model.mog_head = fully_shard(self.tts_model.mog_head, **fsdp_config)
            self.tts_model.embed_subword = fully_shard(self.tts_model.embed_subword, **fsdp_config)
            self.tts_model.embed_context = fully_shard(self.tts_model.embed_context, **fsdp_config)
            self.tts_model.embed_code = fully_shard(self.tts_model.embed_code, **fsdp_config)
            self.tts_model.null_emb = fully_shard(self.tts_model.null_emb, **fsdp_config)
            self.tts_model.bos_emb = fully_shard(self.tts_model.bos_emb, **fsdp_config)
            self.tts_model.lm_head = fully_shard(self.tts_model.lm_head, **fsdp_config)

    def load_state_dict(self, state_dict, strict: bool = True):
        try:
            super().load_state_dict(state_dict, strict=strict)
        except RuntimeError as e:
            logging.info(f"Error loading model state_dict !! Retrying with partial initialization!")
            model_dict = set_model_dict_for_partial_init(state_dict, self.state_dict())
            super().load_state_dict(model_dict, strict=False)
