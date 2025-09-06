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
import math

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
from nemo.collections.speechlm2.modules.ear_tts_commons import SCRIPT_PLACEHOLDER

from nemo.collections.speechlm2.modules.cfm import MatchaTTSCFM
from types import SimpleNamespace


from nemo.collections.speechlm2.modules.rvq_ear_tts_model import RVQEARTTSModel, RVQEARTTSConfig
from nemo.collections.speechlm2.modules.rvq_ear_tts_vae import RVQVAEModel


def replace_control_speech_codes(speech_codes: torch.Tensor, control_codes: torch.Tensor, silence_tokens: torch.Tensor = None) -> torch.Tensor:
    """
    Replaces control codes (speech BOS, EOS, etc) in `speech_codes` with the first frame which is
    assumed to consist of 'valid' codes representing silence.
    """
    if silence_tokens is not None:
        # Expand to [B, 1, 74]
        silence_tokens_expanded = silence_tokens.unsqueeze(0).unsqueeze(1).expand(speech_codes.shape[0], 1, -1)
        return torch.where(torch.isin(speech_codes, control_codes), silence_tokens_expanded, speech_codes)

    if torch.isin(speech_codes[:, :1], control_codes).any():
        return torch.where(torch.isin(speech_codes, control_codes), torch.zeros_like(speech_codes[:, :1]), speech_codes)
    else:
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


def compare_init_dicts(dict1, dict2, atol=1e-5, rtol=1e-3):
    """
    Compare two init_input dictionaries key by key.
    Prints shape, dtype, and whether values match (for tensors).
    """
    keys1 = set(dict1.keys())
    keys2 = set(dict2.keys())
    print("Keys only in dict1:", keys1 - keys2)
    print("Keys only in dict2:", keys2 - keys1)
    print("Common keys:", keys1 & keys2)
    print("=" * 60)

    for key in keys1 & keys2:
        v1, v2 = dict1[key], dict2[key]

        print(f"\n🔑 {key}")
        if isinstance(v1, torch.Tensor) and isinstance(v2, torch.Tensor):
            same_shape = v1.shape == v2.shape
            same_dtype = v1.dtype == v2.dtype
            try:
                close = torch.allclose(v1, v2, atol=atol, rtol=rtol)
            except Exception:
                close = False

            print(f"  shape1={tuple(v1.shape)}, shape2={tuple(v2.shape)}, same_shape={same_shape}")
            print(f"  dtype1={v1.dtype}, dtype2={v2.dtype}, same_dtype={same_dtype}")
            print(f"  allclose={close}")

            # If shapes differ, show min shape content preview
            if not same_shape:
                print("  ⚠️ Shapes differ, showing first few elements:")
                print(f"    dict1[{key}][:5] -> {v1.view(-1)[:5]}")
                print(f"    dict2[{key}][:5] -> {v2.view(-1)[:5]}")
        else:
            same_type = type(v1) == type(v2)
            same_val = v1 == v2 if same_type else False
            print(f"  type1={type(v1)}, type2={type(v2)}, same_type={same_type}")
            print(f"  value equal? {same_val}")
            print(f"  val1={str(v1)[:100]}")
            print(f"  val2={str(v2)[:100]}")

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
        self.data_cfg = cfg.data
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
        self.codec_silence_tokens = self.get_codec_silence_frame()

        # Load tokenizer
        self.tokenizer = AutoTokenizer(self.cfg.pretrained_lm_name, use_fast=True) # Note that we are using fast tokenizer

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
        if self.cfg.get("pretrained_model", None):
            self.init_model_from_another_checkpoint(self.cfg.pretrained_model)


    def get_codec_silence_frame(self):
        audio = torch.zeros(1, 10*self.target_sample_rate).float().to(self.device)
        audio_len = torch.tensor([audio.size(-1)]).long()
        audio, audio_len = self.pad_audio_to_factor(audio, audio_len, self.target_samples_per_frame)

        with fp32_precision(), torch.no_grad():
            sil_codes, sil_codes_lens = self.audio_codec.encode(
                    audio.unsqueeze(1), audio_len
                )
            return sil_codes[0, -1]

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
        assert batch["speaker_reference_audio"].size(0) == batch["target_audio"].size(0)

        target_audio = batch["target_audio"]
        target_audio_lens = batch["target_audio_lens"]
        input_text_tokens = batch["input_text_tokens"]
        audio_mask = batch["audio_mask"]
        desc_mask = batch["desc_mask"]
        non_prompt_mask = batch["non_prompt_mask"]
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
            target_len = target_codes.shape[1]

            # Pad or truncate sequence variables
            def pad_or_truncate(x, pad_value=0):
                if x.dim() == 2:  # [B, T]
                    L = x.shape[1]
                    if L < target_len:
                        return F.pad(x, (0, target_len - L), value=pad_value)
                    else:
                        return x[:, :target_len]
                return x  # leave others for now

            input_text_tokens = pad_or_truncate(input_text_tokens, pad_value=self.text_pad_id)
            audio_mask = pad_or_truncate(audio_mask, pad_value=0)
            desc_mask = pad_or_truncate(desc_mask, pad_value=0)
            non_prompt_mask = pad_or_truncate(non_prompt_mask, pad_value=0)
            aligned_position_ids = pad_or_truncate(aligned_position_ids, pad_value=0)

            # Correct attention mask padding/truncation
            B, H, L1, L2 = aligned_attention_mask.shape
            new_len = target_len
            if L1 < new_len or L2 < new_len:
                pad_rows = new_len - L1
                pad_cols = new_len - L2
                aligned_attention_mask = F.pad(aligned_attention_mask, (0, pad_cols, 0, pad_rows))
            elif L1 > new_len or L2 > new_len:
                aligned_attention_mask = aligned_attention_mask[:, :, :new_len, :new_len]

        # set the pad token when there is desc as in https://gitlab-master.nvidia.com/jaehyeonk/easy-ar-tts/-/blame/simple-bq/scripts/train_tts_with_rvqvae.py#L69
        target_codes_aligned = torch.where(
            desc_mask.unsqueeze(-1),                    # (B, T, 1) for broadcasting
            torch.full_like(target_codes, self.speech_pad_id),  # fill with pad id
            target_codes
        )

        B, T = input_text_tokens.shape
        # ToDo: consider to handle Speech BOS and EOS as duplex
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

        # ToDo: remove links before merge the PR
        # shift text tokens as done in https://gitlab-master.nvidia.com/jaehyeonk/easy-ar-tts/-/blob/simple-bq/scripts/train_tts_with_rvqvae.py#L118
        subword_ids = F.pad(input_text_tokens[:, 1:], [0, 1])
        if self.cfg.get("subword_mask_exactly_as_eartts", False):
            # ignore prompt using non_prompt_mask
            mask_1 = F.pad(non_prompt_mask[:, 1:], [0, 1])
            # ignore extra silences checking subword_ids
            mask_2 = ~(subword_ids == self.text_pad_id)
            # subword_mask is only true when both mask_1 and mask_2 are true
            subword_mask = (mask_1.bool() & mask_2.bool()).detach()
        else:
            # WARNING: note that we are using a text mask where we are ignoring the desc + audio prompt but we are keeping 1 until the audio ends to support duplex
            subword_mask = F.pad(non_prompt_mask[:, 1:], [0, 1])

        # ToDo: implement context from the llm
        # detach embedding as in eartts
        context_hidden_state = self.embed_tokens(input_text_tokens).detach()
        # On EARTTS they use masked_scatter_ and make sure that the where there is the padding tokens it is actually zeros
        if self.cfg.get("context_hidden_mask_exactly_as_eartts", False):
            # context_hidden_mask is True when we have valids BPE tokens
            # ToDo: masking as eartts is producing Nans for some reason, investigate it.
            context_hidden_mask = (input_text_tokens.long() != self.text_pad_id).bool()
            context_hidden_state = context_hidden_state * context_hidden_mask.unsqueeze(-1).to(context_hidden_state.dtype)

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
                print(batch["target_audio"].shape)
                lengths = torch.tensor([batch["target_audio"].shape[1]] * batch["target_audio"].shape[0]).to(
                    self.device
                )
                # reconstruct wav
                print("target_codes_aligned:", target_codes_aligned.shape)
                target_codes_aligned_ = replace_control_speech_codes(target_codes, self._control_codes, self.codec_silence_tokens)
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
                    batch["speaker_reference_audio"][i],
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
            "context_hidden_state": context_hidden_state,
            "output_lens": target_codes_lens,
            "non_prompt_mask": non_prompt_mask,
            "input_text_tokens": input_text_tokens,
        }

    def training_step(self, batch: dict, batch_idx: int):
        for m in (self.tts_model, ):
            if is_frozen(m):
                m.eval()

        inputs = self.prepare_inputs(batch)

        tts_output = self.tts_model(
            code=inputs["code"],
            audio_mask=inputs["audio_mask"],
            attention_mask=inputs["attention_mask"],
            position_ids=inputs["position_ids"],
            context_hidden_state=inputs["context_hidden_state"],
            subword_ids=inputs["subword_ids"],
            subword_mask=inputs["subword_mask"],
            non_prompt_mask=inputs["non_prompt_mask"],
        )
        loss_dict = {"lm_loss": tts_output.lm_loss, "c_loss": tts_output.c_loss, "k_loss": tts_output.k_loss}
        loss = sum(loss_dict.values())

        num_frames = inputs["output_lens"].sum()
        B, T = inputs["code"].shape[:2]
        ans = {
            "loss": loss,
            "learning_rate": (
                torch.as_tensor(self.trainer.optimizers[0].param_groups[0]['lr'] if self._trainer is not None else 0)
            ),
            "batch_size": B,
            "sequence_length": T,
            "num_frames": num_frames.to(torch.float32),  # avoid warning
            "padding_ratio": num_frames / (B * T),
            **loss_dict,
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

    def get_teacher_force_inference_audio(self, batch, guidance_enabled=True):
        inputs = self.prepare_inputs(batch)

        tts_output = self.tts_model(
            code=inputs["code"],
            audio_mask=inputs["audio_mask"],
            attention_mask=inputs["attention_mask"],
            position_ids=inputs["position_ids"],
            context_hidden_state=inputs["context_hidden_state"],
            subword_ids=inputs["subword_ids"],
            subword_mask=inputs["subword_mask"],
            non_prompt_mask=inputs["non_prompt_mask"],
            generation_config=self._get_generation_config(guidance_enabled=guidance_enabled),
            teacher_forcing_inference=True,
            guidance_enabled=guidance_enabled,
        )
        tf_audio_codes_pred = tts_output.codes.squeeze(2)

        # decode audio
        tf_audio_codes_pred = replace_control_speech_codes(tf_audio_codes_pred, self._control_codes, self.codec_silence_tokens)
        with fp32_precision(), torch.no_grad():
            audio_pred, audio_len = self.audio_codec.decode(
                tf_audio_codes_pred, inputs["output_lens"]
            )

        return audio_pred.squeeze(1), audio_len

    def _get_generation_config(self, guidance_enabled: bool = False):
        """Get default generation config for EAR-TTS."""
        return {
            "num_iter": 8,
            "guidance_scale": 0.5 if guidance_enabled else None,
            "top_p_or_k": 0.8,
            "noise_scale": 0.8,
            "eos_threshold": -3.0,
        }

    def validation_step(self, batch: dict, batch_idx: int):

        for name, dataset_batch in batch.items():
            if dataset_batch is None:
                continue  # some dataset is exhausted

            results = {}
            inputs = self.prepare_inputs(dataset_batch)
            """
            # cut it on prompt
            init_inputs = {
                "code": inputs["code"],
                "audio_mask": inputs["audio_mask"],
                "non_prompt_mask": inputs["non_prompt_mask"],
                "context_hidden_state": inputs["context_hidden_state"],
                "subword_ids": inputs["subword_ids"],
                "subword_mask": inputs["subword_mask"],
            }
            # cut init_inputs to consider only the prompt
            for key in init_inputs:
                init_inputs[key] = torch.stack([
                    init_inputs[key][i, :l-1]
                    for i, l in enumerate(dataset_batch["desc_plus_audio_prompt_lens"])
                ])
            """

            # remove the prompt from the input_text_tokens to emulate S2S connected inference
            next_subword_ids = torch.stack([
                inputs["subword_ids"][i, l-1:]  # slice each element
                for i, l in enumerate(dataset_batch["desc_plus_audio_prompt_lens"])
            ])
            next_input_text_tokens = torch.stack([
                inputs["input_text_tokens"][i, l-1:]  # slice each element
                for i, l in enumerate(dataset_batch["desc_plus_audio_prompt_lens"])
            ])

            results["audio"], results["audio_len"] = self.offline_inference(
                speaker_audio=dataset_batch["speaker_reference_audio"],
                speaker_audio_lens=dataset_batch["speaker_reference_audio_lens"],
                next_subword_ids=next_subword_ids,
                next_input_text_tokens=next_input_text_tokens,
                formatter=dataset_batch["formatter"][0],
                # init_inputs=init_inputs,
            )

            results["audio_tf"], results["audio_tf_len"] = self.get_teacher_force_inference_audio(dataset_batch)
            # clean prompt from the audio
            for i, l in enumerate(dataset_batch["desc_plus_audio_prompt_lens"]):
                results["audio_tf"][i, :l*self.target_samples_per_frame] = 0.0

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
                    hyps=dataset_batch["target_texts"],
                    asr_hyps=asr_hyps,
                    samples_id=dataset_batch['sample_id'],
                    pred_audio=results["audio"],
                    pred_audio_tf=results["audio_tf"],
                    pre_audio_trimmed=None,
                    reference_audio=dataset_batch["speaker_reference_audio"],
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

    def get_system_prompt(self, system_prompt=None, user_prompt=None):
        messages = []
        if system_prompt is None:
            system_prompt = (
                "You engage in conversation with the user. When delivering your response as speech, "
                "if the user provides a description such as emotions, scene details, "
                "or speaker style, you adjust your speaking style accordingly when delivering the response. "
                "However, this description should influence only the delivery of your response, not its content. "
                "Your response should remain independent of any stylistic instructions."
            )
        messages.append({"role": "system", "content": system_prompt})
        
        # ToDo: implement dataloading support for descriptions
        """for desc in example["descriptions"]:
            user_prompt = ""
            if random.random() > self.p_drop_description and desc:
                user_prompt += f"```\n{desc}\n```"
            if random.random() > self.p_drop_description:
                if user_prompt:
                    user_prompt += "\n\n"
                user_prompt += self.rng.choice(self.user_prompts)
            if user_prompt:
                messages.append({"role": "user", "content": user_prompt})
            messages.append({"role": "assistant", "content": SCRIPT_PLACEHOLDER})
        """

        # given that descriptions are currently not supported, only added the user prompt
        if user_prompt is None:
            user_prompt = "Can you tell me something interesting?"
        messages.append({"role": "user", "content": user_prompt})
        messages.append({"role": "assistant", "content": SCRIPT_PLACEHOLDER})
        non_script_list = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        ).split(SCRIPT_PLACEHOLDER + self.tokenizer.eos_token)[:-1]

        input_ids = []
        for i, non_script in enumerate(non_script_list):
            desc_ids = self.tokenizer.text_to_ids(non_script)
            input_ids.extend(desc_ids)

        input_ids = torch.tensor(input_ids, dtype=torch.long, device=self.device).view(1, -1)
        return input_ids

    def get_init_inputs(self, speaker_audio, speaker_audio_lens, system_prompt=None, user_prompt=None):
        # compute prompt audio size and slice it
        prompt_audio_size = int(((self.data_cfg.audio_prompt_duration * self.target_sample_rate) // self.target_samples_per_frame) * self.target_samples_per_frame)
        prompt_audio = speaker_audio[:, :prompt_audio_size]
        # get prompt audio size
        prompt_audio_text_pad_size = prompt_audio_size // self.target_samples_per_frame
        
        # get description tokens
        desc_tokens_ids = self.get_system_prompt(system_prompt=system_prompt, user_prompt=user_prompt)

        # create a padding tensor
        prompt_audio_text_pad = torch.ones(prompt_audio_text_pad_size, device=self.device, dtype=desc_tokens_ids.dtype) * self.text_pad_id
        # Add eos to simulate the end of a turn as in EAR-TTS inference
        desc_tokens_ids = torch.cat([desc_tokens_ids.squeeze(), torch.tensor([self.tokenizer.eos], dtype=desc_tokens_ids.dtype, device=desc_tokens_ids.device)])
        # Add padding equivalent to the audio prompt size in number of tokens
        input_text_tokens = torch.cat([desc_tokens_ids.to(desc_tokens_ids.dtype), prompt_audio_text_pad.to(desc_tokens_ids.dtype)])

        # create pad audio for the description
        pad_size = desc_tokens_ids.size(-1) * self.target_samples_per_frame
        pad_audio = torch.zeros(pad_size, device=prompt_audio.device, dtype=prompt_audio.dtype).unsqueeze(0).repeat(prompt_audio.size(0), 1)

        # set eos right after the audio prompt
        # input_text_tokens[len(desc_tokens_ids) + prompt_audio_text_pad_size] = self.tokenizer.eos
        # repeat to reaches the batch size
        input_text_tokens = input_text_tokens.unsqueeze(0).repeat(prompt_audio.size(0), 1)
        target_audio = torch.cat([pad_audio, prompt_audio], dim=1)

        # extract code codes
        target_audio_len = torch.tensor([target_audio.size(-1)] * target_audio.size(0), dtype=torch.long, device=self.device)
        with fp32_precision(), torch.no_grad():
            code, _ = self.audio_codec.encode(target_audio.unsqueeze(1), target_audio_len)

        # get context hidden 
        context_hidden_state = self.embed_tokens(input_text_tokens)

        # create masks
        subword_mask = torch.zeros_like(input_text_tokens) # subword_mask is all zeros because on the warmup there is only the prompt
        # audio mask is all ones except for description
        audio_mask = torch.ones_like(input_text_tokens) 
        audio_mask[:, :desc_tokens_ids.size(-1)] = 0
        # desc mask is all zeros except the description
        desc_mask = torch.zeros_like(input_text_tokens)
        desc_mask[:, :desc_tokens_ids.size(-1)] = 1
        # non_prompt_mask is all zeros, because all processed is prompt
        non_prompt_mask = torch.zeros_like(input_text_tokens) 

        # add special tokens on audio codes
        code = torch.where(
            desc_mask.unsqueeze(-1).bool(),                    # (B, T, 1) for broadcasting
            torch.full_like(code, self.speech_pad_id),  # fill with pad id
            code
        )

        # shift subword_ids
        # subword_ids = F.pad(input_text_tokens[:, 1:], [0, 1], value=current_subword_id)
        subword_ids = F.pad(input_text_tokens[:, 1:], [0, 1], value=0.0)


        init_inputs = {
            "code": code[:, :-1],
            "audio_mask": audio_mask.bool()[:, :-1],
            "context_hidden_state": context_hidden_state[:, :-1],
            "subword_ids": subword_ids[:, :-1],
            "subword_mask": subword_mask.bool()[:, :-1],
            "non_prompt_mask": non_prompt_mask.bool()[:, :-1],
        }

        return init_inputs


    def get_init_inputs_as_eartts(
        self,
        speaker_audio: torch.Tensor,        # [B, L]
        speaker_audio_lens: torch.Tensor,   # [B]
        system_prompt: str | None = None,
        user_prompt: str | None = None,
        description: str | None = None,
    ):
        """
        Create batch init_inputs for warmup, fully compatible with prepare_stream_inputs().
        Truncates the last frame so current_subword_id can be injected during autoregressive inference.
        """
        prompt_audio_size = int(((self.data_cfg.audio_prompt_duration * self.target_sample_rate) // self.target_samples_per_frame) * self.target_samples_per_frame)
        speaker_audio = speaker_audio[:, :prompt_audio_size]
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            B = speaker_audio.size(0)

            # ------------------------------
            # 1. Encode audio prompt for each batch
            # ------------------------------
            token_len_list = []
            audio_code_list = []

            for b_idx in range(B):
                wav = speaker_audio[b_idx : b_idx + 1]  # [1, L]
                if wav.ndim > 1:
                    wav = wav.mean(0, keepdim=True)

                token_len = math.ceil(wav.size(-1) / self.audio_codec.config.wav_to_token_ratio)
                pad_len = token_len * self.audio_codec.config.wav_to_token_ratio - wav.size(-1)
                padded_wav = F.pad(wav, (pad_len, 0)).to(self.device)
                wav_len = torch.tensor([padded_wav.size(-1)], dtype=torch.long, device=self.device)
                audio_code, _ = self.audio_codec.encode(padded_wav.unsqueeze(0), wav_len)

                token_len_list.append(token_len)
                audio_code_list.append(audio_code)
            max_audio_len = max(token_len_list)

            # ------------------------------
            # 2. Build prompts
            # ------------------------------
            # if self.tokenizer.chat_template is None:
            #    self.tokenizer.chat_template = default_chat_template

            if system_prompt is None:
                system_prompt = (
                    "You engage in conversation with the user. When delivering your response as speech, "
                    "if the user provides a description such as emotions, scene details, "
                    "or speaker style, you adjust your speaking style accordingly when delivering the response. "
                    "However, this description should influence only the delivery of your response, not its content. "
                    "Your response should remain independent of any stylistic instructions."
                )
            if user_prompt is None:
                user_prompt = "Can you tell me something interesting?"

            messages = [{"role": "system", "content": system_prompt}]
            full_user_prompt = f"```\n{description}\n```\n\n{user_prompt}" if description else user_prompt
            messages.append({"role": "user", "content": full_user_prompt})
            messages.append({"role": "assistant", "content": SCRIPT_PLACEHOLDER})

            non_script_list = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            ).split(SCRIPT_PLACEHOLDER + self.tokenizer.eos_token)[:-1]

            # ------------------------------
            # 3. Process text tokens + masks
            # ------------------------------
            input_ids, aligned_text_mask, aligned_audio_mask = [], [], []
            for i, non_script in enumerate(non_script_list):
                desc_ids = self.tokenizer.text_to_ids(non_script) # self.tokenizer.encode(non_script, add_special_tokens=False)
                if i == 0:
                    # include EOS after desc
                    # script_ids = [self.tokenizer.eos_token_id]
                    script_ids = [self.tokenizer.eos]
                    desc_len, text_len = len(desc_ids), len(desc_ids) + len(script_ids)
                    audio_len = desc_len + max_audio_len
                    input_ids.extend(desc_ids + script_ids)
                    aligned_text_mask.extend([1] * text_len + [0] * (audio_len - text_len))
                    aligned_audio_mask.extend([0] * (desc_len - 1) + [1] * (audio_len - desc_len) + [0])
                else:
                    desc_len = len(desc_ids)
                    input_ids.extend(desc_ids)
                    aligned_text_mask.extend([1] * desc_len)
                    aligned_audio_mask.extend([0] * (desc_len - 1) + [1])

            input_ids = torch.tensor(input_ids, dtype=torch.long, device=self.device).view(1, -1).repeat(B, 1)
            aligned_text_mask = torch.tensor(aligned_text_mask, dtype=torch.bool, device=self.device).view(1, -1).repeat(B, 1)
            aligned_audio_mask = torch.tensor(aligned_audio_mask, dtype=torch.bool, device=self.device).view(1, -1).repeat(B, 1)

            b, t = aligned_text_mask.size()

            # ------------------------------
            # 4. Align audio codes per batch
            # ------------------------------
            aligned_code = torch.full(
                (B, t, self.tts_model.config.num_quantizers),
                self.tts_model.config.codebook_size,
                dtype=torch.long,
                device=self.device,
            )

            for i in range(B):
                print(aligned_code.shape, aligned_audio_mask.shape)
                aligned_code[i].masked_scatter_(
                    F.pad(aligned_audio_mask[i][:-1], [0, 1]).unsqueeze(-1),
                    audio_code_list[i]
                )

            # ------------------------------
            # 5. Align LM hidden states
            # ------------------------------
            inputs_embeds = self.embed_tokens(input_ids)
            lm_hidden_state = inputs_embeds

            aligned_lm_hidden_state = torch.zeros(
                (B, t, lm_hidden_state.size(2)),
                dtype=lm_hidden_state.dtype,
                device=self.device
            )
            aligned_input_ids = torch.zeros((B, t), dtype=input_ids.dtype, device=self.device)
            for i in range(B):
                aligned_lm_hidden_state[i].masked_scatter_(aligned_text_mask[i].unsqueeze(-1), lm_hidden_state[i])
                aligned_input_ids[i].masked_scatter_(aligned_text_mask[i], input_ids[i])

            # ------------------------------
            # 6. Subword ids + masks
            # ------------------------------
            subword_ids = F.pad(aligned_input_ids[:, 1:], [0, 1], value=self.text_pad_id)
            subword_mask = torch.logical_and(
                F.pad(aligned_text_mask[:, 1:], [0, 1], value=True),
                aligned_audio_mask,
            )

            # ------------------------------
            # 7. Truncate last frame
            # ------------------------------
            init_inputs = {
                "code": aligned_code[:, :-1],
                "audio_mask": aligned_audio_mask[:, :-1],
                "context_hidden_state": aligned_lm_hidden_state[:, :-1],
                "subword_ids": subword_ids[:, :-1],
                "subword_mask": subword_mask[:, :-1],
                "non_prompt_mask": torch.zeros_like(aligned_text_mask[:, :-1]),  # warmup = all prompt
            }
            return init_inputs


    @torch.no_grad()
    def offline_inference(
        self,
        next_subword_ids: torch.Tensor,
        next_input_text_tokens: torch.Tensor,
        speaker_audio: torch.Tensor,
        speaker_audio_lens: torch.Tensor,
        formatter: str = "",
        system_prompt: str = None,
        user_prompt: str = None,
        guidance_enabled: bool = True,
        generation_config: dict = None,
        init_inputs: dict = None,
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
        B = speaker_audio.size(0)

        # init model
        init_inputs = self.get_init_inputs(speaker_audio, speaker_audio_lens, system_prompt=system_prompt, user_prompt=user_prompt)
        # init_inputs = self.get_init_inputs_as_eartts(speaker_audio, speaker_audio_lens, system_prompt=system_prompt, user_prompt=user_prompt)

        if generation_config is None:
            generation_config = self._get_generation_config(guidance_enabled)

        init_inputs.update({"use_cache": True, "past_key_values": None, "guidance_enabled": guidance_enabled})
        # warmup the model and generate the very first audio token
        outputs = self.tts_model(**init_inputs)
        code, _, _ = self.tts_model.generate_step(outputs.hidden_states[:, -1:], **generation_config)
        past_key_values = outputs.past_key_values
        # print(code.shape, code)
        # use the text tokens to stop generation
        max_steps = next_subword_ids.size(-1)
        # create variable to store the audios
        gen_audio_codes = torch.zeros(B, max_steps, self.tts_model.config.num_quantizers, device=self.device, dtype=torch.long)

        for i in range(max_steps-1):
            # current subword id is always seem
            current_subword_id = next_subword_ids[:, i].unsqueeze(-1)
            # get context_hidden_state it is always one step behind
            context_subword_id = next_input_text_tokens[:, i].unsqueeze(-1)
            context_hidden_state = self.embed_tokens(context_subword_id)

            # create subword_mask
            current_subword_mask = (current_subword_id != self.text_pad_id).bool()
            # print(i, current_subword_mask.shape, current_subword_mask.shape)
            # get subword_ids
            inputs = {
                "code": code,
                "context_hidden_state": context_hidden_state,
                "subword_ids": current_subword_id,
                # "subword_mask": current_subword_mask, # ToDo: implement subword_mask it will be required here for S2S
                "past_key_values": past_key_values,
                "use_cache": True,
                "guidance_enabled": guidance_enabled,
                "generation_config": generation_config,
                "ignore_eos_flag_stop": True,
            }

            outputs = self.tts_model(**inputs)

            code = outputs.codes
            past_key_values = outputs.past_key_values
            gen_audio_codes[:, i-1] = code.squeeze(1)


        gen_audio_codes_lens = torch.tensor([gen_audio_codes.shape[1]] * gen_audio_codes.shape[0]).to(self.device)
        # decode audio
        gen_audio_codes = replace_control_speech_codes(gen_audio_codes, self._control_codes, self.codec_silence_tokens)
        with fp32_precision(), torch.no_grad():
            audio_pred, audio_len = self.audio_codec.decode(
                gen_audio_codes, gen_audio_codes_lens
            )

        return audio_pred.squeeze(1), audio_len


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
