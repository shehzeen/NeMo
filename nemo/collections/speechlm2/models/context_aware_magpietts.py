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
from nemo.collections.tts.parts.utils.helpers import get_mask_from_lengths

from nemo.collections.asr.models import EncDecSpeakerLabelModel

from nemo.collections.audio.parts.utils.resampling import resample
from nemo.core.classes.module import NeuralModule
from nemo.collections.common.tokenizers import AutoTokenizer
from nemo.collections.nlp.parts.nlp_overrides import NLPSaveRestoreConnector
from nemo.collections.speechlm2.data.utils import get_pad_id
from nemo.collections.speechlm2.models.duplex_s2s_model import replace_control_speech_codes, tokens_to_str
from nemo.collections.speechlm2.parts.hf_hub import HFHubMixin
from nemo.collections.speechlm2.parts.lora import maybe_install_lora
from nemo.collections.speechlm2.parts.metrics.asr_bleu import ASRBLEU
from nemo.collections.speechlm2.parts.metrics.bleu import BLEU
from nemo.collections.speechlm2.parts.metrics.results_logger import ResultsLogger
from nemo.collections.speechlm2.parts.metrics.token_accuracy import TokenAccuracy
from nemo.collections.speechlm2.parts.optim_setup import configure_optimizers, is_frozen
from nemo.collections.speechlm2.parts.precision import fp32_precision
from nemo.collections.speechlm2.parts.pretrained import (
    load_pretrained_hf,
    set_model_dict_for_partial_init,
    setup_audio_codec,
    setup_speech_encoder,
)
from nemo.core.neural_types import AudioSignal, LabelsType, LengthsType, NeuralType
from nemo.utils import logging

from nemo.collections.tts.modules import transformer_2501


def build_vocabs(subword_vocab: dict, subword_padding_idx: int, special_vocab: dict = None) -> tuple[dict, dict]:
    """
    Builds the character vocabulary and the mapping from subword ids to character ids.
    Args:
        subword_vocab (dict): A dictionary of subword vocab items. Eg.
            tokenizer = AutoTokenizer.from_pretrained(pretrained_tokenizer_name)
            subword_vocab = tokenizer.vocab
        subword_padding_idx (int): The padding index for the subword vocabulary.
        special_vocab (dict): items of special token dictionary (usually BOS, EOS)
            eg. special_vocab = {'<BOS>': 0, '<EOS>': 1}
    Returns:
        subword_id_to_char_ids: A dictionary mapping subword ids to character ids.
        char_vocab: A dictionary mapping character ids to their corresponding characters.
    """
    subword_vocab_items = subword_vocab.items() if isinstance(subword_vocab, dict) else subword_vocab
    org_char_vocab = {subword: subword_id for subword, subword_id in  subword_vocab_items if len(subword) == 1}

    # Add special tokens directly to char vocab
    if special_vocab is not None:
        for special_token, special_token_id in special_vocab.items():
            if special_token in org_char_vocab:
                raise ValueError(f"Special token {special_token} already exists in the character vocabulary.")
            org_char_vocab[special_token] = special_token_id

    sorted_char_vocab = dict(sorted(org_char_vocab.items(), key=lambda x: x[1]))
    char_vocab = {k: i for i, (k, _) in enumerate(sorted_char_vocab.items())}
    assert sorted(char_vocab.values()) == list(range(len(char_vocab)))
    subword_id_to_char_ids = {
        subword_id: tuple(char_vocab[char] for char in subword) for subword, subword_id in subword_vocab_items
    }

    # Creating mapping from subword ids of special tokens to their char ids
    if special_vocab is not None:
        for special_token, special_token_id in special_vocab.items():
            if special_token in subword_id_to_char_ids:
                raise ValueError(f"Special token {special_token} already exists in the subword id Vocabulary.")
            subword_id_to_char_ids[special_token_id] = (char_vocab[special_token],)

    assert max(subword_id_to_char_ids) == len(subword_id_to_char_ids) - 1

    # Always add padding token to the end of the vocab (this is the convention used in the original code)
    subword_id_to_char_ids[subword_padding_idx] = (len(char_vocab),)

    return subword_id_to_char_ids, char_vocab

class CharAwareSubwordEncoder(NeuralModule):
    """
    Char-aware subword encoder for the MagpieTTS model.
    This module takes subword ids as input, maps them to character ids, and then applies a transformer encoder to the character embeddings.
    The output is a tensor of shape (batch_size, max_subword_length, d_embed).
    """
    def __init__(self, d_embed: int, llm_tokenizer_vocab: dict, subword_padding_idx: int, special_vocab: dict = None):
        """
        Args:
            d_embed (int): The dimension of the embedding.
            llm_tokenizer_vocab (dict): A dictionary of subword vocab items. Eg.
                tokenizer = AutoTokenizer.from_pretrained(pretrained_tokenizer_name)
                llm_tokenizer_vocab = tokenizer.vocab
            subword_padding_idx (int): The padding index for the subword vocabulary.
            special_vocab (dict): items of special token dictionary (usually BOS, EOS)
                eg. special_vocab = {'<BOS>': 30001, '<EOS>': 30002}
        """
        super().__init__()
        self.subword_id_to_char_ids, self.char_vocab = build_vocabs(llm_tokenizer_vocab, subword_padding_idx, special_vocab)
        self.embed_tokens = torch.nn.Embedding(self.vocab_size+1, d_embed, padding_idx=self.vocab_size)
        self.encoder = transformer_2501.Transformer(
            n_layers=1,
            d_model=d_embed,
            d_ffn=d_embed * 4,
            sa_n_heads=8,
            kernel_size=1,
            max_length_causal_mask=256,
            use_learnable_pos_emb=True
        )

    @property
    def vocab_size(self):
        return len(self.char_vocab)

    def prepare_inputs(self, subword_ids: Tensor, padding_mask: Tensor) -> tuple[Tensor, Tensor]:
        device = subword_ids.device

        subword_id_list = torch.masked_select(subword_ids, padding_mask).cpu().tolist()
        char_id_list = [list(self.subword_id_to_char_ids[x]) for x in subword_id_list]

        char_lengths = torch.tensor([len(x) for x in char_id_list], dtype=torch.long, device=device)
        batch_size = char_lengths.size(0)

        char_ids = torch.full((batch_size, int(char_lengths.max().item())), self.vocab_size, dtype=torch.long)
        for i in range(batch_size):
            char_ids[i, : char_lengths[i]] = torch.tensor(char_id_list[i])
        char_ids = char_ids.to(device=device)
        return char_ids, char_lengths

    def forward(self, subword_ids: Tensor, subword_mask: Tensor | None = None) -> Tensor:
        """
        Args:
            subword_ids (Tensor): A tensor of shape (batch_size, max_subword_length) containing the subword ids.
            subword_mask (Tensor | None): A tensor of shape (batch_size, max_subword_length) containing the mask for the subword ids.
                If None, a mask of ones will be used.
        Returns:
            Tensor: A tensor of shape (batch_size, max_subword_length, d_embed) containing the subword embeddings.
        """
        device = subword_ids.device
        if subword_mask is None:
            subword_mask = torch.ones_like(subword_ids).bool()
        else:
            subword_mask = subword_mask.bool()

        if subword_mask.ndim == 3:
            subword_mask = subword_mask.squeeze(-1)

        char_ids, char_lengths = self.prepare_inputs(subword_ids, subword_mask)
        char_mask = get_mask_from_lengths(char_lengths)
        char_emb = self.embed_tokens(char_ids)
        # char emb has the shape  [B*T, N, channels], where N is the max number of chars tokens decoded from bpe tokens
        x = self.encoder(
            x=char_emb,
            x_mask=char_mask
        )['output']

        # Get average embedding over the chars
        mean_emb = ((x / char_mask.unsqueeze(-1).sum(1, keepdim=True)) * char_mask.unsqueeze(-1)).sum(1)
        subword_emb = torch.zeros((subword_mask.size(0), subword_mask.size(1), mean_emb.size(-1)), device=device)
        subword_emb[subword_mask.unsqueeze(-1).expand(-1, -1, mean_emb.size(-1))] = mean_emb.view(-1)
        
        return subword_emb

class ContextAwareMagpieTTS(LightningModule, HFHubMixin):
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
        self.use_bpe_char_tokenizer = self.cfg.get("use_bpe_char_tokenizer", False)
        self.condition_spk_emb_on_bos_position = self.cfg.get("condition_spk_emb_on_bos_position", False)
        self.cfg_unconditional_prob = self.cfg.get('cfg_unconditional_prob', 0.0)
        self.cfg_scale = self.cfg.get('cfg_scale', None)

        # codec configs
        setup_audio_codec(self)
        self._codebook_size = self.audio_codec.vector_quantizer.codebook_size_per_group
        self._num_codebooks = self.audio_codec.vector_quantizer.num_groups

        # compute target fps
        self.target_fps = self.target_sample_rate / self.audio_codec.samples_per_frame

        # Load tokenizer
        self.tokenizer = AutoTokenizer(self.cfg.pretrained_llm, use_fast=True)
        if 'Qwen2.5' in self.cfg.pretrained_llm:
            # For Qwen, '<|im_start|>' is a common choice for a BOS token.
            # You can check your tokenizer's vocabulary for the best candidate.
            logging.warning("Tokenizer does not have a `bos_token`. Setting it to '<|im_start|>'.")
            self.tokenizer.bos_token = '<|im_start|>'
            self.tokenizer.eos_token = '<|im_end|>'

        # Load ForCausalLM
        llm = load_pretrained_hf(self.cfg.pretrained_llm, pretrained_weights=self.cfg.pretrained_weights).train()
        self.decoder = llm.model  # fetch PretrainedBaseModel from model "ForCausalLM"

        # Note: we have to "move out" the token embedding outside of LLM to avoid
        #       messing up FSDP/TP hooks.
        self.embed_text_tokens = self.decoder.embed_tokens
        del self.decoder.embed_tokens

        # audio embeddings
        audio_embeddings = []
        for _ in range(self._num_codebooks):
            audio_embeddings.append(nn.Embedding(self.speech_vocab_size, self.decoder.config.hidden_size))
        self.audio_embeddings = nn.ModuleList(audio_embeddings)

        # audio head
        self.final_proj = nn.Linear(self.decoder.config.hidden_size, self._num_codebooks * self.speech_vocab_size)

        # add loss        
        self.cross_entropy_loss = nn.CrossEntropyLoss(reduction='none')

        # use BPE char aware tokenizer
        if self.use_bpe_char_tokenizer:
            llm_tokenizer_vocab_items = self.tokenizer.vocab
            # if vocab is a dict it already has the subword and token id, if not, get it from the tokenizer
            if isinstance(llm_tokenizer_vocab_items, dict):
                llm_tokenizer_vocab_items = llm_tokenizer_vocab_items.items()
            else:
                llm_tokenizer_vocab_items = [
                    (subword, self.tokenizer.tokenizer._tokenizer.token_to_id(subword))
                    for subword in llm_tokenizer_vocab_items
                ]

            self.cas_encoder = CharAwareSubwordEncoder(
                d_embed=self.decoder.config.hidden_size,
                llm_tokenizer_vocab=llm_tokenizer_vocab_items,
                subword_padding_idx=self.tokenizer.pad
            )

        # init speaker encoder
        self.speaker_encoder_model_name = self.cfg.get("speaker_encoder_model_name", 'titanet_large')

        if self.condition_spk_emb_on_bos_position:
            self.max_speaker_reference_len = self.cfg.get("max_speaker_reference_len", 5)
            # setup speaker encoder
            self.setup_speaker_encoder()
            # speaker encoder projection
            self.speaker_encoder_emb_projection = nn.Linear(self.cfg.get("speaker_embedding_dim", 192), self.decoder.config.hidden_size)

        # cached for quicker audio decoding
        self.register_buffer(
            "_control_codes",
            torch.tensor([self.speech_bos_id, self.speech_eos_id, self.speech_delay_id], device=self.device),
        )
        self._use_fsdp = False
        self._use_tp = False

        # load pretrained TTS model
        if self.cfg.get("pretrained_tts", None):
            self.init_model_from_tts_checkpoint(self.cfg.pretrained_tts)

    def setup_speaker_encoder(self):
        with fp32_precision():
            self.speaker_encoder = EncDecSpeakerLabelModel.from_pretrained(model_name=self.speaker_encoder_model_name)

        # freeze the pretrained speaker encoder
        self.speaker_encoder.eval()
        self.speaker_encoder.freeze()

        for p in self.speaker_encoder.parameters():
            p.requires_grad = False

    def init_model_from_tts_checkpoint(self, checkpoint_path):
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
    def speech_vocab_size(self):
        """Return the size of the audio codec codebook including extra speech BOS and EOS tokens."""
        return self._codebook_size + 3

    @property
    def speech_bos_id(self) -> int:
        """Indicates start of utterance generation (not start of inference!)."""
        if self.cfg.get("custom_speech_bos_id", None):
            return self.cfg.get("custom_speech_bos_id")
        return self._codebook_size

    @property
    def speech_eos_id(self) -> int:
        """Indicates end of utterance generation."""
        if self.cfg.get("custom_speech_eos_id", None):
            return self.cfg.get("custom_speech_eos_id")
        return self._codebook_size + 1

    @property
    def speech_delay_id(self) -> int:
        """Indicates start of inference (the very first frame)."""
        if self.cfg.get("custom_speech_delay_id", None):
            return self.cfg.get("custom_speech_delay_id")
        return self._codebook_size + 2

    @property
    def text_vocab_size(self):
        """Return the size of the text tokenizer."""
        return self.tokenizer.vocab_size

    @property
    def text_bos_id(self) -> int:
        return self.tokenizer.bos_id

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

    def get_speaker_embedding(self, audio, audio_len, sr):
        # limit max audio len to avoid memory waste
        audio = audio[:, : int(self.max_speaker_reference_len*sr)]
        with torch.autocast(device_type="cuda", dtype=torch.float32):
            with torch.no_grad():
                model_sr = self.speaker_encoder._cfg.train_ds.get('sample_rate', 16000)
                audio_resampled = torchaudio.functional.resample(audio, sr, model_sr)
                audio_len_resampled = audio_len * (model_sr / sr )
                _, g = self.speaker_encoder(input_signal=audio_resampled, input_signal_length=audio_len_resampled.long())
                g = g.unsqueeze(1)
        return g

    def embed_audio_tokens(self, audio_tokens):
        audio_tokens = audio_tokens.transpose(1, 2).contiguous()
        # Add and average the embeddings of the audio tokens across the codebooks
        audio_embedding = None
        for c in range(self._num_codebooks):
            embedding = self.audio_embeddings[c](audio_tokens[:, c, :])
            if audio_embedding is None:
                audio_embedding = embedding
            else:
                audio_embedding = audio_embedding + embedding
        audio_embedding = audio_embedding / audio_tokens.size(1)
        return audio_embedding

    def forward(
        self,
        input_embeds: Tensor,
        cache=None,
        seq_mask=None,
    ) -> dict[str, Tensor]:
        """
        Separated text and speech prediction:
            - Speech prediction is achieved by a independent AR decoder based on last_hidden_state + audio tokens
            - For KV-cache:
                (1) llm cache depends on input cache is None or Not
                (2) speech_generation cache relys on reset_input_and_kv_cache function.
        """

        if self.cfg_unconditional_prob:
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

        out = self.decoder(
            inputs_embeds=input_embeds,
            attention_mask=seq_mask,
            past_key_values=cache, use_cache=cache is not None, return_dict=True
        )
        B, T = input_embeds.shape[:2]

        # get logits
        logits = self.final_proj(out.last_hidden_state)  # (B, T', num_codebooks * _codebook_size)

        # if using cfg and it is in inference or evaluation mix unconditional and coditional logits
        if self.cfg_scale is not None and self.cfg_unconditional_prob and not self.training:
            batch_size = logits.size(0) // 2
            cond_logits = logits[:batch_size]
            uncond_logits = logits[batch_size:]
            logits = (1 - self.cfg_scale) * uncond_logits + self.cfg_scale * cond_logits
        ans = {
            "logits": logits,
        }
        if cache is not None:
            ans["cache"] = out["past_key_values"]

        return ans

    def prepare_inputs(self, batch: dict):
        """
        Similar to DuplexS2SModel.prepare_inputs, with following changes:
            (1) Add 'input_audio_tokens' and 'seq_mask' in return value for TransformerARSpeechDecoder
            (2) Remove audio codec embedding from 'input_embeds'
        """
        # check if audios has the same batch size
        assert batch["source_audio"].size(0) == batch["target_audio"].size(0)
        assert batch["target_first_turn_audio"].size(0) == batch["target_audio"].size(0)

        # change audio volume randomly
        if self.training and random.random() < self.cfg.get('noise_prob_scale_user', 0.0):
            # prev codebase had 0.0631 and 5.6234 here we round the values
            min_scale_val = self.cfg.get('noise_scale_user_min', 0.0631)  # -15 snr
            max_scale_val = self.cfg.get('noise_scale_user_min', 5.6234)  # 24 snr

            # get a random float value between min and max
            scaling_factor = (
                torch.rand(batch["source_audio"].size(0), device=batch["source_audio"].device)
                * (max_scale_val - min_scale_val)
                + min_scale_val
            )
            batch["source_audio"] = batch["source_audio"] * scaling_factor.unsqueeze(-1)

        # apply low pass filter
        if self.training and random.random() < self.cfg.get('noise_prob_low_pass', 0.0):
            cutoff_freq = self.cfg.get('noise_low_pass_cutoff_freq', 1000.0)
            # note here we are using a biquad filter, older codebase we are using a filter of order 5
            batch["source_audio"] = torchaudio.functional.lowpass_biquad(
                waveform=batch["source_audio"], sample_rate=self.source_sample_rate, cutoff_freq=cutoff_freq
            )

        # extract target audio codes
        with fp32_precision(), torch.no_grad():
            target_codes, target_codes_lens = self.audio_codec.encode(
                audio=batch["target_audio"], audio_len=batch["target_audio_lens"]
            )
            target_codes = target_codes.transpose(1, 2)  # (B, K, T) -> (B, T, K)

        source_audio = batch["source_audio"]
        source_audio_lens = batch["source_audio_lens"]
        if self.source_sample_rate != self.target_sample_rate:
            source_audio = resample(source_audio, self.source_sample_rate, self.target_sample_rate)
            source_audio_lens = source_audio_lens * self.target_sample_rate/self.source_sample_rate
        
        # ToDo: Add a transformer encoder to help the model to better extract contextual information, replace the code bellow with it
        # extract embedding for context audios
        with fp32_precision(), torch.no_grad():
            source_codes, source_codes_lens = self.audio_codec.encode(
                audio=source_audio, audio_len=source_audio_lens
            )
            source_codes = source_codes.transpose(1, 2)  # (B, K, T) -> (B, T, K)

        # source codes to embeddings
        source_audio_emb = self.embed_audio_tokens(
            source_codes
        )

        target_tokens = batch["target_tokens"]
        if (diff := target_tokens.shape[1] - source_audio_emb.shape[1]) < 0:
            target_tokens = torch.cat(
                [
                    target_tokens,
                    (
                        torch.ones(source_audio_emb.shape[0], abs(diff), device=source_audio_emb.device) * self.text_pad_id
                    ).to(torch.long),
                ],
                dim=-1,
            )
        elif diff > 0:
            target_tokens = target_tokens[:, : source_audio_emb.shape[1]]

        if (tl := target_codes.shape[1]) != (sl := source_audio_emb.shape[1]):
            if tl < sl:
                diff = sl - tl
                source_audio_emb = source_audio_emb[:, :tl]
                target_tokens = target_tokens[:, :tl]
                torch.clamp_(source_codes_lens, max=tl)
            else:
                diff = tl - sl
                target_codes = target_codes[:, :sl]
                torch.clamp_(target_codes_lens, max=sl)
            if diff > 2:
                logging.warning(
                    f"A mismatch between source ({sl}) and target ({tl}) sequence length greater than 2 detected. "
                    f"This may indicate significant desynchronization in longer sessions."
                )

        btt = target_tokens[..., None]
        target_codes = torch.where(btt == self.text_bos_id, self.speech_bos_id, target_codes)
        target_codes = torch.where(btt == self.text_eos_id, self.speech_eos_id, target_codes)

        # Add delay token
        target_codes = torch.cat(
            [
                torch.full(
                    [target_codes.shape[0], 1, target_codes.shape[-1]],
                    fill_value=self.speech_delay_id,
                    device=self.device,
                    dtype=torch.long,
                ),
                target_codes[:, :-1],
            ],
            dim=1,
        )

        # move back text channel by x, in inference it advance the text channel prediction
        # it is the oposite of speech delay applied on text channel
        if self.advance_text_channel_by:
            pad = torch.full(
                (target_tokens.shape[0], self.advance_text_channel_by),
                fill_value=self.text_pad_id,
                device=target_tokens.device,
                dtype=torch.long,
            )
            target_tokens = torch.cat([target_tokens[:, self.advance_text_channel_by :], pad], dim=-1)
            # make sure that eos/bos is in the place (it can cut tokens from the first advance_text_channel_by tokens and this will breaks everything)

        # target codes to embeddings
        target_audio_emb = self.embed_audio_tokens(
            target_codes
        )

        # create sequence mask
        seq_mask = get_mask_from_lengths(target_codes_lens)

        if self._use_tp:
            tp_world_size = self.device_mesh["tensor_parallel"].size()
            if (remainder := (target_tokens.shape[1] - 1) % tp_world_size) != 0:
                target_tokens = target_tokens[:, :-remainder]
                source_codes = source_codes[:, :-remainder]
                target_codes = target_codes[:, :-remainder]
                source_audio_emb = source_audio_emb[:, :-remainder]
                target_audio_emb = target_audio_emb[:, :-remainder]
                seq_mask = seq_mask[:, :-remainder]

        # text labels -- tts input and audio labels are aligned
        text_labels = target_tokens[:, 1:]  # (B, T-1)
        audio_labels = target_codes[:, 1:]  # (B, T-1, K)
        # shift source and target codes embeddings for the autoregressive training
        source_audio_emb = source_audio_emb[:, :-1] # (B, T-1, K)
        target_audio_emb = target_audio_emb[:, :-1] # (B, T-1, K)
        seq_mask = seq_mask[:, :-1]

        # Drop EOS tokens with per-token probability (augmentation)
        drop_eos_prob = self.cfg.get("drop_text_eos_prob", 0.0)
        if drop_eos_prob > 0.0 and self.training:
            eos_mask = (text_labels == self.text_eos_id)
            drop_eos_mask = torch.rand_like(text_labels, dtype=torch.float) < drop_eos_prob
            text_labels = torch.where(eos_mask & drop_eos_mask, self.text_pad_id, text_labels)

        # Add source codes embeddings
        input_embeds = source_audio_emb

        # get embedding for text tokens
        text_embedded = self.embed_text_tokens(text_labels)

        # if use bpe char tokenizer sum the embeddings
        if self.use_bpe_char_tokenizer:
            cas_embedding = self.cas_encoder(text_labels, subword_mask=seq_mask)  # (B, L, E)
            text_embedded = text_embedded + cas_embedding

        # Add text to the model input
        input_embeds.add_(text_embedded)

        # Add shifted target codes embeddings
        input_embeds.add_(target_audio_emb)

        # ToDo: Add on speaker embedding per turn to make possible to have multi speaker as agent
        speaker_encoder_emb = None
        if self.condition_spk_emb_on_bos_position:
            # handle speaker conditioning - extract speaker embedding
            target_first_turn_audio = batch["target_first_turn_audio"]
            target_first_turn_audio_lens = batch["target_first_turn_audio_lens"]
            speaker_encoder_emb = self.get_speaker_embedding(
                target_first_turn_audio, target_first_turn_audio_lens, self.target_sample_rate
            ).to(target_audio_emb.dtype)
            # project speaker embedding to match llm input size
            speaker_embedding_projected = self.speaker_encoder_emb_projection(speaker_encoder_emb)  # [B, D]
            # Expand the speaker embedding to match input_embeds at BOS positions
            bos_mask = (text_labels == self.text_bos_id).unsqueeze(-1)  # [B, T, 1]
            # Expand speaker embedding to match [B, T, D]
            speaker_embedding_projected_expanded = speaker_embedding_projected #.repeat(1, input_embeds.size(1), 1)  # [B, T, D]
            # Replace BOS positions in input_embeds with speaker embedding
            input_embeds = torch.where(bos_mask, speaker_embedding_projected_expanded, input_embeds)

        # debug samples:
        if (
            self.cfg.get("debug_dataloader_audios_path", None)
            and self.training
            and "s2s_duplex_overlap_as_s2s_duplex" in batch["formatter"][0]
        ):

            def count_leading_silence_tokens(tensor: torch.Tensor, silence_token: int = 0) -> int:
                """
                Count the number of consecutive silence tokens at the beginning of a 1D tensor.

                Args:
                    tensor (torch.Tensor): 1D tensor of tokens.
                    silence_token (int): The token considered as silence (default: 0).

                Returns:
                    int: Number of consecutive silence tokens at the beginning.
                """
                if tensor.ndim != 1:
                    raise ValueError("Input tensor must be 1D.")

                count = 0
                for token in tensor:
                    if token.item() == silence_token:
                        count += 1
                    else:
                        break
                return count

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
                    self.audio_codec.device
                )
                reconstructed_audio_from_wav, _ = self.audio_codec(audio=batch["target_audio"], audio_len=lengths)
                # reconstruct wav
                audio_labels_ = replace_control_speech_codes(audio_labels, self._control_codes)
                with fp32_precision(), torch.no_grad():
                    lengths = torch.tensor([audio_labels_.shape[1]] * audio_labels_.shape[0]).to(
                        self.audio_codec.device
                    )
                    reconstructed_audio_from_tokens, _ = self.audio_codec.decode(
                        tokens=audio_labels_.transpose(1, 2), tokens_len=lengths
                    )

            for i in range(audio_labels_.shape[0]):
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

                write_wave(
                    reconstructed_audio_from_wav[i],
                    os.path.join(
                        self.cfg.get("debug_dataloader_audios_path"),
                        f"target_audio_reconstructed_from_waveform_{i}.wav",
                    ),
                    sr=self.target_sample_rate,
                )
                if self.cfg.get("use_eou_decoder", None) or self.cfg.get("llm_predict_eou", None):
                    repeat_factor = int(self.target_sample_rate / self.target_fps)
                    eou_wav = (
                        eou_labels[i].unsqueeze(0).unsqueeze(-1).repeat(1, 1, repeat_factor)
                    )  # (B, T, repeat_factor)
                    eou_wav = eou_wav.view(1, -1)  # (B, T * repeat_factor)
                    eou_wav = eou_wav.float() * 0.8  #  make 1 audible and keep 0 as total silence
                    write_wave(
                        eou_wav.squeeze(),
                        os.path.join(self.cfg.get("debug_dataloader_audios_path"), f"eou_{i}.wav"),
                        sr=self.target_sample_rate,
                    )

            num_bos_tokens = (text_labels.unsqueeze(-1) == self.text_bos_id).flatten(1, 2).sum(-1)
            # Count how many EOS tokens are present per sequence
            # Shape: [B]
            num_eos_tokens = (text_labels.unsqueeze(-1) == self.text_eos_id).flatten(1, 2).sum(-1)
            print("Num eos:", num_eos_tokens, "num bos:", num_bos_tokens)
            # check text
            print(
                "text_labels decoded:",
                tokens_to_str(
                    text_labels[-1:], target_codes_lens - 1, tokenizer=self.tokenizer, pad_id=self.text_pad_id
                ),
            )
            print(
                "target labels from dataloader decoded:",
                tokens_to_str(
                    batch["target_tokens"][-1:],
                    target_codes_lens - 1,
                    tokenizer=self.tokenizer,
                    pad_id=self.text_pad_id,
                ),
            )
            print(
                "Number of padding tokens on the begining:",
                count_leading_silence_tokens(text_labels[-1:].squeeze(), self.text_pad_id),
            )

            print(batch["formatter"])
            if audio_labels_.shape[0] > 1:
                exit()

        return {
            "input_embeds": input_embeds,
            "input_lens": source_codes_lens - 1,
            "output_lens": target_codes_lens - 1,
            "text_tokens": text_labels,
            "audio_labels": audio_labels,
            "seq_mask": seq_mask,
            "speaker_encoder_emb": speaker_encoder_emb,
        }

    def compute_loss(self, logits, audio_codes, audio_codes_lens, mask_tokens_mask=None, loss_mask=None):
        """
        Computes the audio codebook loss. Used by
        (1) The main Magpie-TTS transformer
        (2) The local transformer, for both autoregressive and MaskGit methods
        
        logits: (B, T', num_codebooks * _codebook_size)
        audio_codes: (B, T', C)
        audio_codes_lens: (B,)
        mask_tokens_mask: (B, C, T') True for tokens that were replaced with the MASK_TOKEN and should
                                     therefore be the only ones included in the loss computation.
        """
        audio_codes = audio_codes.transpose(1, 2)

        if loss_mask is None:
            loss_mask = get_mask_from_lengths(audio_codes_lens)

        if mask_tokens_mask is not None:
            # For MaskGit we only compute loss for the masked tokens.
            # *Both* conditions must be true:
            # 1. the token is masked
            # 2. the token is not padding
            loss_mask = loss_mask.unsqueeze(1) * mask_tokens_mask
            if not loss_mask.any():
                # Without this we were very rarely getting NaNs in the loss
                logging.warning("No tokens valid were found in compute_loss()!")
                return torch.tensor(0.0, device=loss_mask.device), loss_mask 
        else:            
            # repeat loss mask for each codebook to simplify code below
            loss_mask = loss_mask.unsqueeze(1).repeat(1, audio_codes.size(1), 1)
        total_codebook_loss = None
        for codebook in range(audio_codes.size(1)):
            si = codebook * self.speech_vocab_size
            ei = si + self.speech_vocab_size
            codebook_logits = logits[:, :, si:ei]  # (B, T', _codebook_size)
            codebook_targets = audio_codes[:, codebook]  # (B, T')

            codebook_loss = self.cross_entropy_loss(
                codebook_logits.permute(0, 2, 1), codebook_targets  # (B, _codebook_size, T')
            )  # (B, T')
            codebook_loss = codebook_loss * loss_mask[:, codebook, :]
            codebook_loss = codebook_loss.sum() / loss_mask[:, codebook, :].sum()
            if total_codebook_loss is None:
                total_codebook_loss = codebook_loss
            else:
                total_codebook_loss = total_codebook_loss + codebook_loss

        total_codebook_loss = total_codebook_loss / audio_codes.size(1)
        return total_codebook_loss, loss_mask

    def training_step(self, batch: dict, batch_idx: int):
        for m in (self.decoder, self.embed_text_tokens, self.audio_embeddings, self.final_proj):
            if is_frozen(m):
                m.eval()

        if self.condition_spk_emb_on_bos_position:
            for m in (self.speaker_encoder_emb_projection, self.speaker_encoder):
                if is_frozen(m):
                    m.eval()

        inputs = self.prepare_inputs(batch)

        forward_outputs = self(
            inputs["input_embeds"],
            seq_mask=inputs["seq_mask"]
        )

        codebook_loss, loss_mask = self.compute_loss(forward_outputs["logits"], inputs["audio_labels"],  inputs["output_lens"], loss_mask=inputs["seq_mask"])
        loss = codebook_loss

        # ToDo: Add local transformer losses
        B, T = inputs["input_embeds"].shape[:2]
        num_frames = inputs["input_lens"].sum()
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
        setup_audio_codec(self)  # potentially reloads the audio codec to make sure it's in fp32
        if self.condition_spk_emb_on_bos_position:
            self.setup_speaker_encoder()  # potentially reloads the speaker encoder to make sure it's in fp32

    def on_validation_epoch_start(self) -> None:
        self.on_train_epoch_start()
        self.results_logger = ResultsLogger(self.validation_save_path).reset()
        self.asr_bleu = ASRBLEU(self.cfg.scoring_asr).reset()

    def on_validation_epoch_end(self, prefix="val") -> None:
        asr_bleu = self.asr_bleu.compute()
        for k, m in asr_bleu.items():
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
                text_tokens=dataset_batch["target_tokens"],
            )

            with fp32_precision():  # resample is fragile to bfloat16 default dtype
                asr_hyps = self.asr_bleu.update(
                    name=name,
                    refs=dataset_batch["target_texts"],
                    pred_audio=resample(results["audio"], self.target_sample_rate, self.source_sample_rate),
                    pred_audio_lens=(results["audio_len"] / self.target_sample_rate * self.source_sample_rate).to(torch.long),
                )

                self.results_logger.update(
                    name=name,
                    refs=dataset_batch["target_texts"],
                    hyps=results["text"],
                    asr_hyps=asr_hyps,
                    samples_id=dataset_batch['sample_id'],
                    pred_audio=results["audio"],
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

    def sample_codes_from_logits(self, all_code_logits_t, temperature=0.7, topk=80, unfinished_items={}, finished_items={}):
        # all_code_logits_t: (B, num_codebooks * _codebook_size), logits at a given timestep
        all_preds = []
        for idx in range(self._num_codebooks):
            si = idx * self.speech_vocab_size
            ei = si + self.speech_vocab_size
            codebook_logits = all_code_logits_t[:, si:ei]  # (B, _codebook_size)
            for item_idx in unfinished_items:
                codebook_logits[item_idx, self.speech_eos_id] = float('-inf')
            for item_idx in finished_items:
                codebook_logits[item_idx, :] = float('-inf')
                codebook_logits[item_idx, self.speech_eos_id] = 0.0
            codebook_logits_topk = torch.topk(codebook_logits, topk, dim=-1)[0]  # (B, topk)
            indices_to_remove = codebook_logits < codebook_logits_topk[:, -1].unsqueeze(
                -1
            )  # (B, _codebook_size)
            codebook_logits_rescored = codebook_logits.clone()
            codebook_logits_rescored[indices_to_remove] = float('-inf')

            codebook_probs = torch.softmax(codebook_logits_rescored / temperature, dim=-1)  # (B, _codebook_size)
            codebook_preds = torch.multinomial(codebook_probs, 1)  # (B, 1)
            all_preds.append(codebook_preds)
        all_preds = torch.cat(all_preds, dim=1).long()  # (B, num_codebooks)
        return all_preds

    @torch.no_grad()
    def offline_inference(
        self,
        source_audio: torch.Tensor,
        source_audio_lens: torch.Tensor,
        speaker_audio: torch.Tensor,
        speaker_audio_lens: torch.Tensor,
        text_tokens: torch.Tensor,
        decode_audio: bool = True,
        temperature: float = 0.7, 
        topk: int = 80,
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

        # get context audio embedding for the whole audio
        # make sure that the audio is in target sampling rate
        if self.source_sample_rate != self.target_sample_rate:
            source_audio = resample(source_audio, self.source_sample_rate, self.target_sample_rate)
            source_audio_lens = source_audio_lens * self.target_sample_rate/self.source_sample_rate

        # ToDo: Add a transformer encoder to help the model to better extract contextual information, replace the code bellow with it
        # extract embedding for context audios
        with fp32_precision(), torch.no_grad():
            source_codes, lengths = self.audio_codec.encode(
                audio=source_audio, audio_len=source_audio_lens
            )
            source_codes = source_codes.transpose(1, 2)  # (B, K, T) -> (B, T, K)

        # source codes to embeddings
        source_audio_emb = self.embed_audio_tokens(
            source_codes
        )

        text_mask = get_mask_from_lengths(lengths)

        # make sure text tokens and context have the same size
        if source_audio_emb.size(1)> text_tokens.size(1):
            source_audio_emb = source_audio_emb[:, :text_tokens.size(1), :]
        else:
            text_tokens = text_tokens[:, :source_audio_emb.size(1)]

        # get speaker embedding
        if self.condition_spk_emb_on_bos_position:
            speaker_emb = self.get_speaker_embedding(
                speaker_audio, speaker_audio_lens, self.target_sample_rate
            ).to(source_audio_emb.dtype).to(source_audio_emb.device)

            # project speaker embedding to match llm input size
            speaker_embedding_projected = self.speaker_encoder_emb_projection(speaker_emb)  # [B, 1, D]

        B, T_local, H = source_audio_emb.shape

        # Determine decoding length and pad if FSDP
        if self._use_fsdp:
            T_tensor = torch.tensor([T_local], device=source_audio_emb.device)
            dist.all_reduce(T_tensor, op=dist.ReduceOp.MAX)
            T = int(T_tensor.item())
            if T > T_local:
                # pad source emb
                last_frame_source = source_audio_emb[:, T_local - 1: T_local, :]
                pad_source = last_frame_source.repeat(1, T - T_local, 1)
                source_audio_emb = torch.cat([source_audio_emb, pad_source], dim=1)
                # pad text tokens
                last_frame_text = text_tokens[:, T_local - 1: T_local]
                pad_text = last_frame_text.repeat(1, T - T_local)
                text_tokens = torch.cat([text_tokens, pad_text], dim=1)
        else:
            T = T_local

        # Create empty tensor for store model input
        input_embeds = torch.zeros(B, T, H, device=self.device, dtype=source_audio_emb.dtype) # source_audio_emb.clone()

        # This cache is for self.decoder
        cache = DynamicCache()
        gen_audio = torch.zeros(B, T, self._num_codebooks, device=self.device, dtype=torch.long)

        # Add source audio first frame to the model input
        input_embeds[:, 0] = source_audio_emb[:, 0]

        # first audio tokens are full with speech_delay_id
        first_audio_codes = torch.full(
            [B, 1, self._num_codebooks],
            fill_value=self.speech_delay_id,
            device=self.device,
            dtype=torch.long,
        )

        # First step, use speech_delay token
        text_pad_token, text_pad_embedded = self._get_text_pad_embedding()
        # if use bpe char tokenizer sum the embeddings
        if self.use_bpe_char_tokenizer:
            cas_embedding = self.cas_encoder(text_pad_token.unsqueeze(0), subword_mask=None).squeeze(0)  # (B, L, E)
            text_pad_embedded = text_pad_embedded + cas_embedding

        # Add text to the model input
        input_embeds[:, 0] += text_pad_embedded # padding token is the first thing that the model see in text channel, because the target text channel is padded with pad during the context audio condition

        # Add shifted target codes embeddings
        target_audio_emb = self.embed_audio_tokens(
            first_audio_codes
        )
        input_embeds[:, 0] += target_audio_emb[:, 0]

        ans = self(
            input_embeds[:, :1],
            cache=cache,
            seq_mask=None,
        )
        gen_audio[:, 0] = self.sample_codes_from_logits(ans["logits"][:, -1], temperature=self.cfg.get('temperature', 0.7), topk=self.cfg.get('topk', 80)) # (B, num_codebooks)

        speech_state = torch.zeros(B, device=self.device, dtype=torch.long)
        # Autoregressive loop
        for t in range(1, T):

            # add source audio embedding          
            input_embeds[:, t] += source_audio_emb[:, t]
            text_emb = self.embed_text_tokens(text_tokens[:, t : t + 1])

            # if use bpe char tokenizer sum the embeddings
            if self.use_bpe_char_tokenizer:
                cas_embedding = self.cas_encoder(text_tokens[:, t : t + 1], subword_mask=text_mask[:, t : t + 1])  # (B, L, E)
                text_emb = text_emb + cas_embedding
            input_embeds[:, t] += text_emb[:, -1]

            # add audio tokens
            prev_audio_codes = gen_audio[:, t - 1 : t, :]
            input_embeds[:, t] += self.embed_audio_tokens(
                prev_audio_codes
            )[:, -1]

            # replace bos token by speaker embedding
            if self.condition_spk_emb_on_bos_position:
                bos_mask = (text_tokens[:, t] == self.text_bos_id)  # [B]

                if bos_mask.any():
                    # Select embeddings for BOS samples only
                    selected_speaker_emb = speaker_embedding_projected.squeeze(1)[bos_mask]  # [N, D]
                    # Assign to input embeddings at time t
                    input_embeds[bos_mask, t] = selected_speaker_emb  # [N, D]

            ans = self(
                input_embeds[:, t : t + 1],
                cache=ans["cache"],
                seq_mask=None,
            )

            gen_audio[:, t] = self.sample_codes_from_logits(ans["logits"][:, -1], temperature=self.cfg.get('temperature', 0.7), topk=self.cfg.get('topk', 80)) # (B, num_codebooks)

            if self.cfg.get('inference_force_speech_state', None):
                # state 0 - silence, state 1 - speech
                speech_state = torch.where(
                    text_tokens[:, t] == self.text_bos_id, torch.ones_like(speech_state), speech_state
                )
                speech_state = torch.where(
                    text_tokens[:, t] == self.text_eos_id, torch.zeros_like(speech_state), speech_state
                )
                gen_audio[:, t] = torch.where(
                    speech_state.unsqueeze(-1) == 0,
                    gen_audio[:, 0],  # silence
                    gen_audio[:, t],  # speech
                )

        # Trim back to local length if padded
        if self._use_fsdp and T > T_local:
            text_tokens = text_tokens[:, :T_local]
            gen_audio = gen_audio[:, :T_local]

        ans = {
            "text": tokens_to_str(text_tokens, lengths, tokenizer=self.tokenizer, pad_id=self.text_pad_id),
            "tokens_text": text_tokens,
            "tokens_audio": gen_audio,
            "tokens_len": lengths,
        }

        if decode_audio:
            gen_audio_codes = replace_control_speech_codes(gen_audio, self._control_codes)
            with fp32_precision(), torch.no_grad():
                predicted_audio, predicted_audio_lens = self.audio_codec.decode(
                    tokens=gen_audio_codes.transpose(1, 2), tokens_len=lengths
                )
            ans["audio"] = predicted_audio
            ans["audio_len"] = predicted_audio_lens

        if (
            self.cfg.get("use_eou_decoder", None)
            or self.cfg.get("llm_predict_eou", None)
            or self.cfg.get("inference_use_external_eou_predictor", None) or self.cfg.get("inference_eou_from_bos_eos", None)
        ):
            ans["gen_eou"] = gen_eou
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
                    "name": "target_tokens",
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

        llm = self.decoder
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

            for m in (self.final_proj, self.audio_embeddings):
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
            self.decoder = fully_shard(self.decoder, **fsdp_config)

    def load_state_dict(self, state_dict, strict: bool = True):
        try:
            super().load_state_dict(state_dict, strict=strict)
        except RuntimeError as e:
            logging.info(f"Error loading model state_dict !! Retrying with partial initialization!")
            model_dict = set_model_dict_for_partial_init(state_dict, self.state_dict())
            super().load_state_dict(model_dict, strict=False)
