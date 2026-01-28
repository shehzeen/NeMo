# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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
import random
import time
from functools import partial
from typing import List, Sequence, Tuple

import torch
import wandb
from hydra.utils import instantiate
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from omegaconf import DictConfig
from torch import nn
from torch.utils.data import get_worker_info
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from nemo.collections.common.data.lhotse import get_lhotse_dataloader_from_config
from nemo.collections.tts.data.text_to_speech_dataset_lhotse import (
    MagpieTTSLhotseDataset,
    instantiate_phoneme_tokenizer,
    setup_tokenizers,
)
from nemo.collections.tts.models import AudioCodecModel
from nemo.collections.tts.modules import transformer_2501
from nemo.collections.tts.modules.audio_codec_modules import VectorQuantizerIndexConverter
from nemo.collections.tts.modules.magpietts_modules import (
    CharAwareSubwordEncoder,
    LocalTransformerType,
    SpecialAudioToken,
    cosine_schedule,
)
from nemo.collections.tts.parts.utils.helpers import get_mask_from_lengths
from nemo.core.classes import ModelPT
from nemo.core.classes.common import PretrainedModelInfo
from nemo.utils import logging


def worker_init_fn(worker_id):
    # For mp.set_start_method("spawn", force=True)
    # The dataset class should be picklable, so we initialize non-picklable objects here
    logging.info(f"Worker {worker_id} initializing...")
    worker_info = get_worker_info()
    dataset = worker_info.dataset  # Get the dataset instance in this worker
    tokenizer = setup_tokenizers(dataset.tokenizer_config, mode=dataset.dataset_type)
    dataset.text_tokenizer = tokenizer
    if hasattr(dataset, 'phoneme_tokenizer_config'):
        dataset.phoneme_tokenizer = instantiate_phoneme_tokenizer(dataset.phoneme_tokenizer_config)


class EasyMagpieTTSModel(ModelPT):
    """
    Magpie-TTS Model Decoder Only Model
    audio/text
    """

    def __init__(self, cfg: DictConfig, trainer: 'Trainer' = None):
        self.world_size = 1
        if trainer is not None:
            self.world_size = trainer.num_nodes * trainer.num_devices

        # load codec
        codec_model = AudioCodecModel.restore_from(cfg.get('codecmodel_path'), strict=False)
        self.sample_rate = codec_model.sample_rate
        self.output_sample_rate = codec_model.output_sample_rate

        if hasattr(codec_model, "discriminator"):
            # del codec discriminator to free memory
            del codec_model.discriminator

        # Set up codebook configuration
        vector_quantizer = cfg.get('vector_quantizer')
        if vector_quantizer is not None:
            vector_quantizer = instantiate(vector_quantizer)
            num_audio_codebooks = vector_quantizer.num_codebooks
            codebook_size = vector_quantizer.codebook_size
            codec_converter = VectorQuantizerIndexConverter(
                vector_quantizer_original=codec_model.vector_quantizer,
                vector_quantizer_new=vector_quantizer,
            )
            data_num_audio_codebooks = codec_model.vector_quantizer.num_codebooks
        else:
            num_audio_codebooks = codec_model.num_codebooks
            data_num_audio_codebooks = num_audio_codebooks
            codebook_size = codec_model.codebook_size
            codec_converter = None

        # The dataloader needs to know the number of codebooks that the context codes were stored in
        # In the case where there are no context codes saved, and there is no context audio (in the text context path),
        # We create a dummy context code tensor that is only [context_BOS, context_EOS] that is repeated for
        # data_num_audio_codebooks
        self.data_num_audio_codebooks = data_num_audio_codebooks
        self.num_audio_codebooks = num_audio_codebooks
        self.codebook_size = codebook_size

        self.codec_model_samples_per_frame = codec_model.samples_per_frame
        # Our codebooks start with actual audio codec tokens, followed by special tokens.
        # The `forced_*` options are for backward compatibility for models trained with older code.
        # Our codebooks start with actual audio codec tokens, followed by special tokens.
        # The `forced_*` options are for backward compatibility for models trained with older code.
        get_token_index = partial(SpecialAudioToken.get_index, base_codebook_size=self.codebook_size)
        self.audio_bos_id = get_token_index(SpecialAudioToken.AUDIO_BOS)
        self.audio_eos_id = get_token_index(SpecialAudioToken.AUDIO_EOS)
        self.context_audio_bos_id = get_token_index(SpecialAudioToken.AUDIO_CONTEXT_BOS)
        self.context_audio_eos_id = get_token_index(SpecialAudioToken.AUDIO_CONTEXT_EOS)
        self.mask_token_id = get_token_index(SpecialAudioToken.MASK_TOKEN)
        self.num_all_tokens_per_codebook = self.codebook_size + len(SpecialAudioToken)
        self.use_bpe_char_tokenizer = cfg.get('use_bpe_char_tokenizer', False)

        # If specified, use this as the text conditioning tokenizer. Otherwise, use the first tokenizer.
        self.text_conditioning_tokenizer_name = cfg.get('text_conditioning_tokenizer_name', None)
        if self.text_conditioning_tokenizer_name is None:
            self.text_conditioning_tokenizer_name = list(cfg.text_tokenizers.keys())[0]

        self.cfg_unconditional_prob = cfg.get('cfg_unconditional_prob', 0.0)
        self.text_input_mode = cfg.get('text_input_mode', 'full')
        self.streaming_speech_delay = cfg.get('streaming_speech_delay', 3)
        self.streaming_phonemes_delay = cfg.get('streaming_phonemes_delay', 2)
        self.frame_stacking_factor = cfg.get('frame_stacking_factor', 1)

        self.tokenizer = setup_tokenizers(
            all_tokenizers_config=cfg.text_tokenizers,
            mode='train',
        )

        num_tokens_tokenizer = len(self.tokenizer.tokens)
        num_tokens = num_tokens_tokenizer + 3  # +2 for BOS and EOS
        self.bos_id = num_tokens - 3
        self.eos_id = num_tokens - 2
        self.cfg_unk_token_id = num_tokens - 1
        self.phoneme_tokenizer = None
        self.dropout_text_input_prob = cfg.get('dropout_text_input_prob', 0.0)
        self.dropout_phoneme_input_prob = cfg.get('dropout_phoneme_input_prob', 0.0)
        if cfg.get('phoneme_tokenizer', None) is not None:
            self.phoneme_tokenizer = instantiate_phoneme_tokenizer(cfg.phoneme_tokenizer)
            self.phoneme_stacking_factor = cfg.get('phoneme_stacking_factor', 1)
            self.phoneme_vocab_size = self.phoneme_tokenizer.vocab_size

        self.pad_context_text_to_max_duration = False

        super().__init__(cfg=cfg, trainer=trainer)

        # This needs to happen after super().__init__()
        self._codec_model = codec_model
        self._codec_model.freeze()  # Lightning does requires_grad = False and self.eval()
        self._codec_converter = codec_converter

        audio_embeddings = []
        for _ in range(self.num_audio_codebooks * self.frame_stacking_factor):
            audio_embeddings.append(nn.Embedding(self.num_all_tokens_per_codebook, cfg.embedding_dim))
        self.audio_embeddings = nn.ModuleList(audio_embeddings)

        if self.phoneme_tokenizer is not None:
            phoneme_embeddings = []
            for _ in range(self.phoneme_stacking_factor):
                phoneme_embeddings.append(nn.Embedding(self.phoneme_vocab_size, cfg.embedding_dim))
            self.phoneme_embeddings = nn.ModuleList(phoneme_embeddings)
            self.phoneme_final_proj = nn.Linear(cfg.hidden_dim, self.phoneme_vocab_size * self.phoneme_stacking_factor)

        if cfg.transformer_hf_backend == "custom_qwen3_moe_5layer":
            from transformers.models import qwen3_moe

            config = qwen3_moe.configuration_qwen3_moe.Qwen3MoeConfig(
                hidden_size=1536, intermediate_size=3072, num_hidden_layers=5, num_experts=64
            )
            self.decoder = qwen3_moe.modeling_qwen3_moe.Qwen3MoeModel(config)
        elif cfg.transformer_hf_backend == "custom_qwen3_moe_10layer":
            from transformers.models import qwen3_moe

            config = qwen3_moe.configuration_qwen3_moe.Qwen3MoeConfig(
                hidden_size=1536, intermediate_size=3072, num_hidden_layers=10, num_experts=64
            )
            self.decoder = qwen3_moe.modeling_qwen3_moe.Qwen3MoeModel(config)
        elif cfg.transformer_hf_backend == "custom_qwen3_moe_15layer":
            from transformers.models import qwen3_moe

            config = qwen3_moe.configuration_qwen3_moe.Qwen3MoeConfig(
                hidden_size=1536, intermediate_size=3072, num_hidden_layers=15, num_experts=64
            )
            self.decoder = qwen3_moe.modeling_qwen3_moe.Qwen3MoeModel(config)
        elif cfg.transformer_hf_backend == "custom_qwen3_moe_20layer":
            from transformers.models import qwen3_moe

            config = qwen3_moe.configuration_qwen3_moe.Qwen3MoeConfig(
                hidden_size=1536, intermediate_size=3072, num_hidden_layers=20, num_experts=64
            )
            self.decoder = qwen3_moe.modeling_qwen3_moe.Qwen3MoeModel(config)
            # from transformers.models import qwen2_moe
            # config_qwen2 = qwen2_moe.configuration_qwen2_moe.Qwen2MoeConfig(
            #     hidden_size=1536, intermediate_size=3072, num_hidden_layers=5, num_experts=32
            # )
            # self.decoder = qwen2_moe.modeling_qwen2_moe.Qwen2MoeModel(config_qwen2)
        else:
            self.transformer_backend_config = AutoConfig.from_pretrained(
                cfg.transformer_hf_backend,
                trust_remote_code=True,
            )

            hf_transformer = AutoModelForCausalLM.from_config(self.transformer_backend_config)
            self.decoder = hf_transformer.model
            self.lm_text_head = hf_transformer.lm_head

        self.text_embedding = nn.Embedding(num_tokens, cfg.embedding_dim)
        self.decoder.set_input_embeddings(self.text_embedding)

        if self.use_bpe_char_tokenizer:
            # BPE char tokenizer
            assert len(self.tokenizer.tokenizers) == 1, "BPE char tokenizer should only be used with one tokenizer"
            tokenizer_name = self.tokenizer.tokenizer_names[0]
            tokenizer = self.tokenizer.tokenizers[tokenizer_name]
            subword_vocab = tokenizer.get_vocab()
            # special tokens will be stored as it is in the char_vocab
            # Each special token will only be mapped to one char id
            special_vocab = {
                '<BOS>': self.bos_id,
                '<EOS>': self.eos_id,
                '<CFG_UNK>': self.cfg_unk_token_id,
            }
            self.cas_encoder = CharAwareSubwordEncoder(
                d_embed=cfg.embedding_dim,
                llm_tokenizer_vocab=subword_vocab,
                subword_padding_idx=self.tokenizer.pad,
                special_vocab=special_vocab,
            )

        self.final_proj = nn.Linear(
            cfg.hidden_dim, self.num_audio_codebooks * self.num_all_tokens_per_codebook * self.frame_stacking_factor
        )
        self.cross_entropy_loss = nn.CrossEntropyLoss(reduction='none')

        self.local_transformer_type = LocalTransformerType(cfg.get('local_transformer_type', 'none').lower())
        logging.info(f"Local transformer type: {self.local_transformer_type}")
        if self.local_transformer_type != LocalTransformerType.NO_LT:
            local_transformer_hidden_dim = cfg.get('local_transformer_hidden_dim', 256)
            if local_transformer_hidden_dim != cfg.hidden_dim:
                self.local_transformer_in_projection = nn.Linear(cfg.hidden_dim, local_transformer_hidden_dim)
            else:
                self.local_transformer_in_projection = nn.Identity()
            self.local_transformer = transformer_2501.Transformer(
                n_layers=self.cfg.get('local_transformer_n_layers', 2),
                d_model=local_transformer_hidden_dim,
                d_ffn=local_transformer_hidden_dim * 4,
                sa_n_heads=self.cfg.get('local_transformer_n_heads', 1),
                kernel_size=1,
                is_causal=self.local_transformer_type == LocalTransformerType.AR,
                max_length_causal_mask=self.num_audio_codebooks * self.frame_stacking_factor + 2,
                use_learnable_pos_emb=True,
            )
            local_transformer_out_projections = []
            for _ in range(self.num_audio_codebooks * self.frame_stacking_factor):
                # Have a separate projection layer for each codebook, to distinguish between them
                local_transformer_out_projections.append(
                    nn.Linear(local_transformer_hidden_dim, self.num_all_tokens_per_codebook)
                )
            self.local_transformer_out_projections = nn.ModuleList(local_transformer_out_projections)

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        """
        Only used for saving checkpoints. On save, we remove _speaker_verification_model and _codec_model
        from the checkpoint. The codec model is saved in a separate checkpoint.
        """
        if hasattr(self, '_no_state_dict') and self._no_state_dict:
            return {}
        # Don't save the speaker verification and codec model in the state dict
        state_dict = super().state_dict(destination, prefix, keep_vars)
        keys_substrings_to_exclude = ['_speaker_verification_model', '_codec_model']
        for key in list(state_dict.keys()):
            if any([substring in key for substring in keys_substrings_to_exclude]):
                del state_dict[key]
        return state_dict

    def load_state_dict(self, state_dict, strict=True):
        """
        Modify load_state_dict so that we don't restore weights to _speaker_verification_model and _codec_model when
        strict is True.
        When strict is False, we can call pytorch's load_state_dict.
        When strict is True, we loop through all parameters and rename them to enable loading.
        """
        if strict == False:
            super().load_state_dict(state_dict, strict=False)
        for name, child in self.named_children():
            if name in ['_speaker_verification_model', '_codec_model']:
                continue
            if any(param.numel() > 0 for param in child.parameters()):
                # If the module has parameters, we want to change the default mapping so that the state_dict gets
                # loaded.
                # Ex: state_dict[encoder.position_embeddings.weight] -> new_state_dict[position_embeddings.weight]
                new_state_dict = {}
                for key in state_dict.keys():
                    name_with_dot = f"{name}."
                    if key.startswith(name_with_dot):
                        new_state_dict[key[len(name_with_dot) :]] = state_dict[key]
                child.load_state_dict(new_state_dict)

    def add_eos_token(self, codes, codes_len, eos_id, num_eos_tokens=1):
        # codes: (B, C, T')
        # codes_len: (B,)
        codes = torch.nn.functional.pad(input=codes, pad=(0, num_eos_tokens), value=0)
        codes_len = codes_len + num_eos_tokens
        # Insert EOS token at new final token entry
        for idx in range(codes.size(0)):
            codes[idx, :, codes_len[idx] - 1] = eos_id

        return codes, codes_len

    def add_special_tokens(self, codes, codes_len, bos_id, eos_id, num_bos_tokens=1, num_eos_tokens=1):
        # codes: (B, C, T')
        # codes_len: (B,)
        codes = torch.nn.functional.pad(input=codes, pad=(num_bos_tokens, 0), value=bos_id)
        codes_len = codes_len + num_bos_tokens
        codes, codes_len = self.add_eos_token(
            codes=codes, codes_len=codes_len, eos_id=eos_id, num_eos_tokens=num_eos_tokens
        )
        return codes, codes_len

    def remove_bos_token(self, codes, codes_len, num_tokens=1):
        # codes: (B, C, T')
        # codes_len: (B,)
        codes = codes[:, :, num_tokens:]
        codes_len = codes_len - num_tokens
        return codes, codes_len

    def remove_embedded_bos_token(self, embedded, embedded_len):
        # codes: (B, T', C)
        # codes_len: (B,)
        embedded = embedded[:, 1:, :]
        embedded_len = embedded_len - 1
        return embedded, embedded_len

    def remove_eos_token(self, codes, codes_len):
        # codes: (B, C, T')
        # codes_len: (B,)
        codes_len = codes_len - 1
        codes = codes[:, :, :-1]
        mask = get_mask_from_lengths(lengths=codes_len)
        codes = codes * mask.unsqueeze(1)
        return codes, codes_len

    def remove_embedded_eos_token(self, embedded, embedded_len):
        # embedded: (B, T', D)
        # embedded_len: (B,)
        embedded_len = embedded_len - 1
        embedded = embedded[:, :-1, :]
        mask = get_mask_from_lengths(lengths=embedded_len)
        embedded = embedded * mask.unsqueeze(2)
        return embedded, embedded_len

    def remove_special_tokens(self, codes, codes_len, num_bos_tokens=1):
        codes, codes_len = self.remove_bos_token(codes=codes, codes_len=codes_len, num_tokens=num_bos_tokens)
        codes, codes_len = self.remove_eos_token(codes=codes, codes_len=codes_len)
        return codes, codes_len

    def audio_to_codes(self, audio, audio_len, sample_rate=None):
        self._codec_model.eval()
        with torch.no_grad(), torch.autocast(device_type=audio.device.type, dtype=torch.float32):
            codes, codes_len = self._codec_model.encode(audio=audio, audio_len=audio_len, sample_rate=sample_rate)
            return codes, codes_len

    def codes_to_audio(self, codes, codes_len):
        # codes: (B, C, T')
        # codes_len: (B,)
        self._codec_model.eval()
        if self.frame_stacking_factor > 1 and codes.size(1) == self.num_audio_codebooks * self.frame_stacking_factor:
            # Unstack the audio codes if they are stacked
            codes, codes_len = self.unstack_codes(codes, codes_len, self.frame_stacking_factor)

        with torch.no_grad(), torch.autocast(device_type=codes.device.type, dtype=torch.float32):
            # Pass the modified integer token IDs
            if self._codec_converter is not None:
                codes = self._codec_converter.convert_new_to_original(audio_tokens=codes, audio_lens=codes_len)
            audio, audio_len = self._codec_model.decode(tokens=codes, tokens_len=codes_len)
            # audio: (B, T)
            # audio_len: (B,)
            return audio, audio_len, codes

    def embed_audio_tokens(self, audio_tokens):
        # audio_tokens: (B, C, T')
        # Add and average the embeddings of the audio tokens across the codebooks
        audio_embedding = None
        for c in range(audio_tokens.size(1)):
            embedding = self.audio_embeddings[c](audio_tokens[:, c, :])
            if audio_embedding is None:
                audio_embedding = embedding
            else:
                audio_embedding = audio_embedding + embedding
        audio_embedding = audio_embedding / audio_tokens.size(1)
        return audio_embedding

    def embed_phoneme_tokens(self, phoneme_tokens):
        # phoneme_tokens: (B, S, T')
        phoneme_embedding = None
        for c in range(phoneme_tokens.size(1)):
            embedding = self.phoneme_embeddings[c](phoneme_tokens[:, c, :])
            if phoneme_embedding is None:
                phoneme_embedding = embedding
            else:
                phoneme_embedding = phoneme_embedding + embedding
        phoneme_embedding = phoneme_embedding / phoneme_tokens.size(1)
        return phoneme_embedding

    def compute_local_transformer_logits(self, dec_out, audio_codes_target, targets_offset_by_one=False):
        """
        Predicts the logits for all codebooks using the local transformer. Used in both autoregressive (AR) and MaskGit (MG) modes.
        This function is used in training and validation, not inference/sampling.
        The sequence layout is slightly different between AR and MG modes, as shown in the diagram below,
        (using an 8-codebook setup as an example):
        +------------+---------+---------+---------+---------+---------+---------+---------+---------+---------+
        | AR target  |    0    |    1    |    2    |    3    |    4    |    5    |    6    |    7    |   none  |
        +------------+---------+---------+---------+---------+---------+---------+---------+---------+---------+
        | MG target  |  none   |    0    |    1    |    2    |    3    |    4    |    5    |    6    |    7    |
        +------------+---------+---------+---------+---------+---------+---------+---------+---------+---------+
        |   Input    | Magpie  |    0    |    1    |    2    |    3    |    4    |    5    |    6    |    7    |
        |            | Latent  | or MASK | or MASK | or MASK | or MASK | or MASK | or MASK | or MASK | or MASK |
        +------------+---------+---------+---------+---------+---------+---------+---------+---------+---------+
        | Seq. Index |    0    |    1    |    2    |    3    |    4    |    5    |    6    |    7    |    8    |
        +------------+---------+---------+---------+---------+---------+---------+---------+---------+---------+

        dec_out: (B, T', E)
        audio_codes_target: (B, C, T')
        targets_offset_by_one: bool, if False, the target for index 0 is codebook 0, for index 1 is codebook 1, etc. (autoregressive)
                                     if True,  the target for index 1 is codebook 0, for index 2 is codebook 1, etc. (MaskGit)
        """
        dec_out_all = dec_out.reshape(-1, dec_out.size(-1))  # (B*T', E)
        local_transformer_input = [dec_out_all]
        for codebook_num in range(audio_codes_target.size(1)):
            codes = audio_codes_target[:, codebook_num]  # (B, T')
            codes = codes.reshape(-1)  # (B*T',)
            codebook_embedding = self.audio_embeddings[codebook_num](codes)  # (B*T', E)
            local_transformer_input.append(codebook_embedding)

        local_transformer_input = torch.stack(local_transformer_input, dim=1)  # (B*T', C+1, E)
        local_transformer_input = self.local_transformer_in_projection(local_transformer_input)  # (B*T', C+1, 128)
        _mask = torch.ones(
            local_transformer_input.size(0), local_transformer_input.size(1), device=local_transformer_input.device
        )
        local_transformer_output = self.local_transformer(local_transformer_input, _mask)['output']  # (B*T', C+1, E)
        if not targets_offset_by_one:
            # for autoregressive local transformer the target for index 0 is codebook 0, for index 1 is codebook 1, etc.
            local_transformer_output = local_transformer_output[:, :-1, :]  # (B*T', C, E)
        else:
            # for MaskGit the target for index **1** is codebook 0, for index 2 is codebook 1, etc.
            local_transformer_output = local_transformer_output[:, 1:, :]  # (B*T', C, E)
        all_code_logits = []
        for codebook_num in range(audio_codes_target.size(1)):
            # Using a separate projection layer for each codebook (to distinguish between them)
            # Checked the time - this loop is not taking much time (compared to the local transformer forward pass)
            codebook_logits = self.local_transformer_out_projections[codebook_num](
                local_transformer_output[:, codebook_num, :]
            )  # (B*T', num_all_tokens_per_codebook)
            all_code_logits.append(codebook_logits)
        all_code_logits = torch.cat(all_code_logits, dim=1)  # (B*T', num_codebooks * num_all_tokens_per_codebook)

        all_code_logits = all_code_logits.view(
            audio_codes_target.size(0), audio_codes_target.size(2), -1
        )  # (B, T', C * num_all_tokens_per_codebook)

        return all_code_logits

    def compute_loss(self, logits, audio_codes, audio_codes_lens, mask_tokens_mask=None):
        """
        Computes the audio codebook loss. Used by
        (1) The main Magpie-TTS transformer
        (2) The local transformer, for both autoregressive and MaskGit methods

        logits: (B, T', num_codebooks * num_tokens_per_codebook)
        audio_codes: (B, C, T')
        audio_codes_lens: (B,)
        mask_tokens_mask: (B, C, T') True for tokens that were replaced with the MASK_TOKEN and should
                                     therefore be the only ones included in the loss computation.
        """
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
            si = codebook * self.num_all_tokens_per_codebook
            ei = si + self.num_all_tokens_per_codebook
            codebook_logits = logits[:, :, si:ei]  # (B, T', num_tokens_per_codebook)
            codebook_targets = audio_codes[:, codebook]  # (B, T')
            codebook_loss = self.cross_entropy_loss(
                codebook_logits.permute(0, 2, 1), codebook_targets.long()  # (B, num_tokens_per_codebook, T')
            )  # (B, T')
            codebook_loss = codebook_loss * loss_mask[:, codebook, :]
            codebook_loss = codebook_loss.sum() / loss_mask[:, codebook, :].sum()
            if total_codebook_loss is None:
                total_codebook_loss = codebook_loss
            else:
                total_codebook_loss = total_codebook_loss + codebook_loss

        total_codebook_loss = total_codebook_loss / audio_codes.size(1)
        return total_codebook_loss, loss_mask

    def compute_phoneme_loss(self, logits, phoneme_tokens, phoneme_tokens_lens):
        loss_mask = get_mask_from_lengths(phoneme_tokens_lens)
        total_phoneme_loss = None
        for codebook in range(self.phoneme_stacking_factor):
            si = codebook * self.phoneme_vocab_size
            ei = si + self.phoneme_vocab_size
            phoneme_logits = logits[:, :, si:ei]
            phoneme_targets = phoneme_tokens[:, codebook]
            phoneme_loss = self.cross_entropy_loss(phoneme_logits.permute(0, 2, 1), phoneme_targets)
            phoneme_loss = phoneme_loss * loss_mask
            phoneme_loss = phoneme_loss.sum() / loss_mask.sum()
            if total_phoneme_loss is None:
                total_phoneme_loss = phoneme_loss
            else:
                total_phoneme_loss = total_phoneme_loss + phoneme_loss
        total_phoneme_loss = total_phoneme_loss / self.phoneme_stacking_factor
        return total_phoneme_loss, loss_mask

    def forward(self, inputs_embeds, attention_mask, use_cache=False, past_key_values=None):
        backend_out = self.decoder(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            use_cache=use_cache,
            past_key_values=past_key_values,
        )
        # hidden_states = backend_out.last_hidden_state  # (B, T_total, H)
        return backend_out

    def logits_to_audio_codes(self, all_code_logits, audio_codes_lens):
        # all_code_logits: (B, T', num_codebooks * num_tokens_per_codebook)
        # audio_codes_lens: (B,)
        all_preds = []
        for idx in range(self.num_audio_codebooks * self.frame_stacking_factor):
            si = idx * self.num_all_tokens_per_codebook
            ei = si + self.num_all_tokens_per_codebook
            codebook_logits = all_code_logits[:, :, si:ei]
            codebook_probs = torch.softmax(codebook_logits, dim=-1)  # (B, T', num_tokens_per_codebook)
            # argmax to get the tokens
            codebook_preds = torch.argmax(codebook_probs, dim=-1)  # (B, T')
            all_preds.append(codebook_preds)

        all_preds = torch.stack(all_preds, dim=1)  # (B, C, T')
        audio_mask = get_mask_from_lengths(audio_codes_lens)
        all_preds = all_preds * audio_mask.unsqueeze(1)

        return all_preds

    def local_transformer_sample_autoregressive(
        self,
        dec_output,
        temperature=0.7,
        topk=80,
        unfinished_items={},
        finished_items={},
        use_cfg=False,
        cfg_scale=1.0,
    ):
        # dec_output: (B, E)
        self.local_transformer.reset_cache(use_cache=False)
        dec_output = dec_output.unsqueeze(1)  # (B, 1, E)
        local_transformer_input = self.local_transformer_in_projection(dec_output)  # (B, 1, 128)
        all_preds = []
        for codebook_num in range(self.num_audio_codebooks * self.frame_stacking_factor):
            _mask = torch.ones(
                local_transformer_input.size(0), local_transformer_input.size(1), device=local_transformer_input.device
            )
            local_transformer_output = self.local_transformer(local_transformer_input, _mask)['output']  # (B, T, 128)
            codebook_logits = self.local_transformer_out_projections[codebook_num](
                local_transformer_output[:, -1, :]
            )  # (B, num_all_tokens_per_codebook)
            if use_cfg:
                actual_batch_size = codebook_logits.size(0) // 2
                conditional_logits = codebook_logits[:actual_batch_size]
                unconditional_logits = codebook_logits[actual_batch_size:]
                cfg_logits = cfg_scale * conditional_logits + (1.0 - cfg_scale) * unconditional_logits
                codebook_logits[:actual_batch_size] = cfg_logits

            for item_idx in unfinished_items:
                codebook_logits[item_idx, self.audio_eos_id] = float('-inf')
            for item_idx in finished_items:
                codebook_logits[item_idx, :] = float('-inf')
                codebook_logits[item_idx, self.audio_eos_id] = 0.0

            codebook_logits_topk = torch.topk(codebook_logits, topk, dim=-1)[0]  # (B, topk)
            indices_to_remove = codebook_logits < codebook_logits_topk[:, -1].unsqueeze(
                -1
            )  # (B, num_tokens_per_codebook)
            codebook_logits_rescored = codebook_logits.clone()
            codebook_logits_rescored[indices_to_remove] = float('-inf')
            codebook_probs = torch.softmax(
                codebook_logits_rescored / temperature, dim=-1
            )  # (B, num_tokens_per_codebook)
            codebook_preds = torch.multinomial(codebook_probs, 1)  # (B, 1)
            if use_cfg:
                codebook_preds[actual_batch_size:] = codebook_preds[:actual_batch_size]
            all_preds.append(codebook_preds)
            next_local_transformer_input = self.audio_embeddings[codebook_num](codebook_preds.squeeze(-1)).unsqueeze(
                1
            )  # (B, 1, 128)
            next_local_transformer_input = self.local_transformer_in_projection(
                next_local_transformer_input
            )  # (B, 1, 128)
            local_transformer_input = torch.cat(
                [local_transformer_input, next_local_transformer_input], dim=1
            )  # (B, T+1, 128)

        all_preds = torch.cat(all_preds, dim=1).long()  # (B, num_codebooks)
        if use_cfg:
            all_preds = all_preds[:actual_batch_size]

        return all_preds

    def sample_codes_from_logits(
        self, all_code_logits_t, temperature=0.7, topk=80, unfinished_items={}, finished_items={}
    ):
        # all_code_logits_t: (B, num_codebooks * num_tokens_per_codebook), logits at a given timestep
        all_preds = []
        for idx in range(self.num_audio_codebooks * self.frame_stacking_factor):
            si = idx * self.num_all_tokens_per_codebook
            ei = si + self.num_all_tokens_per_codebook
            codebook_logits = all_code_logits_t[:, si:ei]  # (B, num_tokens_per_codebook)
            for item_idx in unfinished_items:
                codebook_logits[item_idx, self.audio_eos_id] = float('-inf')
            for item_idx in finished_items:
                codebook_logits[item_idx, :] = float('-inf')
                codebook_logits[item_idx, self.audio_eos_id] = 0.0
            codebook_logits_topk = torch.topk(codebook_logits, topk, dim=-1)[0]  # (B, topk)
            indices_to_remove = codebook_logits < codebook_logits_topk[:, -1].unsqueeze(
                -1
            )  # (B, num_tokens_per_codebook)
            codebook_logits_rescored = codebook_logits.clone()
            codebook_logits_rescored[indices_to_remove] = float('-inf')

            codebook_probs = torch.softmax(
                codebook_logits_rescored / temperature, dim=-1
            )  # (B, num_tokens_per_codebook)
            codebook_preds = torch.multinomial(codebook_probs, 1)  # (B, 1)
            all_preds.append(codebook_preds)
        all_preds = torch.cat(all_preds, dim=1).long()  # (B, num_codebooks)
        return all_preds

    def sample_codes_from_logits_phoneme(self, all_code_logits_t, temperature=0.7, topk=80):
        # all_code_logits_t: (B, phoneme_stacking_factor * phoneme_vocab_size), logits at a given timestep
        all_preds = []
        for idx in range(self.phoneme_stacking_factor):
            si = idx * self.phoneme_vocab_size
            ei = si + self.phoneme_vocab_size
            codebook_logits = all_code_logits_t[:, si:ei]  # (B, num_tokens_per_codebook)
            codebook_logits_topk = torch.topk(codebook_logits, topk, dim=-1)[0]  # (B, topk)
            indices_to_remove = codebook_logits < codebook_logits_topk[:, -1].unsqueeze(
                -1
            )  # (B, num_tokens_per_codebook)
            codebook_logits_rescored = codebook_logits.clone()
            codebook_logits_rescored[indices_to_remove] = float('-inf')

            codebook_probs = torch.softmax(
                codebook_logits_rescored / temperature, dim=-1
            )  # (B, num_tokens_per_codebook)
            codebook_preds = torch.multinomial(codebook_probs, 1)  # (B, 1)
            all_preds.append(codebook_preds)
        all_preds = torch.cat(all_preds, dim=1).long()  # (B, num_codebooks)
        return all_preds

    def log_val_audio_example(
        self,
        logits,
        target_audio_codes,
        audio_codes_lens_target,
        context_audio_codes=None,
        context_audio_codes_lens=None,
    ):
        wandb_audio_log = {}

        pred_audio_codes = self.logits_to_audio_codes(logits, audio_codes_lens_target)
        pred_audio_codes, _ = self.remove_eos_token(
            codes=pred_audio_codes,
            codes_len=audio_codes_lens_target,
        )
        pred_audio, pred_audio_lens, _ = self.codes_to_audio(pred_audio_codes, audio_codes_lens_target - 1)
        target_audio_codes, _ = self.remove_eos_token(
            codes=target_audio_codes,
            codes_len=audio_codes_lens_target,
        )
        target_audio, target_audio_lens, _ = self.codes_to_audio(target_audio_codes, audio_codes_lens_target - 1)

        context_audio, context_audio_lens = None, None
        if context_audio_codes is not None and context_audio_codes.shape[2] > 3:
            # > 3 ensures, it is a valid context audio tensor (and not dummy tensor used in text context)
            context_audio_codes, context_audio_codes_lens = self.remove_special_tokens(
                codes=context_audio_codes,
                codes_len=context_audio_codes_lens,
            )
            context_audio, context_audio_lens, _ = self.codes_to_audio(context_audio_codes, context_audio_codes_lens)

        for logger in self.loggers:
            is_wandb = isinstance(logger, WandbLogger)
            is_tb = isinstance(logger, TensorBoardLogger)
            if not is_wandb and not is_tb:
                raise ValueError(
                    f"Invalid logger type for audio logging: {type(logger)}. Only `WandbLogger` and `TensorBoardLogger` are supported."
                )

            for idx in range(min(3, pred_audio.size(0))):
                pred_audio_np = pred_audio[idx].float().detach().cpu().numpy()
                target_audio_np = target_audio[idx].float().detach().cpu().numpy()
                pred_audio_np = pred_audio_np[: pred_audio_lens[idx]]
                target_audio_np = target_audio_np[: target_audio_lens[idx]]
                context_audio_np = None
                if context_audio is not None:
                    context_audio_np = context_audio[idx].float().detach().cpu().numpy()
                    context_audio_np = context_audio_np[: context_audio_lens[idx]]

                if is_wandb:
                    wandb_audio_log[f"Audio/Example_{idx}"] = list()
                    if context_audio_np is not None:
                        wandb_audio_log[f"Audio/Example_{idx}"].append(
                            wandb.Audio(context_audio_np, sample_rate=self.sample_rate, caption="context")
                        )
                    wandb_audio_log[f"Audio/Example_{idx}"].append(
                        wandb.Audio(pred_audio_np, sample_rate=self.sample_rate, caption="prediction")
                    )
                    wandb_audio_log[f"Audio/Example_{idx}"].append(
                        wandb.Audio(target_audio_np, sample_rate=self.sample_rate, caption="target")
                    )

                if is_tb:
                    if context_audio_np is not None:
                        logger.experiment.add_audio(
                            f'Example_{idx}/context',
                            context_audio_np,
                            global_step=self.global_step,
                            sample_rate=self.sample_rate,
                        )
                    logger.experiment.add_audio(
                        f'Example_{idx}/prediction',
                        pred_audio_np,
                        global_step=self.global_step,
                        sample_rate=self.sample_rate,
                    )
                    logger.experiment.add_audio(
                        f'Example_{idx}/target',
                        target_audio_np,
                        global_step=self.global_step,
                        sample_rate=self.sample_rate,
                    )

        return wandb_audio_log

    def join_embeddings_temporally(
        self,
        embeddings: Sequence[torch.Tensor],  # [ (B, Ti, E), … ]
        lengths: Sequence[torch.Tensor],  # [ (B,), … ]  same order/size as `embeddings`
        pad_embed: torch.Tensor | None = None,  # (E,)  defaults to zeros
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Merges Multiple Embedding sequences into a single Embedding Sequence.

        Args:
            embeddings  : Sequence of tensors, each of shape (B, Ti, E) — batch, time, embedding
            lengths     : Sequence of tensors, each of shape (B,)
            pad_embed   : (E,)  — embedding to use for padding, defaults to zeros

        Returns:
            joined      : (B, max_sum_len, E)  — merged & padded
            out_lengths : (B,)  — total lengths of each batch element after merging
        """
        if len(embeddings) == 0:
            raise ValueError("contexts must be non-empty")

        B, _, E = embeddings[0].shape
        device = embeddings[0].device
        dtype = embeddings[0].dtype

        # 1. compute output sizes
        len_stack = torch.stack(tuple(lengths), dim=0)  # (N, B)
        out_lengths = len_stack.sum(0)
        max_len = int(out_lengths.max())

        if pad_embed is None:
            pad_embed = torch.zeros(E, dtype=dtype, device=device)

        joined = pad_embed.expand(B, max_len, E).clone()  # (B,max_len,E)

        # batch row indices
        batch_rows = torch.arange(B, device=device).unsqueeze(1)  # (B,1)

        # running offset keeps “write cursor” for each row
        offset = torch.zeros(B, dtype=torch.long, device=device)  # (B,)

        for i, (embedding_i, len_i) in enumerate(zip(embeddings, lengths)):
            Ti = embedding_i.shape[1]
            t_idx = torch.arange(Ti, device=device)  # (Ti,)
            mask = t_idx.unsqueeze(0) < len_i.unsqueeze(1)  # (B,Ti)

            # destination columns: offset + t
            dest_cols = offset.unsqueeze(1) + t_idx  # (B,Ti)

            # Assign embedding_i to the correct positions in joined
            joined[batch_rows.expand_as(mask)[mask], dest_cols[mask]] = embedding_i[mask]

            # move cursor past this segment
            offset += len_i

        return joined, out_lengths

    def prepare_context_tensors(self, batch, dropout_text_input=False):
        # Transcript
        text = batch['text']
        text_lens = batch['text_lens']
        text_embedded = self.decoder.get_input_embeddings()(text)
        if self.use_bpe_char_tokenizer:
            text_mask = get_mask_from_lengths(text_lens)
            cas_embedding = self.cas_encoder(text, subword_mask=text_mask)  # (B, L, E)
            text_embedded = text_embedded + cas_embedding

        if text_embedded.shape[1] < self.streaming_speech_delay + 1:
            # If text is too short, pad it with zeros
            padding_tensor = torch.zeros(
                text_embedded.shape[0],
                self.streaming_speech_delay + 1 - text_embedded.shape[1],
                text_embedded.shape[2],
                device=text_embedded.device,
            )
            text_embedded = torch.cat([text_embedded, padding_tensor], dim=1)

        if dropout_text_input:
            # Make text embedding all zeros
            text_embedded = text_embedded * 0.0

        # Context Audio
        if 'context_audio_codes' in batch:
            context_audio_codes = batch['context_audio_codes']
            context_audio_codes_lens = batch['context_audio_codes_lens']
        else:
            context_audio_codes, context_audio_codes_lens = self.audio_to_codes(
                batch['context_audio'], batch['context_audio_lens']
            )

        if self._codec_converter is not None:
            context_audio_codes = self._codec_converter.convert_original_to_new(
                audio_tokens=context_audio_codes, audio_lens=context_audio_codes_lens
            ).long()

        context_audio_codes, context_audio_codes_lens = self.add_special_tokens(
            codes=context_audio_codes,
            codes_len=context_audio_codes_lens,
            bos_id=self.context_audio_bos_id,
            eos_id=self.context_audio_eos_id,
        )

        context_audio_codes, context_audio_codes_lens = self.stack_codes(
            context_audio_codes,
            context_audio_codes_lens,
            self.audio_bos_id,
            self.audio_eos_id,
            self.frame_stacking_factor,
            self.num_audio_codebooks,
        )
        context_audio_embedded = self.embed_audio_tokens(context_audio_codes)  # (B, T', E)

        # Context Text
        context_text_tokens = batch['context_text_tokens']
        context_text_lens = batch['context_text_tokens_lens']
        context_text_embedded = self.decoder.get_input_embeddings()(context_text_tokens)  # (B, L, E)

        remaining_text_embedded = None
        remaining_text_lens = None
        if self.text_input_mode == 'full':
            context_embedding, context_lens = self.join_embeddings_temporally(
                embeddings=[context_audio_embedded, context_text_embedded, text_embedded],
                lengths=[context_audio_codes_lens, context_text_lens, text_lens],
            )
        elif self.text_input_mode == 'streaming':
            prompt_text_embedded = text_embedded[:, : self.streaming_speech_delay, :]
            prompt_text_lens = torch.ones_like(text_lens) * self.streaming_speech_delay
            context_embedding, context_lens = self.join_embeddings_temporally(
                embeddings=[context_audio_embedded, context_text_embedded, prompt_text_embedded],
                lengths=[context_audio_codes_lens, context_text_lens, prompt_text_lens],
            )
            remaining_text_embedded = text_embedded[:, self.streaming_speech_delay :, :]
            remaining_text_lens = text_lens - self.streaming_speech_delay
            remaining_text_lens = remaining_text_lens.clamp(min=0)
            remaining_text_mask = get_mask_from_lengths(remaining_text_lens)
            remaining_text_embedded = remaining_text_embedded * remaining_text_mask.unsqueeze(2)  # (B, T, E)
        else:
            raise ValueError(f"Invalid text input mode: {self.text_input_mode}")

        return {
            'context_embedding': context_embedding,  # (B, T_total, E)
            'context_lens': context_lens,  # (B,)
            'context_audio_codes': context_audio_codes,  # (B, C, T')
            'context_audio_embedded': context_audio_embedded,  # (B, T', E)
            'context_audio_codes_lens': context_audio_codes_lens,  # (B,)
            'text_embedded': text_embedded,  # (B, L, E)
            'text_lens': text_lens,  # (B,)
            'context_text_tokens': context_text_tokens,  # (B, L)
            'context_text_lens': context_text_lens,  # (B,)
            'remaining_text_embedded': remaining_text_embedded,  # (B, T, E)
            'remaining_text_lens': remaining_text_lens,  # (B,)
        }

    def slice_pred_embeddings(self, transformer_out, context_lens, target_lens):
        """
        Slices the transformer output to get the predicted embeddings for the target sequence.
        Args:
            transformer_out: (B, T, E)
            context_lens: (B,) - start index of target per batch
            target_lens: (B,) - length of target per batch

        Returns: (B, T_max, E) tensor where T_max = max(target_lens)
        """
        B, T, E = transformer_out.shape
        device = transformer_out.device

        # Compute max target length in batch for padding
        max_len = target_lens.max().item()

        # Build index tensor for each batch element
        # Shape: (B, max_len)
        range_indices = torch.arange(max_len, device=device).unsqueeze(0).expand(B, -1)
        gather_indices = context_lens.unsqueeze(1) + range_indices  # (B, max_len)
        gather_indices = torch.clamp(gather_indices, max=transformer_out.size(1) - 1)

        # Expand to shape (B, max_len, E) for gather
        gather_indices_exp = gather_indices.unsqueeze(2).expand(-1, -1, E)
        sliced = torch.gather(transformer_out, dim=1, index=gather_indices_exp)
        return sliced

    def stack_codes(self, codes, codes_lens, bos_id, eos_id, stacking_factor, num_codebooks):
        if stacking_factor == 1:
            return codes, codes_lens

        contains_bos = codes[0, 0, 0].item() == bos_id
        if contains_bos:
            bos_tensor_repeated = torch.full(
                (codes.size(0), (stacking_factor) * num_codebooks, 1), bos_id, device=codes.device
            )  # (B,stacking_factor*C, 1)
            codes = codes[:, :, 1:]  # Remove the bos token
            codes_lens = codes_lens - 1  # Remove the bos token
        B, C, T = codes.shape
        s = int(stacking_factor)

        # --- Compute max padding needed ---
        pad_t = (-T) % s  # pad so that T' is divisible by s
        pad_tail = torch.full((B, C, pad_t), eos_id, dtype=codes.dtype, device=codes.device)
        codes = torch.cat([codes, pad_tail], dim=-1)

        # --- Stack time into channel dimension ---
        Tp = codes.shape[-1]
        T_out = Tp // s
        codes = codes.view(B, C, T_out, s)
        codes = codes.permute(0, 1, 3, 2).reshape(B, C * s, T_out)

        new_lens = torch.div(codes_lens + s - 1, s, rounding_mode='floor')
        if contains_bos:
            codes = torch.cat([bos_tensor_repeated, codes], dim=2)
            new_lens = new_lens + 1

        return codes, new_lens

    def unstack_codes(self, stacked_codes, stacked_lens, stacking_factor):
        if stacking_factor == 1:
            return stacked_codes, stacked_lens

        B, CxS, T_out = stacked_codes.shape
        s = int(stacking_factor)
        assert CxS % s == 0, f"Channel dim ({CxS}) must be divisible by stacking_factor ({s})"

        C = CxS // s
        # Reshape: split channels back into (C, s)
        x = stacked_codes.view(B, C, s, T_out)
        # Bring s back into time dimension
        x = x.permute(0, 1, 3, 2).reshape(B, C, T_out * s)

        # Recover original lengths (before padding)
        orig_lens = stacked_lens * s

        return x, orig_lens

    def prepare_phoneme_channel_input(self, phoneme_tokens, phoneme_tokens_lens, context_lens):
        # import ipdb; ipdb.set_trace()
        phoneme_tokens = phoneme_tokens.unsqueeze(1)  # (B, 1, L)
        phoneme_tokens, phoneme_tokens_lens = self.stack_codes(
            phoneme_tokens,
            phoneme_tokens_lens,
            self.phoneme_tokenizer.bos_token_id,
            self.phoneme_tokenizer.eos_token_id,
            self.phoneme_stacking_factor,
            1,
        )
        # import ipdb; ipdb.set_trace()
        phoneme_tokens_embedded = self.embed_phoneme_tokens(phoneme_tokens)  # (B, T', E)

        phoneme_mask = get_mask_from_lengths(phoneme_tokens_lens)
        phoneme_tokens_embedded = phoneme_tokens_embedded * phoneme_mask.unsqueeze(2)  # (B, T', E)

        zero_context_tensor = torch.zeros(
            context_lens.size(0), context_lens.max().item(), self.cfg.embedding_dim, device=phoneme_tokens.device
        )
        phoneme_channel_input, phoneme_channel_input_lens = self.join_embeddings_temporally(
            embeddings=[zero_context_tensor, phoneme_tokens_embedded],
            lengths=[context_lens, phoneme_tokens_lens],
        )
        return phoneme_channel_input, phoneme_channel_input_lens, phoneme_tokens, phoneme_tokens_lens

    def process_batch(self, batch, mode="train"):
        dropout_text_input = (random.random() < self.dropout_text_input_prob) if mode == 'train' else False
        dropout_phoneme_input = (
            ((random.random() < self.dropout_phoneme_input_prob) and (not dropout_text_input))
            if mode == 'train'
            else False
        )
        context_tensors = self.prepare_context_tensors(batch, dropout_text_input)
        # print("text lens", context_tensors['text_lens'])
        remaining_text_embedded = context_tensors['remaining_text_embedded']
        context_embedding = context_tensors['context_embedding']
        context_lens = context_tensors['context_lens']

        dropout_conditional_input = False
        if mode == 'train' and self.cfg_unconditional_prob > 0.0:
            if torch.rand(1).item() < self.cfg_unconditional_prob:
                dropout_conditional_input = True
                # Get embedding of a special UNCONDITIONAL_TOKEN
                cfg_token_id = self.cfg_unk_token_id  # int
                cfg_token_embedding = self.decoder.get_input_embeddings()(
                    torch.full((context_embedding.size(0), 1), cfg_token_id, device=context_embedding.device)
                )  # (B, 1, E)
                # Keeping the dummy context same size as the context embedding makes
                # inference easier especially with KV caching and using a duplicated batch.
                context_embedding = cfg_token_embedding.expand(-1, context_embedding.size(1), -1)  # (B, T_total, E)
                # Make unconditional remaining text embedding all zeros. Simplifies the inference implementation.
                if self.text_input_mode == 'streaming':
                    remaining_text_embedded = torch.zeros_like(remaining_text_embedded)

        if 'audio_codes' not in batch:
            audio_codes, audio_codes_lens = self.audio_to_codes(batch['audio'], batch['audio_lens'])
        else:
            audio_codes = batch['audio_codes']
            audio_codes_lens = batch['audio_codes_lens']

        if self._codec_converter is not None:
            audio_codes = self._codec_converter.convert_original_to_new(
                audio_tokens=audio_codes, audio_lens=audio_codes_lens
            ).long()

        audio_codes, audio_codes_lens = self.add_special_tokens(
            codes=audio_codes,
            codes_len=audio_codes_lens,
            bos_id=self.audio_bos_id,
            eos_id=self.audio_eos_id,
        )

        audio_codes, audio_codes_lens = self.stack_codes(
            audio_codes,
            audio_codes_lens,
            self.audio_bos_id,
            self.audio_eos_id,
            self.frame_stacking_factor,
            self.num_audio_codebooks,
        )
        audio_codes_lens_input = audio_codes_lens_target = audio_codes_lens - 1
        audio_codes_target = audio_codes[:, :, 1:]  # (B, C, T') Target for the decoder
        audio_codes_input = audio_codes[:, :, :-1]  # (B, C, T') Input to the decoder
        audio_codes_input_embedded = self.embed_audio_tokens(
            audio_codes_input
        )  # (B, T, E) # Computing this to be use in the alignment encoder
        if remaining_text_embedded is not None:
            # Make remaining text embedded the same size as audio_codes_input_embedded by padding with zeros on the right
            padding_len = audio_codes_input_embedded.size(1) - remaining_text_embedded.size(1)
            padding_tensor = torch.zeros(
                remaining_text_embedded.size(0),
                padding_len,
                remaining_text_embedded.size(2),
                device=remaining_text_embedded.device,
            )
            remaining_text_embedded = torch.cat([remaining_text_embedded, padding_tensor], dim=1)
            audio_codes_input_embedded = audio_codes_input_embedded + remaining_text_embedded

        context_plus_audio_embedded, context_plus_audio_lens = self.join_embeddings_temporally(
            embeddings=[context_embedding, audio_codes_input_embedded],
            lengths=[context_lens, audio_codes_lens_input],
        )

        if self.phoneme_tokenizer is not None:
            context_lens_for_phonemes = context_lens - self.streaming_speech_delay + self.streaming_phonemes_delay
            phoneme_channel_input, phoneme_channel_input_lens, phoneme_tokens, phoneme_tokens_lens = (
                self.prepare_phoneme_channel_input(
                    batch['phoneme_tokens'], batch['phoneme_tokens_lens'], context_lens_for_phonemes
                )
            )
            # print("phoneme_tokens_lens", phoneme_tokens_lens)
            # print("audio_codes_lens", audio_codes_lens_input)
            if phoneme_channel_input.shape[1] < context_plus_audio_embedded.shape[1]:
                padding_tensor = torch.zeros(
                    phoneme_channel_input.shape[0],
                    context_plus_audio_embedded.shape[1] - phoneme_channel_input.shape[1],
                    phoneme_channel_input.shape[2],
                    device=phoneme_channel_input.device,
                )
                phoneme_channel_input = torch.cat([phoneme_channel_input, padding_tensor], dim=1)
            else:
                phoneme_channel_input = phoneme_channel_input[:, : context_plus_audio_embedded.shape[1], :]

            if (not dropout_conditional_input) and (not dropout_phoneme_input):
                context_plus_audio_embedded = context_plus_audio_embedded + phoneme_channel_input

        transformer_out = self.forward(
            inputs_embeds=context_plus_audio_embedded,
            attention_mask=get_mask_from_lengths(context_plus_audio_lens),
        )
        transformer_hidden_states = transformer_out.last_hidden_state  # (B, T_total, E)

        pred_embeddings = self.slice_pred_embeddings(
            transformer_hidden_states,
            context_lens=context_lens,
            target_lens=audio_codes_lens_target,
        )

        logits = self.final_proj(pred_embeddings)  # (B, T', num_codebooks * num_tokens_per_codebook)
        # import ipdb; ipdb.set_trace()
        codebook_loss, loss_mask = self.compute_loss(logits, audio_codes_target, audio_codes_lens_target)
        loss = codebook_loss

        local_transformer_loss = None
        local_transformer_logits = None
        if self.local_transformer_type != LocalTransformerType.NO_LT:
            assert self.local_transformer_type == LocalTransformerType.AR, "Unexpected local transformer type"
            local_transformer_logits = self.compute_local_transformer_logits(
                pred_embeddings, audio_codes_target, targets_offset_by_one=False
            )
            local_transformer_loss, _ = self.compute_loss(
                local_transformer_logits, audio_codes_target, audio_codes_lens_target, None
            )
            local_transformer_loss_scale = self.cfg.get('local_transformer_loss_scale', 1.0)
            loss = loss + local_transformer_loss_scale * local_transformer_loss

        phoneme_loss = None
        if self.phoneme_tokenizer is not None:
            pred_embeddings_phoneme = self.slice_pred_embeddings(
                transformer_hidden_states,
                context_lens=context_lens_for_phonemes,
                target_lens=phoneme_tokens_lens - 1,
            )
            phoneme_logits = self.phoneme_final_proj(
                pred_embeddings_phoneme
            )  # (B, T', phoneme_stacking_factor * phoneme_vocab_size)
            if not (dropout_conditional_input or dropout_text_input or dropout_phoneme_input):
                # Only compute phoneme loss if not doing unconditional training or text dropout
                phoneme_loss, _ = self.compute_phoneme_loss(
                    phoneme_logits, phoneme_tokens[:, :, 1:].long(), phoneme_tokens_lens - 1
                )
                print("No Dropout - phoneme loss:", phoneme_loss.item())
            else:
                phoneme_loss = torch.tensor(0.0, device=logits.device)
                print("Dropout - phoneme loss skipped", phoneme_loss.item())

            loss = loss + phoneme_loss

        return {
            'loss': loss,
            'codebook_loss': codebook_loss,
            'phoneme_loss': phoneme_loss,
            'local_transformer_loss': local_transformer_loss,
            'local_transformer_logits': local_transformer_logits,  # (B, T', num_codebooks * num_tokens_per_codebook)
            'logits': logits,
            'audio_codes_target': audio_codes_target,  # (B, C, T')
            'audio_codes_lens_target': audio_codes_lens_target,  # (B,)
            'context_audio_codes': context_tensors['context_audio_codes'],  # (B, C, T')
            'context_audio_codes_lens': context_tensors['context_audio_codes_lens'],  # (B,)
        }

    def training_step(self, batch, batch_idx):
        batch_output = self.process_batch(batch)
        loss = batch_output['loss']
        codebook_loss = batch_output['codebook_loss']
        self.log('train/codebook_loss', codebook_loss, prog_bar=True, sync_dist=True)
        self.log('train/loss', loss, prog_bar=True, sync_dist=True)

        if self.phoneme_tokenizer is not None:
            phoneme_loss = batch_output['phoneme_loss']
            self.log('train/phoneme_loss', phoneme_loss, prog_bar=True, sync_dist=True)

        local_transformer_loss = batch_output['local_transformer_loss']
        if local_transformer_loss is not None:
            self.log('train/local_transformer_loss', local_transformer_loss, prog_bar=True, sync_dist=True)

        # Log batch info
        batch_size, text_token_max_len = batch["text"].shape
        text_token_total_num = batch["text_lens"].sum()
        batch_info_dict = {
            "train/batch_size": batch_size,
            "train/text_token_max_len": text_token_max_len,
            "train/text_token_total_num_in_batch": text_token_total_num,
            "train/text_token_pad_ratio_percent_in_batch": 100
            * (1 - text_token_total_num / (batch_size * text_token_max_len)),
        }

        if "audio_codes" in batch:
            audio_codes_max_len = batch["audio_codes"].shape[-1]
            audio_codes_total_num = batch["audio_codes_lens"].sum()
            batch_info_dict.update(
                {
                    "train/audio_codes_max_len": audio_codes_max_len,
                    "train/audio_codes_total_num_in_batch": audio_codes_total_num,
                    "train/audio_codes_pad_ratio_percent_in_batch": 100
                    * (1 - audio_codes_total_num / (batch_size * audio_codes_max_len)),
                }
            )
        else:
            audio_samples_max_len = batch["audio"].shape[-1]
            audio_samples_total_num = batch["audio_lens"].sum()
            batch_info_dict.update(
                {
                    "train/audio_samples_max_len": audio_samples_max_len,
                    "train/audio_samples_total_num_in_batch": audio_samples_total_num,
                    "train/audio_samples_pad_ratio_percent_in_batch": 100
                    * (1 - audio_samples_total_num / (batch_size * audio_samples_max_len)),
                }
            )

        self.log_dict(batch_info_dict, on_step=True)

        return loss

    def validation_step(self, batch, batch_idx):
        batch_output = self.process_batch(batch, mode="val")
        # self.process_batch returns a dict. We currently only log "logits" which come from the parallel prediction
        # head. If we use local_transformer, then the local_transformer returns "local_transformer_logits"
        loss = batch_output['loss']
        codebook_loss = batch_output['codebook_loss']
        logits = batch_output['logits']
        audio_codes_target = batch_output['audio_codes_target']
        audio_codes_lens_target = batch_output['audio_codes_lens_target']
        context_audio_codes = batch_output['context_audio_codes']
        context_audio_codes_lens = batch_output['context_audio_codes_lens']

        if batch_idx == 0 and self.global_rank == 0:
            # Prepare dictionary for aggregated wandb logging
            wandb_log_dict = {}

            # Get audio data for logging
            wandb_log_dict.update(
                self.log_val_audio_example(
                    logits, audio_codes_target, audio_codes_lens_target, context_audio_codes, context_audio_codes_lens
                )
            )

            # Perform single wandb log call if wandb is active and there is data
            for logger in self.loggers:
                if isinstance(logger, WandbLogger) and wandb_log_dict:
                    logger.experiment.log(wandb_log_dict)

        local_transformer_loss = batch_output['local_transformer_loss']
        val_output = {
            'val_loss': loss,
            'val_codebook_loss': codebook_loss,
            'val_local_transformer_loss': local_transformer_loss,
        }

        if self.phoneme_tokenizer is not None:
            phoneme_loss = batch_output['phoneme_loss']
            val_output['val_phoneme_loss'] = phoneme_loss

        self.validation_step_outputs.append(val_output)

        return val_output

    def on_validation_epoch_end(self):
        collect = lambda key: torch.stack([x[key] for x in self.validation_step_outputs]).mean()
        val_loss = collect("val_loss")
        val_codebook_loss = collect("val_codebook_loss")

        self.log("val_loss", val_loss, prog_bar=True, sync_dist=True)
        self.log("val/codebook_loss", val_codebook_loss, prog_bar=True, sync_dist=True)

        if self.local_transformer_type != LocalTransformerType.NO_LT:
            val_local_transformer_loss = collect("val_local_transformer_loss")
            self.log("val/local_transformer_loss", val_local_transformer_loss, prog_bar=True, sync_dist=True)

        if self.phoneme_tokenizer is not None:
            val_phoneme_loss = collect("val_phoneme_loss")
            self.log("val/phoneme_loss", val_phoneme_loss, prog_bar=True, sync_dist=True)

        self.validation_step_outputs.clear()  # free memory

    def get_dataset(self, dataset_cfg, dataset_type):
        dataset = instantiate(
            dataset_cfg.dataset,
            sample_rate=self.sample_rate,
            bos_id=None,
            eos_id=self.eos_id,
            num_audio_codebooks=self.data_num_audio_codebooks,
            codec_model_samples_per_frame=self.codec_model_samples_per_frame,
            prior_scaling_factor=0.0,
            load_cached_codes_if_available=self.cfg.load_cached_codes_if_available,
            dataset_type=dataset_type,  # train or test used for setting phone prob to 1.0 in test dataset (worker_init_fn)
            use_text_conditioning_tokenizer=True,
            text_conditioning_tokenizer_name=self.text_conditioning_tokenizer_name,
            pad_context_text_to_max_duration=self.pad_context_text_to_max_duration,
            context_duration_min=self.cfg.context_duration_min,
            context_duration_max=self.cfg.context_duration_max,
        )
        dataset.load_16khz_audio = False
        dataset.tokenizer_config = (
            self.cfg.text_tokenizers
        )  # This will be used in worker_init_fn for instantiating tokenizer
        if self.phoneme_tokenizer is not None:
            dataset.phoneme_tokenizer_config = self.cfg.phoneme_tokenizer

        return dataset

    def get_lhotse_dataloader(self, dataset_cfg, mode='train') -> torch.utils.data.DataLoader:
        # TODO @xueyang: better to distinguish cfg. self.cfg is the model cfg, while cfg here is train_ds cfg. Also
        #   cfg is a classifier-free guidance.
        dataset = MagpieTTSLhotseDataset(
            sample_rate=self.sample_rate,
            volume_norm=dataset_cfg.volume_norm,
            codec_model_samples_per_frame=self.codec_model_samples_per_frame,
            num_audio_codebooks=self.data_num_audio_codebooks,
            prior_scaling_factor=0.0,
            load_cached_codes_if_available=self.cfg.load_cached_codes_if_available,
            dataset_type=mode,  # train or test used for setting phone prob to 1.0 in test dataset (worker_init_fn)
            load_16khz_audio=False,
            pad_context_text_to_max_duration=self.pad_context_text_to_max_duration,
            context_duration_min=self.cfg.context_duration_min,
            context_duration_max=self.cfg.context_duration_max,
            use_text_conditioning_tokenizer=True,
            text_conditioning_tokenizer_name=self.text_conditioning_tokenizer_name,
            tokenizer_config=self.cfg.text_tokenizers,
            phoneme_tokenizer_config=self.cfg.get("phoneme_tokenizer", None),
        )

        data_loader = get_lhotse_dataloader_from_config(
            config=dataset_cfg.dataset,
            global_rank=self.global_rank,
            world_size=self.world_size,
            dataset=dataset,
        )
        return data_loader

    def setup_training_data(self, dataset_cfg):
        if dataset_cfg.get("use_lhotse", False):
            # TODO @xueyang: better to distinguish cfg. self.cfg is the model cfg, while cfg here is train_ds cfg. Also
            #   cfg is a classifier-free guidance.
            self._train_dl = self.get_lhotse_dataloader(dataset_cfg, mode='train')
        else:
            dataset = self.get_dataset(dataset_cfg, dataset_type='train')
            sampler = dataset.get_sampler(dataset_cfg.dataloader_params.batch_size, world_size=self.trainer.world_size)
            persistent_workers = True
            if dataset_cfg.dataloader_params.num_workers == 0:
                persistent_workers = False
                # For num workers > 0 tokenizer will be assigned in worker_init_fn (since it is not picklable)
                dataset.text_tokenizer = setup_tokenizers(
                    all_tokenizers_config=self.cfg.text_tokenizers,
                    mode='train',
                )
                if self.cfg.get("phoneme_tokenizer", None) is not None:
                    dataset.phoneme_tokenizer = instantiate_phoneme_tokenizer(self.cfg.phoneme_tokenizer)

            self._train_dl = torch.utils.data.DataLoader(
                dataset,
                collate_fn=dataset.collate_fn,
                sampler=sampler,
                **dataset_cfg.dataloader_params,
                worker_init_fn=worker_init_fn,
                persistent_workers=persistent_workers,
            )

    def _setup_test_dataloader(self, dataset_cfg) -> torch.utils.data.DataLoader:
        if dataset_cfg.get("use_lhotse", False):
            data_loader = self.get_lhotse_dataloader(dataset_cfg, mode='test')
        else:
            dataset = self.get_dataset(dataset_cfg, dataset_type='test')
            persistent_workers = True
            if dataset_cfg.dataloader_params.num_workers == 0:
                persistent_workers = False
                # For num workers > 0 tokenizer will be assigned in worker_init_fn (since it is not picklable)
                dataset.text_tokenizer = setup_tokenizers(all_tokenizers_config=self.cfg.text_tokenizers, mode='test')
                if self.cfg.get("phoneme_tokenizer", None) is not None:
                    dataset.phoneme_tokenizer = instantiate_phoneme_tokenizer(self.cfg.phoneme_tokenizer)

            data_loader = torch.utils.data.DataLoader(
                dataset,
                collate_fn=dataset.collate_fn,
                **dataset_cfg.dataloader_params,
                worker_init_fn=worker_init_fn,
                persistent_workers=persistent_workers,
            )
        return data_loader

    def setup_validation_data(self, cfg):
        self._validation_dl = self._setup_test_dataloader(cfg)

    def setup_test_data(self, cfg):
        self._test_dl = self._setup_test_dataloader(cfg)

    def infer_batch(
        self,
        batch,
        max_decoder_steps=500,
        temperature=0.7,
        topk=80,
        use_local_transformer_for_inference=False,
        maskgit_n_steps=3,
        use_cfg=False,
        cfg_scale=1.0,
        phoneme_input_type='gt',
        phoneme_sampling_method='argmax',
        dropout_text_input=False,
    ):
        # TODO: Make this API same as MagpieTTS model.
        with torch.inference_mode():
            start_time = time.time()
            context_tensors = self.prepare_context_tensors(batch, dropout_text_input=dropout_text_input)
            context_embedding = context_tensors['context_embedding']  # (B, T_total, E)
            context_lens = context_tensors['context_lens']  # (B,)
            remaining_text_embedded = context_tensors['remaining_text_embedded']
            remaining_text_lens = context_tensors['remaining_text_lens']

            if self.phoneme_tokenizer is not None:
                context_lens_for_phonemes = context_lens - self.streaming_speech_delay + self.streaming_phonemes_delay
                phoneme_channel_input, phoneme_channel_input_lens, gt_phoneme_tokens, gt_phoneme_token_lens = (
                    self.prepare_phoneme_channel_input(
                        batch['phoneme_tokens'], batch['phoneme_tokens_lens'], context_lens_for_phonemes
                    )
                )
                phoneme_channel_input_pad_tensor = torch.zeros(
                    phoneme_channel_input.size(0),
                    max_decoder_steps,
                    phoneme_channel_input.size(2),
                    device=phoneme_channel_input.device,
                )
                phoneme_channel_input = torch.cat([phoneme_channel_input, phoneme_channel_input_pad_tensor], dim=1)

            audio_codes_bos = torch.full(
                (context_embedding.size(0), self.num_audio_codebooks * self.frame_stacking_factor, 1),
                self.audio_bos_id,
                device=context_embedding.device,
            ).long()
            audio_codes_lens = torch.full((context_embedding.size(0),), 1, device=context_embedding.device).long()
            audio_codes_input = audio_codes_bos

            audio_codes_input_embedded = self.embed_audio_tokens(audio_codes_input)  # (B, T, E)
            if self.text_input_mode == 'streaming':
                remaining_text_pad_length = max_decoder_steps - remaining_text_lens.max().item() + 1
                remaining_text_pad_tensor = torch.zeros(
                    remaining_text_embedded.size(0),
                    remaining_text_pad_length,
                    remaining_text_embedded.size(2),
                    device=remaining_text_embedded.device,
                )
                remaining_text_embedded = torch.cat([remaining_text_embedded, remaining_text_pad_tensor], dim=1)
                audio_codes_input_embedded = (
                    audio_codes_input_embedded + remaining_text_embedded[:, :1, :]
                )  # :1 corresponds to audio BOS.

            context_plus_audio_embedded, context_plus_audio_lens = self.join_embeddings_temporally(
                embeddings=[context_embedding, audio_codes_input_embedded],
                lengths=[context_lens, audio_codes_lens],
            )
            min_context_len = context_plus_audio_lens.min().item()
            if self.phoneme_tokenizer is not None:
                min_context_len = (
                    min_context_len - self.streaming_speech_delay + self.streaming_phonemes_delay - 1
                )  # 1 for audio BOS that we had added.

            actual_batch_size = context_embedding.size(0)
            if use_cfg:
                dummy_context_embedding_unconditional = self.decoder.get_input_embeddings()(
                    torch.full((actual_batch_size, 1), self.cfg_unk_token_id, device=context_embedding.device)
                )  # (B, 1, E)
                dummy_context_embedding_unconditional_expanded = dummy_context_embedding_unconditional.expand(
                    -1, context_embedding.size(1), -1
                )  # (B, T_total, E)

                dummy_context_plus_audio_embedded, _ = self.join_embeddings_temporally(
                    embeddings=[dummy_context_embedding_unconditional_expanded, audio_codes_input_embedded],
                    lengths=[context_lens, audio_codes_lens],
                )
                first_inference_input = torch.cat(
                    [context_plus_audio_embedded, dummy_context_plus_audio_embedded], dim=0
                )[
                    :, :min_context_len, :
                ]  # (2B, T_min, E)
            else:
                first_inference_input = context_plus_audio_embedded[:, :min_context_len, :]  # (B, T_min, E)
            # First forward pass to get the initial hidden state and past key values
            transformer_out = self.forward(
                inputs_embeds=first_inference_input,
                attention_mask=None,
                use_cache=True,
                past_key_values=None,  # No past key values for the first step
            )

            time_to_first_prediction = time.time() - start_time
            last_hidden = transformer_out.last_hidden_state  # (B, T_total, E)
            past_kv = transformer_out.past_key_values

            all_predictions = []
            end_indices = {}

            current_text_positions = []
            for item_idx in range(context_embedding.size(0)):
                # 0 if we have started reading the remaining text otherwise negative (indicating how far we are before we start reading the remaining text)
                current_text_positions.append(min_context_len - context_plus_audio_lens[item_idx])
            current_text_positions = torch.tensor(current_text_positions, device=context_embedding.device).long()
            if self.phoneme_tokenizer is not None:
                current_phoneme_positions = (
                    current_text_positions - current_text_positions.max() - 1
                )  # Make it 0-indexed.
                # current_text_positions = current_text_positions - self.streaming_speech_delay + self.streaming_phonemes_delay
            pred_phoneme_token_lists = [[] for _ in range(actual_batch_size)]
            gt_phoneme_token_lists = [[] for _ in range(actual_batch_size)]
            phoneme_stream_ended = torch.zeros(
                actual_batch_size, device=context_embedding.device
            ).bool()  # (B,) Whether phoneme stream has ended for this item.
            for idx in range(max_decoder_steps):
                # import ipdb; ipdb.set_trace()
                current_text_positions += 1
                if self.phoneme_tokenizer is not None:
                    current_phoneme_positions += 1
                    print("current_phoneme_positions", current_phoneme_positions)
                if idx % 20 == 0:
                    print(f"Decoding timestep {idx}")

                all_code_logits_t = self.final_proj(
                    last_hidden[:, -1, :]
                )  # (B, num_codebooks * num_tokens_per_codebook)

                if self.phoneme_tokenizer is not None:
                    all_code_logits_t_phoneme = self.phoneme_final_proj(
                        last_hidden[:, -1, :]
                    )  # (B, phoneme_stacking_factor * phoneme_vocab_size)
                    all_code_logits_t_phoneme = all_code_logits_t_phoneme[:actual_batch_size]

                if use_cfg:
                    conditional_logits = all_code_logits_t[:actual_batch_size]
                    unconditional_logits = all_code_logits_t[actual_batch_size:]
                    all_code_logits_t = cfg_scale * conditional_logits + (1.0 - cfg_scale) * unconditional_logits

                if use_local_transformer_for_inference:
                    if self.local_transformer_type == LocalTransformerType.AR:
                        # Autoregressive sampling with local transformer
                        audio_codes_next = self.local_transformer_sample_autoregressive(
                            dec_output=last_hidden[:, -1, :],
                            temperature=temperature,
                            topk=topk,
                            use_cfg=use_cfg,
                            cfg_scale=cfg_scale,
                        )
                    else:
                        raise ValueError(
                            f"Local transformer inference requested by but local transformer type is {self.local_transformer_type}"
                        )
                    # TODO @rfejgin: should we add argmax sampling for EOS here too?
                    all_codes_next_argmax = audio_codes_next
                else:
                    # Parallel sampling from logits
                    audio_codes_next = self.sample_codes_from_logits(
                        all_code_logits_t, temperature=temperature, topk=topk
                    )  # (B, num_codebooks)
                    all_codes_next_argmax = self.sample_codes_from_logits(
                        all_code_logits_t, temperature=0.01
                    )  # (B, num_codebooks)

                phoneme_channel_input_t = None

                if self.phoneme_tokenizer is not None:
                    all_codes_next_phoneme = self.sample_codes_from_logits_phoneme(
                        all_code_logits_t_phoneme, temperature=temperature, topk=topk
                    )  # (B, phoneme_stacking_factor)
                    all_codes_next_phoneme_argmax = self.sample_codes_from_logits_phoneme(
                        all_code_logits_t_phoneme, temperature=0.01
                    )  # (B, phoneme_stacking_factor)
                    pred_phoneme_tokens = (
                        all_codes_next_phoneme_argmax
                        if phoneme_sampling_method == 'argmax'
                        else all_codes_next_phoneme
                    )  # B, phoneme_stacking_factor
                    phoneme_bos_tensor = torch.full(
                        (actual_batch_size, self.phoneme_stacking_factor),
                        self.phoneme_tokenizer.bos_token_id,
                        device=context_embedding.device,
                    ).long()  # (B, phoneme_stacking_factor)
                    use_bos_phoneme = (current_phoneme_positions == 0).unsqueeze(1).long()
                    print("use_bos_phoneme", use_bos_phoneme)
                    pred_phoneme_tokens = (
                        use_bos_phoneme * phoneme_bos_tensor + (1 - use_bos_phoneme) * pred_phoneme_tokens
                    ).long()  # (B, phoneme_stacking_factor)

                    print("pred_phoneme_tokens", pred_phoneme_tokens)
                    gt_phoneme_idx = min(idx, gt_phoneme_tokens.size(2) - 1)
                    gt_phoneme_tokens_current = gt_phoneme_tokens[:, :, gt_phoneme_idx]  # (B, phoneme_stacking_factor)
                    print("gt_phoneme_tokens_current", gt_phoneme_tokens_current)

                    input_phoneme_tokens_current = (
                        gt_phoneme_tokens_current if phoneme_input_type == 'gt' else pred_phoneme_tokens
                    )
                    input_phoneme_embedding = self.embed_phoneme_tokens(
                        input_phoneme_tokens_current.unsqueeze(2)
                    )  # (B, phoneme_stacking_factor, E)

                    use_phoneme_input = (current_phoneme_positions >= 0) * (~phoneme_stream_ended)  # (B,)
                    use_phoneme_input = use_phoneme_input.unsqueeze(1).unsqueeze(2).float()  # (B, 1, 1)
                    zero_phoneme_embedding = torch.zeros(
                        actual_batch_size, self.cfg.embedding_dim, device=all_codes_next_phoneme.device
                    ).unsqueeze(
                        1
                    )  # (B, 1, E)
                    # phoneme_channel_input_t = phoneme_channel_input[torch.arange(actual_batch_size), current_phoneme_positions.clamp(min=0) + min_context_len, :].unsqueeze(1) # (B, 1, E)
                    phoneme_channel_input_t = (
                        use_phoneme_input * input_phoneme_embedding + (1 - use_phoneme_input) * zero_phoneme_embedding
                    )
                    print("use_phoneme_input", use_phoneme_input)
                    for item_idx in range(actual_batch_size):
                        if use_phoneme_input[item_idx, 0, 0] > 0:
                            for phoneme_channel_idx in range(self.phoneme_stacking_factor):
                                _phoneme_token = pred_phoneme_tokens[item_idx, phoneme_channel_idx].item()
                                if _phoneme_token not in [
                                    self.phoneme_tokenizer.eos_token_id,
                                    self.phoneme_tokenizer.bos_token_id,
                                    self.phoneme_tokenizer.pad,
                                ]:
                                    pred_phoneme_token_lists[item_idx].append(_phoneme_token)

                                _gt_phoneme_token = gt_phoneme_tokens_current[item_idx, phoneme_channel_idx].item()
                                if _gt_phoneme_token not in [
                                    self.phoneme_tokenizer.eos_token_id,
                                    self.phoneme_tokenizer.bos_token_id,
                                    self.phoneme_tokenizer.pad,
                                ]:
                                    gt_phoneme_token_lists[item_idx].append(_gt_phoneme_token)

                        if torch.any(input_phoneme_tokens_current[item_idx] == self.phoneme_tokenizer.eos_token_id):
                            print("Phoneme end detected for item {} at timestep {}".format(item_idx, idx))
                            phoneme_stream_ended[item_idx] = True
                    all_codes_next_phoneme = all_codes_next_phoneme.unsqueeze(1)
                    # import ipdb; ipdb.set_trace()

                for item_idx in range(all_codes_next_argmax.size(0)):
                    if item_idx not in end_indices and idx + min_context_len > context_plus_audio_lens[item_idx]:
                        pred_tokens = all_codes_next_argmax[item_idx]
                        pred_tokens_multinomial = audio_codes_next[item_idx]
                        if torch.any(pred_tokens == self.audio_eos_id) or torch.any(
                            pred_tokens_multinomial == self.audio_eos_id
                        ):
                            print("End detected for item {} at timestep {}".format(item_idx, idx))
                            end_indices[item_idx] = idx

                all_predictions.append(audio_codes_next)

                new_emb = self.embed_audio_tokens(audio_codes_next.unsqueeze(2))  # (B, 1, E)
                new_emb_unconditional = new_emb * 1

                if self.text_input_mode == 'streaming':
                    _bs = context_embedding.size(0)
                    remaining_text_embedded_current = remaining_text_embedded[
                        torch.arange(_bs), current_text_positions.clamp(min=0), :
                    ].unsqueeze(
                        1
                    )  # (B, 1, E)
                    new_emb = new_emb + remaining_text_embedded_current

                context_incomplete_mask = context_plus_audio_lens > idx + min_context_len  # (B,)
                # import ipdb; ipdb.set_trace()
                # True if we have not yet reached the end of the context for this item
                # import ipdb; ipdb.set_trace()
                if context_incomplete_mask.any():
                    # If some contexts are not yet complete.
                    context_incomplete_mask = context_incomplete_mask.unsqueeze(1).unsqueeze(2).float()  # (B, 1, 1)
                    context_embedding = context_plus_audio_embedded[
                        :, min_context_len + idx : min_context_len + idx + 1, :
                    ]  # (B, 1, E)
                    next_input = context_incomplete_mask * context_embedding + (1 - context_incomplete_mask) * new_emb
                    if phoneme_channel_input_t is not None:
                        next_input += phoneme_channel_input_t
                    if use_cfg:
                        next_input_unconditional = (
                            context_incomplete_mask * dummy_context_embedding_unconditional
                            + (1 - context_incomplete_mask) * new_emb_unconditional
                        )
                        next_input = torch.cat([next_input, next_input_unconditional], dim=0)  # (2B, 1, E)
                else:
                    next_input = new_emb
                    if phoneme_channel_input_t is not None:
                        next_input += phoneme_channel_input_t
                    if use_cfg:
                        next_input = torch.cat([next_input, new_emb_unconditional], dim=0)  # (2B, 1, E)

                transformer_out = self.forward(
                    inputs_embeds=next_input,
                    attention_mask=None,
                    use_cache=True,
                    past_key_values=past_kv,
                )
                last_hidden = transformer_out.last_hidden_state
                past_kv = transformer_out.past_key_values
                if len(end_indices) == audio_codes_next.size(0):
                    print("All items finished at timestep {}".format(idx))
                    break

            if self.phoneme_tokenizer is not None:
                for item_idx in range(actual_batch_size):
                    print(
                        "Predicted phoneme tokens for item {}: {}".format(item_idx, pred_phoneme_token_lists[item_idx])
                    )
                    print("GT phoneme tokens for item {}: {}".format(item_idx, gt_phoneme_token_lists[item_idx]))
                    predicted_phoneme_text = self.phoneme_tokenizer.decode(pred_phoneme_token_lists[item_idx])
                    gt_phoneme_text = self.phoneme_tokenizer.decode(gt_phoneme_token_lists[item_idx])
                    print("Predicted phoneme text for item {}: {}".format(item_idx, predicted_phoneme_text))
                    print("GT phoneme text for item {}: {}".format(item_idx, gt_phoneme_text))

            tts_generation_time = time.time() - start_time
            tts_generation_time_per_frame = tts_generation_time / len(all_predictions)
            pred_codes_start_indices = context_plus_audio_lens - min_context_len  # (B,)
            predicted_lens = [
                end_indices.get(idx, max_decoder_steps) for idx in range(context_embedding.size(0))
            ]  #  Ensure that the codec is atleast of length 4
            predicted_codes_lens = torch.tensor(predicted_lens, device=context_embedding.device).long()
            predicted_codes_lens = predicted_codes_lens - pred_codes_start_indices  # (B,)

            predicted_codes = torch.stack(all_predictions, dim=-1)  # (B, num_codebooks, T)
            predicted_codes = self.slice_pred_embeddings(
                predicted_codes.permute(0, 2, 1),
                context_lens=pred_codes_start_indices,
                target_lens=predicted_codes_lens,
            )
            predicted_codes = predicted_codes.permute(0, 2, 1)  # (B, num_codebooks, T)
            predicted_codes, predicted_codes_lens = self.remove_eos_token(predicted_codes, predicted_codes_lens)
            predicted_audio, predicted_audio_lens, _ = self.codes_to_audio(predicted_codes, predicted_codes_lens)

            end_time = time.time()
            total_audio_duration_generated = (
                predicted_audio_lens.max().item() * predicted_audio_lens.shape[0]
            ) / self.sample_rate
            rtf = total_audio_duration_generated / (end_time - start_time)

            rtf_metrics = {
                'rtf': rtf,
                'time_to_first_prediction': time_to_first_prediction,
                'tts_generation_time': tts_generation_time,
                'max_frames_generated': len(all_predictions),
                'tts_generation_time_per_frame': tts_generation_time_per_frame,
                'batch_size': context_embedding.size(0),
            }

            return predicted_audio, predicted_audio_lens, predicted_codes, predicted_codes_lens, rtf_metrics

    @classmethod
    def list_available_models(cls) -> List[PretrainedModelInfo]:
        return []
