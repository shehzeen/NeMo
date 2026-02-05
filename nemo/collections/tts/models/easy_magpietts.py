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
from dataclasses import dataclass
from functools import partial
from typing import Any, Dict, List, Optional, Sequence, Tuple

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


@dataclass
class TrainingMode:
    """
    Configuration for a training mode in multi-mode training.

    Attributes:
        name: Unique identifier for this mode (e.g., "full", "streaming_4_8")
        text_input_mode: Either "full" or "streaming"
        streaming_phonemes_delay: Delay for phoneme stream (only used in streaming mode)
        streaming_speech_delay: Delay for speech stream (only used in streaming mode)
        mode_idx: Index of this mode in the list of modes (used for task embedding lookup)
    """

    name: str
    text_input_mode: str
    streaming_phonemes_delay: int
    streaming_speech_delay: int
    mode_idx: int


@dataclass
class ProcessBatchOutput:
    """
    Output dataclass from process_batch containing loss values and model predictions.

    Attributes:
        loss: Total combined loss (codebook_loss + phoneme_loss + local_transformer_loss)
        codebook_loss: Loss for audio codebook prediction
        phoneme_loss: Loss for phoneme prediction (None if phoneme_tokenizer is not used)
        local_transformer_loss: Loss from local transformer (None if not using local transformer)
        local_transformer_logits: Logits from local transformer, shape (B, T', num_codebooks * num_tokens_per_codebook)
        logits: Predicted logits from the main decoder, shape (B, T', num_codebooks * num_tokens_per_codebook)
        audio_codes_target: Target audio codes for the decoder, shape (B, C, T')
        audio_codes_lens_target: Length of target audio codes for each batch item, shape (B,)
        context_audio_codes: Audio codes extracted from context audio, shape (B, C, T')
        context_audio_codes_lens: Length of context audio codes for each batch item, shape (B,)
        selected_training_mode: Name of the selected training mode (None if multi_mode_training is disabled)
    """

    loss: torch.Tensor
    codebook_loss: torch.Tensor
    phoneme_loss: Optional[torch.Tensor]
    local_transformer_loss: Optional[torch.Tensor]
    local_transformer_logits: Optional[torch.Tensor]
    logits: torch.Tensor
    audio_codes_target: torch.Tensor
    audio_codes_lens_target: torch.Tensor
    context_audio_codes: torch.Tensor
    context_audio_codes_lens: torch.Tensor
    selected_training_mode: Optional[str] = None


@dataclass
class StreamingState:
    """
    State for streaming TTS inference with batch support.

    This dataclass maintains all the necessary state for autoregressive streaming
    generation, allowing text tokens to be fed incrementally. Supports arbitrary
    batch sizes where each batch item can have different context lengths and be
    in different phases.

    The streaming operates in four phases (per batch item):
    1. Context phase (context_position < full_context_lens): Processing remaining context
    2. Prompt phase (text_tokens_seen < phoneme_delay): Only text, no predictions
    3. Phoneme-only phase (phoneme_delay <= text_tokens_seen < speech_delay): Phoneme predictions only
    4. Audio phase (text_tokens_seen >= speech_delay): Both phoneme and audio predictions

    Attributes:
        batch_size: Number of items in the batch.
        past_key_values: KV cache from the transformer for efficient autoregressive decoding.
        cache_seq_len: Current sequence length in the cache.
        all_predictions: List of predicted audio codes at each timestep, each tensor is (B, C, S) unstacked.
        all_phoneme_predictions: List of predicted phoneme tokens at each timestep, each tensor is (B, phoneme_stacking_factor).
        context_audio_codes: Processed context audio codes with special tokens.
        context_audio_codes_lens: Length of context audio codes.
        context_lens: Total context length (task_embedding + context_audio + context_text).
        full_context_embedding: Full context embedding for each batch item (B, T_max_context, E).
        full_context_lens: Full context length for each batch item (B,).
        context_position: How much context has been processed per batch item (B,).
        text_tokens_seen: Number of text tokens processed so far per batch item (B,).
        phoneme_steps: Number of phoneme prediction steps taken per batch item (B,).
        audio_steps: Number of audio prediction steps taken per batch item (B,).
        phoneme_stream_ended: Whether the phoneme stream has ended per batch item (B,) bool tensor.
        finished: Whether generation is complete per batch item (B,) bool tensor.
        device: Device tensors are on.
        training_mode: The training mode being used for inference.
        use_cfg: Whether classifier-free guidance is enabled.
        cfg_scale: CFG scale factor.
        use_local_transformer: Whether to use local transformer for inference.
        temperature: Sampling temperature.
        topk: Top-k sampling parameter.
        dummy_context_embedding_unconditional: Unconditional embedding for CFG (if enabled).
        last_hidden: Last hidden state from transformer.
        text_finished: Whether text input has finished per batch item (B,) bool tensor.
        phoneme_input_type: 'gt' or 'pred' for phoneme tokens.
        phoneme_sampling_method: 'argmax' or 'sample' for phoneme token selection.
        last_phoneme_tokens: Last predicted phoneme tokens (B, phoneme_stacking_factor).
        last_audio_codes: Last predicted audio codes (B, num_codebooks).
        audio_prediction_start_idx: Global frame index where audio predictions start per batch item (B,).
        audio_prediction_end_idx: Global frame index where audio predictions end per batch item (B,), -1 if not ended.
        phoneme_prediction_start_idx: Global step index where phoneme predictions start per batch item (B,).
        phoneme_prediction_end_idx: Global step index where phoneme predictions end per batch item (B,), -1 if not ended.
    """

    batch_size: int
    past_key_values: Optional[Tuple]
    cache_seq_len: int
    all_predictions: List[torch.Tensor]
    all_phoneme_predictions: List[torch.Tensor]
    context_audio_codes: torch.Tensor
    context_audio_codes_lens: torch.Tensor
    context_lens: torch.Tensor
    full_context_embedding: torch.Tensor
    full_context_lens: torch.Tensor
    context_position: torch.Tensor
    text_tokens_seen: torch.Tensor
    phoneme_steps: torch.Tensor
    audio_steps: torch.Tensor
    phoneme_stream_ended: torch.Tensor
    finished: torch.Tensor
    device: torch.device
    training_mode: TrainingMode
    use_cfg: bool
    cfg_scale: float
    use_local_transformer: bool
    temperature: float
    topk: int
    dummy_context_embedding_unconditional: Optional[torch.Tensor]
    last_hidden: torch.Tensor
    text_finished: torch.Tensor
    phoneme_input_type: str
    phoneme_sampling_method: str
    last_phoneme_tokens: Optional[torch.Tensor]
    last_audio_codes: Optional[torch.Tensor]
    audio_prediction_start_idx: torch.Tensor
    audio_prediction_end_idx: torch.Tensor
    phoneme_prediction_start_idx: torch.Tensor
    phoneme_prediction_end_idx: torch.Tensor
    gt_phoneme_embeddings: Optional[torch.Tensor] = None  # (B, T', E) pre-computed GT embeddings
    gt_phoneme_lens: Optional[torch.Tensor] = None  # (B,) lengths after stacking


@dataclass
class StreamingFinalizeOutput:
    """Output from streaming_finalize containing audio and phoneme predictions."""

    audio: torch.Tensor  # (B, max_audio_len) generated audio waveform
    audio_len: torch.Tensor  # (B,) length of audio per batch item
    audio_codes: torch.Tensor  # (B, num_codebooks, T) generated audio codes
    audio_codes_len: torch.Tensor  # (B,) length of codes per batch item
    phoneme_tokens: List[List[int]]  # List of phoneme token sequences per batch item
    phoneme_text: List[str]  # Decoded phoneme strings per batch item


@dataclass
class InferBatchOutput:
    """Output dataclass for EasyMagpieTTS infer_batch method."""

    predicted_audio: torch.Tensor  # (B, T_audio)
    predicted_audio_lens: torch.Tensor  # (B,)
    predicted_codes: torch.Tensor  # (B, num_codebooks, T_frames)
    predicted_codes_lens: torch.Tensor  # (B,)
    rtf_metrics: Dict[str, Any]


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

        # Multi-mode training configuration
        # The model trains with multiple text input modes (full, streaming with various delays)
        # Each mode has its own task embedding that is prepended to the context
        training_modes_cfg = cfg.get('training_modes', None)
        if training_modes_cfg is None:
            # Create a default training mode for backward compatibility
            self.training_modes = [
                TrainingMode(
                    name="streaming_4_8",
                    text_input_mode="streaming",
                    streaming_phonemes_delay=4,
                    streaming_speech_delay=8,
                    mode_idx=0,
                )
            ]

        else:
            self.training_modes = []
            for mode_idx, mode_cfg in enumerate(training_modes_cfg):
                mode = TrainingMode(
                    name=mode_cfg.name,
                    text_input_mode=mode_cfg.text_input_mode,
                    streaming_phonemes_delay=mode_cfg.get('streaming_phonemes_delay', 0),
                    streaming_speech_delay=mode_cfg.get('streaming_speech_delay', 0),
                    mode_idx=mode_idx,
                )
                self.training_modes.append(mode)

        logging.info(f"Multi-mode training with {len(self.training_modes)} modes:")
        for mode in self.training_modes:
            logging.info(
                f"  - {mode.name}: text_input_mode={mode.text_input_mode}, "
                f"streaming_phonemes_delay={mode.streaming_phonemes_delay}, "
                f"streaming_speech_delay={mode.streaming_speech_delay}"
            )

        # Create a mapping from mode name to mode object for easy lookup during inference
        self.mode_name_to_mode = {mode.name: mode for mode in self.training_modes}
        # Default mode for inference if not specified (first mode in the list)
        self.default_inference_mode = self.training_modes[0].name

        self.frame_stacking_factor = cfg.get('frame_stacking_factor', 1)

        self.tokenizer = setup_tokenizers(
            all_tokenizers_config=cfg.text_tokenizers,
            mode='train',
        )

        num_tokens_tokenizer = len(self.tokenizer.tokens)
        num_tokens = num_tokens_tokenizer + 3  # +3 for BOS, EOS, CFG_UNK
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

        # Audio embedding dimension - can be smaller than hidden_dim to reduce parameters
        self.audio_embedding_dim = cfg.get('audio_embedding_dim', cfg.hidden_dim)

        audio_embeddings = []
        for _ in range(self.num_audio_codebooks * self.frame_stacking_factor):
            audio_embeddings.append(nn.Embedding(self.num_all_tokens_per_codebook, self.audio_embedding_dim))
        self.audio_embeddings = nn.ModuleList(audio_embeddings)

        # Projection from audio_embedding_dim to embedding_dim (Identity if same)
        if self.audio_embedding_dim != cfg.embedding_dim:
            self.audio_in_projection = nn.Linear(self.audio_embedding_dim, cfg.embedding_dim)
        else:
            self.audio_in_projection = nn.Identity()

        if self.phoneme_tokenizer is not None:
            phoneme_embeddings = []
            for _ in range(self.phoneme_stacking_factor):
                phoneme_embeddings.append(nn.Embedding(self.phoneme_vocab_size, cfg.embedding_dim))
            self.phoneme_embeddings = nn.ModuleList(phoneme_embeddings)
            self.phoneme_final_proj = nn.Linear(cfg.hidden_dim, self.phoneme_vocab_size * self.phoneme_stacking_factor)

        # Decoder backend selection - supports HuggingFace models or NemotronH
        self.decoder_type = cfg.get('decoder_type', 'huggingface')  # backward compatible default
        logging.info(f"Using decoder type: {self.decoder_type}")

        if self.decoder_type == 'huggingface':
            # Existing HuggingFace path
            self.transformer_backend_config = AutoConfig.from_pretrained(
                cfg.transformer_hf_backend,
                trust_remote_code=True,
            )
            hf_transformer = AutoModelForCausalLM.from_config(self.transformer_backend_config)
            self.decoder = hf_transformer.model
            self.lm_text_head = hf_transformer.lm_head

        elif self.decoder_type == 'nemotron_h':
            # NemotronH hybrid Mamba2/Attention backend
            from nemo.collections.tts.modules.nemotron_h_decoder import NemotronHConfig, NemotronHForCausalLM

            # Build config from YAML parameters
            nemotron_h_config_dict = dict(cfg.get('nemotron_h_config', {}))
            # Ensure hidden_size matches embedding_dim for compatibility
            if 'hidden_size' not in nemotron_h_config_dict:
                nemotron_h_config_dict['hidden_size'] = cfg.embedding_dim
            nemotron_config = NemotronHConfig(**nemotron_h_config_dict)
            nemotron_model = NemotronHForCausalLM(nemotron_config)
            self.decoder = nemotron_model.backbone
            self.lm_text_head = nemotron_model.lm_head
            logging.info(
                f"NemotronH config: {nemotron_config.num_hidden_layers} layers, pattern={nemotron_config.hybrid_override_pattern[:20]}..."
            )

        else:
            raise ValueError(f"Unknown decoder_type: {self.decoder_type}. Supported: 'huggingface', 'nemotron_h'")

        self.text_embedding = nn.Embedding(num_tokens, cfg.embedding_dim)
        self.decoder.set_input_embeddings(self.text_embedding)

        # Task embedding for multi-mode training
        # Each mode has a unique task embedding that is prepended to the context
        # Only create task embedding if there are multiple modes
        num_modes = len(self.training_modes)
        if num_modes > 1:
            self.task_embedding = nn.Embedding(num_modes, cfg.embedding_dim)
            logging.info(f"Created task embedding with {num_modes} modes, embedding_dim={cfg.embedding_dim}")
        else:
            self.task_embedding = None
            logging.info(f"Single training mode '{self.training_modes[0].name}', skipping task embedding")

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

        # Projection from hidden_dim to audio_embedding_dim before final_proj (Identity if same)
        if self.audio_embedding_dim != cfg.hidden_dim:
            self.audio_out_projection = nn.Linear(cfg.hidden_dim, self.audio_embedding_dim)
        else:
            self.audio_out_projection = nn.Identity()

        self.final_proj = nn.Linear(
            self.audio_embedding_dim,
            self.num_audio_codebooks * self.num_all_tokens_per_codebook * self.frame_stacking_factor,
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
            # Projection from local_transformer_hidden_dim to audio_embedding_dim (Identity if same)
            if self.audio_embedding_dim != local_transformer_hidden_dim:
                self.local_transformer_audio_out_projection = nn.Linear(
                    local_transformer_hidden_dim, self.audio_embedding_dim
                )
            else:
                self.local_transformer_audio_out_projection = nn.Identity()
            local_transformer_out_projections = []
            for _ in range(self.num_audio_codebooks * self.frame_stacking_factor):
                # Have a separate projection layer for each codebook, to distinguish between them
                local_transformer_out_projections.append(
                    nn.Linear(self.audio_embedding_dim, self.num_all_tokens_per_codebook)
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
        # Project from audio_embedding_dim to embedding_dim
        audio_embedding = self.audio_in_projection(audio_embedding)
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
        dec_out_all = dec_out.reshape(-1, dec_out.size(-1))  # (B*T', hidden_dim)
        local_transformer_input = [dec_out_all]
        for codebook_num in range(audio_codes_target.size(1)):
            codes = audio_codes_target[:, codebook_num]  # (B, T')
            codes = codes.reshape(-1)  # (B*T',)
            codebook_embedding = self.audio_embeddings[codebook_num](codes)  # (B*T', audio_embedding_dim)
            # Project from audio_embedding_dim to embedding_dim
            codebook_embedding = self.audio_in_projection(codebook_embedding)
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
        # Project from local_transformer_hidden_dim to audio_embedding_dim
        local_transformer_output = self.local_transformer_audio_out_projection(local_transformer_output)
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

    def compute_loss(self, logits, audio_codes, audio_codes_lens):
        """
        Computes the audio codebook loss. Used by
        (1) The main Magpie-TTS transformer
        (2) The local transformer

        logits: (B, T', num_codebooks * num_tokens_per_codebook)
        audio_codes: (B, C, T')
        audio_codes_lens: (B,)
        """
        loss_mask = get_mask_from_lengths(audio_codes_lens)
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

    def forward(self, inputs_embeds, attention_mask, use_cache=False, past_key_values=None, cache_position=None):
        # Only pass cache_position for NemotronH (HF transformers may not accept it)
        if self.decoder_type == 'nemotron_h':
            backend_out = self.decoder(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                use_cache=use_cache,
                past_key_values=past_key_values,
                cache_position=cache_position,
            )
        else:
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
            # Project from local_transformer_hidden_dim to audio_embedding_dim
            local_transformer_output_projected = self.local_transformer_audio_out_projection(
                local_transformer_output[:, -1, :]
            )
            codebook_logits = self.local_transformer_out_projections[codebook_num](
                local_transformer_output_projected
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
            )  # (B, 1, audio_embedding_dim)
            # Project from audio_embedding_dim to embedding_dim, then to local_transformer_hidden_dim
            next_local_transformer_input = self.audio_in_projection(next_local_transformer_input)
            next_local_transformer_input = self.local_transformer_in_projection(
                next_local_transformer_input
            )  # (B, 1, local_transformer_hidden_dim)
            local_transformer_input = torch.cat(
                [local_transformer_input, next_local_transformer_input], dim=1
            )  # (B, T+1, local_transformer_hidden_dim)

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
                            wandb.Audio(context_audio_np, sample_rate=self.output_sample_rate, caption="context")
                        )
                    wandb_audio_log[f"Audio/Example_{idx}"].append(
                        wandb.Audio(pred_audio_np, sample_rate=self.output_sample_rate, caption="prediction")
                    )
                    wandb_audio_log[f"Audio/Example_{idx}"].append(
                        wandb.Audio(target_audio_np, sample_rate=self.output_sample_rate, caption="target")
                    )

                if is_tb:
                    if context_audio_np is not None:
                        logger.experiment.add_audio(
                            f'Example_{idx}/context',
                            context_audio_np,
                            global_step=self.global_step,
                            sample_rate=self.output_sample_rate,
                        )
                    logger.experiment.add_audio(
                        f'Example_{idx}/prediction',
                        pred_audio_np,
                        global_step=self.global_step,
                        sample_rate=self.output_sample_rate,
                    )
                    logger.experiment.add_audio(
                        f'Example_{idx}/target',
                        target_audio_np,
                        global_step=self.global_step,
                        sample_rate=self.output_sample_rate,
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
            # Ensure dtype matches to avoid errors during mixed-precision training
            joined[batch_rows.expand_as(mask)[mask], dest_cols[mask]] = embedding_i[mask].to(joined.dtype)

            # move cursor past this segment
            offset += len_i

        return joined, out_lengths

    def prepare_context_tensors(
        self,
        context_text_tokens: torch.Tensor,
        context_text_tokens_lens: torch.Tensor,
        context_audio_codes: Optional[torch.Tensor] = None,
        context_audio_codes_lens: Optional[torch.Tensor] = None,
        context_audio: Optional[torch.Tensor] = None,
        context_audio_lens: Optional[torch.Tensor] = None,
        training_mode: Optional[TrainingMode] = None,
        dropout_conditional_input: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Prepare context tensors (without text) for the simplified process_batch.

        This function processes context audio and context text to create the combined
        context embedding.
        Args:
            context_text_tokens: Context text token IDs for speaker/style conditioning (B, L)
            context_text_tokens_lens: Length of context text for each batch item (B,)
            context_audio_codes: Pre-computed audio codes for context audio (B, C, T').
                If None, will be computed from context_audio.
            context_audio_codes_lens: Length of context audio codes (B,).
                Required if context_audio_codes is provided.
            context_audio: Raw context audio waveform (B, T).
                Used to compute context_audio_codes if not provided.
            context_audio_lens: Length of context audio (B,).
                Required if context_audio is provided.
            training_mode: Optional TrainingMode object specifying the mode to use.
                If None, uses the first mode from training_modes as default.
            dropout_conditional_input: If True, replace context with CFG unconditional token.

        Returns:
            Tuple of:
                - context_embedding: Combined context embedding (B, T_context, E)
                - context_lens: Total context length per batch item (B,)
                - context_audio_codes: Processed audio codes with special tokens (B, C, T')
                - context_audio_codes_lens: Length of processed context audio codes (B,)
        """
        # Determine the mode parameters to use
        if training_mode is None:
            training_mode = self.training_modes[0]

        current_mode_idx = training_mode.mode_idx
        batch_size = context_text_tokens.size(0)
        device = context_text_tokens.device

        # Context Audio
        if context_audio_codes is None:
            if context_audio is None:
                raise ValueError("Either context_audio_codes or context_audio must be provided")
            context_audio_codes, context_audio_codes_lens = self.audio_to_codes(context_audio, context_audio_lens)

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
        context_text_lens = context_text_tokens_lens
        context_text_embedded = self.decoder.get_input_embeddings()(context_text_tokens)  # (B, L, E)

        # Prepare task embedding for multi-mode training
        task_embedding = None
        task_embedding_lens = None
        if self.task_embedding is not None and current_mode_idx is not None:
            mode_idx_tensor = torch.full((batch_size,), current_mode_idx, dtype=torch.long, device=device)
            task_embedding = self.task_embedding(mode_idx_tensor).unsqueeze(1)  # (B, 1, E)
            task_embedding_lens = torch.ones(batch_size, dtype=torch.long, device=device)  # (B,)

        # Combine context embeddings: [task_embedding | context_audio | context_text]
        if task_embedding is not None:
            context_embedding, context_lens = self.join_embeddings_temporally(
                embeddings=[task_embedding, context_audio_embedded, context_text_embedded],
                lengths=[task_embedding_lens, context_audio_codes_lens, context_text_lens],
            )
        else:
            context_embedding, context_lens = self.join_embeddings_temporally(
                embeddings=[context_audio_embedded, context_text_embedded],
                lengths=[context_audio_codes_lens, context_text_lens],
            )

        # Handle CFG unconditional dropout
        if dropout_conditional_input:
            cfg_token_id = self.cfg_unk_token_id
            cfg_token_embedding = self.decoder.get_input_embeddings()(
                torch.full((batch_size, 1), cfg_token_id, device=device)
            )  # (B, 1, E)
            # Expand CFG token to match context embedding size
            context_embedding = cfg_token_embedding.expand(-1, context_embedding.size(1), -1)  # (B, T_context, E)

        return context_embedding, context_lens, context_audio_codes, context_audio_codes_lens

    def prepare_text_channel_embeddings(
        self,
        text: torch.Tensor,
        text_lens: torch.Tensor,
        delay: torch.Tensor,
        dropout_text_input: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare text embeddings as a channel input with delay handling.

        This function embeds text tokens and prepends zero-padding based on the delay
        parameter. The delay represents the number of zero positions to prepend before
        the text embeddings, aligning the text channel with other channels.

        Args:
            text: Input text token IDs (B, L)
            text_lens: Length of text for each batch item (B,)
            delay: Number of zero positions to prepend for each batch item (B,).
                   For text channel, this is typically just context_lens.
            dropout_text_input: If True, return all zeros (for text dropout regularization).

        Returns:
            Tuple of:
                - text_channel_embedding: Text embeddings with zero-padded delay (B, T_delay + T_text, E)
                - text_channel_lens: Total length of text channel for each batch item (B,)
        """
        batch_size = text.size(0)
        device = text.device

        # Embed text tokens
        text_embedded = self.decoder.get_input_embeddings()(text)  # (B, L, E)

        # Apply CAS encoding if using BPE char tokenizer
        if self.use_bpe_char_tokenizer:
            text_mask = get_mask_from_lengths(text_lens)
            cas_embedding = self.cas_encoder(text, subword_mask=text_mask)  # (B, L, E)
            text_embedded = text_embedded + cas_embedding

        # Handle text dropout - zero out the embeddings
        if dropout_text_input:
            text_embedded = text_embedded * 0.0

        # Create zero tensor for delay padding
        max_delay = delay.max().item()
        zero_delay_tensor = torch.zeros(batch_size, max_delay, self.cfg.embedding_dim, device=device)

        # Join delay zeros with text embeddings
        text_channel_embedding, text_channel_lens = self.join_embeddings_temporally(
            embeddings=[zero_delay_tensor, text_embedded],
            lengths=[delay, text_lens],
        )

        return text_channel_embedding, text_channel_lens

    def prepare_phoneme_channel_embeddings(
        self,
        phoneme_tokens: torch.Tensor,
        phoneme_tokens_lens: torch.Tensor,
        delay: torch.Tensor,
        dropout_phoneme_input: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Prepare phoneme embeddings as a channel input with delay handling.

        This function stacks phoneme tokens (if configured), embeds them, and prepends
        zero-padding based on the delay parameter. The delay represents the number of
        zero positions to prepend before the phoneme embeddings.

        Args:
            phoneme_tokens: Phoneme token IDs (B, L)
            phoneme_tokens_lens: Length of phoneme tokens for each batch item (B,)
            delay: Number of zero positions to prepend for each batch item (B,).
                   This is typically context_lens + phoneme_delay.
            dropout_phoneme_input: If True, return all zeros (for phoneme dropout regularization).

        Returns:
            Tuple of:
                - phoneme_channel_embedding: Phoneme embeddings with zero-padded delay (B, T_delay + T_phoneme, E)
                - phoneme_channel_lens: Total length of phoneme channel for each batch item (B,)
                - phoneme_tokens_stacked: Stacked phoneme tokens (B, S, T')
                - phoneme_tokens_lens_stacked: Length of stacked phoneme tokens (B,)
        """
        batch_size = phoneme_tokens.size(0)
        device = phoneme_tokens.device

        # Stack phoneme tokens
        phoneme_tokens_expanded = phoneme_tokens.unsqueeze(1)  # (B, 1, L)
        phoneme_tokens_stacked, phoneme_tokens_lens_stacked = self.stack_codes(
            phoneme_tokens_expanded,
            phoneme_tokens_lens,
            self.phoneme_tokenizer.bos_token_id,
            self.phoneme_tokenizer.eos_token_id,
            self.phoneme_stacking_factor,
            1,
        )

        # Embed phoneme tokens
        phoneme_embedded = self.embed_phoneme_tokens(phoneme_tokens_stacked)  # (B, T', E)

        # Apply mask to zero out padding
        phoneme_mask = get_mask_from_lengths(phoneme_tokens_lens_stacked)
        phoneme_embedded = phoneme_embedded * phoneme_mask.unsqueeze(2)  # (B, T', E)

        # Handle phoneme dropout - zero out the embeddings
        if dropout_phoneme_input:
            phoneme_embedded = phoneme_embedded * 0.0

        # Create zero tensor for delay padding
        max_delay = delay.max().item()
        zero_delay_tensor = torch.zeros(batch_size, max_delay, self.cfg.embedding_dim, device=device)

        # Join delay zeros with phoneme embeddings
        phoneme_channel_embedding, phoneme_channel_lens = self.join_embeddings_temporally(
            embeddings=[zero_delay_tensor, phoneme_embedded],
            lengths=[delay, phoneme_tokens_lens_stacked],
        )

        return phoneme_channel_embedding, phoneme_channel_lens, phoneme_tokens_stacked, phoneme_tokens_lens_stacked

    def prepare_audio_channel_embeddings(
        self,
        audio_codes: torch.Tensor,
        audio_codes_lens: torch.Tensor,
        delay: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Prepare audio embeddings as a channel input with delay handling.

        This function processes audio codes by adding special tokens, stacking them,
        and embedding them. It prepends zero-padding based on the delay parameter.
        Also prepares input/target split for autoregressive training.

        Args:
            audio_codes: Audio codes (B, C, T) - raw codes without special tokens
            audio_codes_lens: Length of audio codes for each batch item (B,)
            delay: Number of zero positions to prepend for each batch item (B,).
                   In full mode: context_lens + text_lens + speech_delay
                   In streaming mode: context_lens + speech_delay

        Returns:
            Tuple of:
                - audio_channel_embedding: Audio embeddings with zero-padded delay (B, T_delay + T_audio, E)
                - audio_channel_lens: Total length of audio channel for each batch item (B,)
                - audio_codes_target: Target audio codes for loss computation (B, C, T'-1)
                - audio_codes_lens_target: Length of target audio codes (B,)
        """
        batch_size = audio_codes.size(0)
        device = audio_codes.device

        # Apply codec conversion if configured
        if self._codec_converter is not None:
            audio_codes = self._codec_converter.convert_original_to_new(
                audio_tokens=audio_codes, audio_lens=audio_codes_lens
            ).long()

        # Add BOS and EOS tokens
        audio_codes, audio_codes_lens = self.add_special_tokens(
            codes=audio_codes,
            codes_len=audio_codes_lens,
            bos_id=self.audio_bos_id,
            eos_id=self.audio_eos_id,
        )

        # Stack audio codes across codebooks
        audio_codes, audio_codes_lens = self.stack_codes(
            audio_codes,
            audio_codes_lens,
            self.audio_bos_id,
            self.audio_eos_id,
            self.frame_stacking_factor,
            self.num_audio_codebooks,
        )

        # Prepare input and target for autoregressive training
        # Input: all tokens except the last (teacher forcing)
        # Target: all tokens except the first (shifted by one)
        audio_codes_lens_target = audio_codes_lens - 1
        audio_codes_target = audio_codes[:, :, 1:]  # (B, C, T'-1)
        audio_codes_input = audio_codes[:, :, :-1]  # (B, C, T'-1)

        # Embed audio tokens
        audio_embedded = self.embed_audio_tokens(audio_codes_input)  # (B, T'-1, E)

        # Create zero tensor for delay padding
        max_delay = delay.max().item()
        zero_delay_tensor = torch.zeros(batch_size, max_delay, self.cfg.embedding_dim, device=device)

        # Join delay zeros with audio embeddings
        audio_channel_embedding, audio_channel_lens = self.join_embeddings_temporally(
            embeddings=[zero_delay_tensor, audio_embedded],
            lengths=[delay, audio_codes_lens_target],
        )

        return audio_channel_embedding, audio_channel_lens, audio_codes_target, audio_codes_lens_target

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
        """
        Stack multiple time steps into the channel dimension to reduce sequence length.

        This function reshapes audio/phoneme codes by grouping consecutive time steps together
        and placing them in the channel dimension. This allows the model to process multiple
        frames in parallel while reducing the sequence length.

        Args:
            codes: Input codes tensor of shape (B, C, T) where B is batch size,
                   C is number of codebooks, and T is sequence length.
            codes_lens: Length of valid codes for each batch item, shape (B,).
            bos_id: Beginning-of-sequence token ID used to detect and handle BOS tokens.
            eos_id: End-of-sequence token ID used for padding.
            stacking_factor: Number of time steps to stack together. If 1, no stacking is performed.
            num_codebooks: Number of codebooks in the input.

        Returns:
            Tuple of:
                - stacked_codes: Reshaped codes of shape (B, C * stacking_factor, T // stacking_factor).
                  If input contains BOS tokens, they are preserved at the beginning.
                - new_lens: Updated sequence lengths after stacking, shape (B,).
        """
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
        """
        Reverse the stacking operation to recover the original time dimension.

        This is the inverse of `stack_codes`. It takes codes that have been stacked
        in the channel dimension and expands them back into the time dimension.

        Args:
            stacked_codes: Stacked codes tensor of shape (B, C * stacking_factor, T_stacked)
                          where T_stacked = T_original // stacking_factor.
            stacked_lens: Length of valid stacked sequences for each batch item, shape (B,).
            stacking_factor: The stacking factor used in the original `stack_codes` call.
                            If 1, no unstacking is performed.

        Returns:
            Tuple of:
                - unstacked_codes: Codes with restored time dimension, shape (B, C, T_stacked * stacking_factor).
                - orig_lens: Recovered sequence lengths, shape (B,). Note that these are the
                  maximum possible lengths; actual valid lengths may be shorter due to
                  padding applied during stacking.
        """
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

    def process_batch(
        self,
        text: torch.Tensor,
        text_lens: torch.Tensor,
        context_text_tokens: torch.Tensor,
        context_text_tokens_lens: torch.Tensor,
        audio_codes: torch.Tensor,
        audio_codes_lens: torch.Tensor,
        context_audio_codes: torch.Tensor,
        context_audio_codes_lens: torch.Tensor,
        phoneme_tokens: Optional[torch.Tensor] = None,
        phoneme_tokens_lens: Optional[torch.Tensor] = None,
        mode: str = "train",
        training_mode: Optional[TrainingMode] = None,
    ) -> ProcessBatchOutput:
        """
        Simplified batch processing using channel-based embedding architecture.

        This function provides a cleaner implementation of process_batch where:
        1. Context is prepared separately (without text)
        2. Text, phoneme, and audio are each treated as channels with delay-based alignment
        3. Channels are summed element-wise and joined temporally with context

        The delay handling ensures proper temporal alignment:
        - Text channel delay: context_lens (no additional delay)
        - Phoneme channel delay: context_lens + phoneme_delay
        - Audio channel delay: context_lens + text_lens + speech_delay (full mode)
                              or context_lens + speech_delay (streaming mode)

        Args:
            text: Input text token IDs (B, L)
            text_lens: Length of text for each batch item (B,)
            context_text_tokens: Context text token IDs for conditioning (B, L_ctx)
            context_text_tokens_lens: Length of context text (B,)
            audio_codes: Audio codes (B, C, T) - raw codes without special tokens
            audio_codes_lens: Length of audio codes (B,)
            context_audio_codes: Pre-computed context audio codes (B, C, T')
            context_audio_codes_lens: Length of context audio codes (B,)
            phoneme_tokens: Phoneme token IDs (optional) (B, L_phoneme)
            phoneme_tokens_lens: Length of phoneme tokens (B,)
            mode: Training mode, either "train" or "val"
            training_mode: Optional TrainingMode object

        Returns:
            ProcessBatchOutput: Contains loss values and model predictions
        """
        # Select training mode
        selected_training_mode = training_mode
        if selected_training_mode is None:
            if mode == 'train':
                selected_training_mode = random.choice(self.training_modes)
            else:
                selected_training_mode = self.training_modes[0]

        current_text_input_mode = selected_training_mode.text_input_mode
        current_streaming_speech_delay = selected_training_mode.streaming_speech_delay
        current_streaming_phonemes_delay = selected_training_mode.streaming_phonemes_delay

        # Determine dropout flags
        dropout_text_input = (random.random() < self.dropout_text_input_prob) if mode == 'train' else False
        dropout_phoneme_input = (random.random() < self.dropout_phoneme_input_prob) if mode == 'train' else False
        if dropout_phoneme_input and dropout_text_input:
            dropout_phoneme_input = random.random() < 0.5
            dropout_text_input = not dropout_phoneme_input

        # Determine CFG unconditional dropout
        dropout_conditional_input = False
        if mode == 'train' and self.cfg_unconditional_prob > 0.0:
            if torch.rand(1).item() < self.cfg_unconditional_prob:
                dropout_conditional_input = True

        # 1. Prepare context tensors (without text)
        context_embedding, context_lens, context_audio_codes_processed, context_audio_codes_lens_processed = (
            self.prepare_context_tensors(
                context_text_tokens=context_text_tokens,
                context_text_tokens_lens=context_text_tokens_lens,
                context_audio_codes=context_audio_codes,
                context_audio_codes_lens=context_audio_codes_lens,
                training_mode=selected_training_mode,
                dropout_conditional_input=dropout_conditional_input,
            )
        )

        # 2. Compute delays for each channel based on mode
        # Text channel delay: always context_lens
        text_delay = context_lens.clone()

        # Phoneme channel delay: context_lens + phoneme_delay (both modes)
        phoneme_delay = context_lens + current_streaming_phonemes_delay

        # Audio channel delay depends on mode
        if current_text_input_mode == 'full':
            # Full mode: context_lens + text_lens + speech_delay
            audio_delay = context_lens + text_lens + current_streaming_speech_delay
        else:
            # Streaming mode: context_lens + speech_delay
            audio_delay = context_lens + current_streaming_speech_delay

        # 3. Prepare text channel embeddings
        text_channel_embedding, text_channel_lens = self.prepare_text_channel_embeddings(
            text=text,
            text_lens=text_lens,
            delay=text_delay,
            dropout_text_input=dropout_text_input or dropout_conditional_input,
        )

        # 4. Prepare phoneme channel embeddings (if phoneme tokenizer is configured)
        phoneme_channel_embedding = None
        phoneme_tokens_stacked = None
        phoneme_tokens_lens_stacked = None
        if self.phoneme_tokenizer is not None and phoneme_tokens is not None:
            (
                phoneme_channel_embedding,
                phoneme_channel_lens,
                phoneme_tokens_stacked,
                phoneme_tokens_lens_stacked,
            ) = self.prepare_phoneme_channel_embeddings(
                phoneme_tokens=phoneme_tokens,
                phoneme_tokens_lens=phoneme_tokens_lens,
                delay=phoneme_delay,
                dropout_phoneme_input=dropout_phoneme_input or dropout_conditional_input,
            )

        # 5. Prepare audio channel embeddings
        (
            audio_channel_embedding,
            audio_channel_lens,
            audio_codes_target,
            audio_codes_lens_target,
        ) = self.prepare_audio_channel_embeddings(
            audio_codes=audio_codes,
            audio_codes_lens=audio_codes_lens,
            delay=audio_delay,
        )

        # 6. Sum the channel embeddings element-wise
        # First, align all channels to the same length (max of all channel lengths)
        max_channel_len = max(
            text_channel_embedding.size(1),
            audio_channel_embedding.size(1),
            phoneme_channel_embedding.size(1) if phoneme_channel_embedding is not None else 0,
        )

        # Pad text channel if needed
        if text_channel_embedding.size(1) < max_channel_len:
            padding = torch.zeros(
                text_channel_embedding.size(0),
                max_channel_len - text_channel_embedding.size(1),
                text_channel_embedding.size(2),
                device=text_channel_embedding.device,
            )
            text_channel_embedding = torch.cat([text_channel_embedding, padding], dim=1)

        # Pad audio channel if needed
        if audio_channel_embedding.size(1) < max_channel_len:
            padding = torch.zeros(
                audio_channel_embedding.size(0),
                max_channel_len - audio_channel_embedding.size(1),
                audio_channel_embedding.size(2),
                device=audio_channel_embedding.device,
            )
            audio_channel_embedding = torch.cat([audio_channel_embedding, padding], dim=1)

        # Sum channels
        combined_channel_embedding = text_channel_embedding + audio_channel_embedding

        # Add phoneme channel if available
        if phoneme_channel_embedding is not None:
            if phoneme_channel_embedding.size(1) < max_channel_len:
                padding = torch.zeros(
                    phoneme_channel_embedding.size(0),
                    max_channel_len - phoneme_channel_embedding.size(1),
                    phoneme_channel_embedding.size(2),
                    device=phoneme_channel_embedding.device,
                )
                phoneme_channel_embedding = torch.cat([phoneme_channel_embedding, padding], dim=1)
            combined_channel_embedding = combined_channel_embedding + phoneme_channel_embedding

        # 7. Join context with combined channel embeddings
        # The combined_channel_lens is the max of all channel lens for each batch item
        combined_channel_lens = (
            torch.stack(
                [
                    text_channel_lens,
                    audio_channel_lens,
                    phoneme_channel_lens if phoneme_channel_embedding is not None else audio_channel_lens,
                ],
                dim=0,
            )
            .max(dim=0)
            .values
        )

        # Right pad context embedding
        context_padding = torch.zeros(
            context_embedding.size(0),
            combined_channel_embedding.size(1) - context_embedding.size(1),
            context_embedding.size(2),
            device=context_embedding.device,
        )
        context_embedding_padded = torch.cat([context_embedding, context_padding], dim=1)

        full_embedding = context_embedding_padded + combined_channel_embedding

        # 8. Forward pass through transformer
        transformer_out = self.forward(
            inputs_embeds=full_embedding,
            attention_mask=get_mask_from_lengths(combined_channel_lens),
        )
        transformer_hidden_states = transformer_out.last_hidden_state  # (B, T_total, E)

        # 9. Extract prediction embeddings and compute losses
        # Audio predictions start at audio_delay
        pred_embeddings = self.slice_pred_embeddings(
            transformer_hidden_states,
            context_lens=audio_delay,
            target_lens=audio_codes_lens_target,
        )

        # Project to audio logits
        pred_embeddings_audio = self.audio_out_projection(pred_embeddings)
        logits = self.final_proj(pred_embeddings_audio)

        # Compute codebook loss
        codebook_loss, _ = self.compute_loss(logits, audio_codes_target, audio_codes_lens_target)
        loss = codebook_loss

        # Compute local transformer loss if applicable
        local_transformer_loss = None
        local_transformer_logits = None
        if self.local_transformer_type != LocalTransformerType.NO_LT:
            assert self.local_transformer_type == LocalTransformerType.AR, "Unexpected local transformer type"
            local_transformer_logits = self.compute_local_transformer_logits(
                pred_embeddings, audio_codes_target, targets_offset_by_one=False
            )
            local_transformer_loss, _ = self.compute_loss(
                local_transformer_logits, audio_codes_target, audio_codes_lens_target
            )
            local_transformer_loss_scale = self.cfg.get('local_transformer_loss_scale', 1.0)
            loss = loss + local_transformer_loss_scale * local_transformer_loss

        # Compute phoneme loss if applicable
        phoneme_loss = None
        if self.phoneme_tokenizer is not None and phoneme_tokens_stacked is not None:
            # Phoneme predictions start at phoneme_delay
            pred_embeddings_phoneme = self.slice_pred_embeddings(
                transformer_hidden_states,
                context_lens=phoneme_delay,
                target_lens=phoneme_tokens_lens_stacked - 1,
            )
            phoneme_logits = self.phoneme_final_proj(pred_embeddings_phoneme)

            if not (dropout_conditional_input or dropout_text_input or dropout_phoneme_input):
                phoneme_loss, _ = self.compute_phoneme_loss(
                    phoneme_logits, phoneme_tokens_stacked[:, :, 1:].long(), phoneme_tokens_lens_stacked - 1
                )
                print("No Dropout - phoneme loss:", phoneme_loss.item())
            else:
                phoneme_loss = torch.tensor(0.0, device=logits.device)
                print("Dropout - phoneme loss skipped", phoneme_loss.item())

            loss = loss + phoneme_loss

        return ProcessBatchOutput(
            loss=loss,
            codebook_loss=codebook_loss,
            phoneme_loss=phoneme_loss,
            local_transformer_loss=local_transformer_loss,
            local_transformer_logits=local_transformer_logits,
            logits=logits,
            audio_codes_target=audio_codes_target,
            audio_codes_lens_target=audio_codes_lens_target,
            context_audio_codes=context_audio_codes_processed,
            context_audio_codes_lens=context_audio_codes_lens_processed,
            selected_training_mode=selected_training_mode.name if selected_training_mode is not None else None,
        )

    def training_step(self, batch, batch_idx):
        if 'context_audio_codes' in batch:
            context_audio_codes = batch['context_audio_codes']
            context_audio_codes_lens = batch['context_audio_codes_lens']
        else:
            context_audio = batch['context_audio']
            context_audio_lens = batch['context_audio_lens']
            context_audio_codes, context_audio_codes_lens = self.audio_to_codes(context_audio, context_audio_lens)

        if 'audio_codes' in batch:
            audio_codes = batch['audio_codes']
            audio_codes_lens = batch['audio_codes_lens']
        else:
            audio = batch['audio']
            audio_lens = batch['audio_lens']
            audio_codes, audio_codes_lens = self.audio_to_codes(audio, audio_lens)

        batch_output = self.process_batch(
            text=batch['text'],
            text_lens=batch['text_lens'],
            context_text_tokens=batch['context_text_tokens'],
            context_text_tokens_lens=batch['context_text_tokens_lens'],
            audio_codes=audio_codes,
            audio_codes_lens=audio_codes_lens,
            context_audio_codes=context_audio_codes,
            context_audio_codes_lens=context_audio_codes_lens,
            phoneme_tokens=batch.get('phoneme_tokens'),
            phoneme_tokens_lens=batch.get('phoneme_tokens_lens'),
            mode="train",
        )
        loss = batch_output.loss
        codebook_loss = batch_output.codebook_loss
        self.log('train/codebook_loss', codebook_loss, prog_bar=True, sync_dist=True)
        self.log('train/loss', loss, prog_bar=True, sync_dist=True)

        if self.phoneme_tokenizer is not None:
            phoneme_loss = batch_output.phoneme_loss
            self.log('train/phoneme_loss', phoneme_loss, prog_bar=True, sync_dist=True)

        local_transformer_loss = batch_output.local_transformer_loss
        if local_transformer_loss is not None:
            self.log('train/local_transformer_loss', local_transformer_loss, prog_bar=True, sync_dist=True)

        # Log training mode info for multi-mode training
        if batch_output.selected_training_mode is not None:
            # Log which mode was selected for this batch
            # Convert mode name to an index for logging
            mode_idx = self.mode_name_to_mode[batch_output.selected_training_mode].mode_idx
            self.log('train/training_mode_idx', float(mode_idx), on_step=True)

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
        # Extract inputs from batch and pass explicitly to process_batch
        if 'context_audio_codes' in batch:
            context_audio_codes = batch['context_audio_codes']
            context_audio_codes_lens = batch['context_audio_codes_lens']
        else:
            context_audio = batch['context_audio']
            context_audio_lens = batch['context_audio_lens']
            context_audio_codes, context_audio_codes_lens = self.audio_to_codes(context_audio, context_audio_lens)

        if 'audio_codes' in batch:
            audio_codes = batch['audio_codes']
            audio_codes_lens = batch['audio_codes_lens']
        else:
            audio = batch['audio']
            audio_lens = batch['audio_lens']
            audio_codes, audio_codes_lens = self.audio_to_codes(audio, audio_lens)

        batch_output = self.process_batch(
            text=batch['text'],
            text_lens=batch['text_lens'],
            context_text_tokens=batch['context_text_tokens'],
            context_text_tokens_lens=batch['context_text_tokens_lens'],
            audio_codes=audio_codes,
            audio_codes_lens=audio_codes_lens,
            context_audio_codes=context_audio_codes,
            context_audio_codes_lens=context_audio_codes_lens,
            phoneme_tokens=batch.get('phoneme_tokens'),
            phoneme_tokens_lens=batch.get('phoneme_tokens_lens'),
            mode="val",
        )
        # Access ProcessBatchOutput dataclass attributes
        # logits come from the parallel prediction head
        # If using local_transformer, local_transformer_logits are also available
        loss = batch_output.loss
        codebook_loss = batch_output.codebook_loss
        logits = batch_output.logits
        audio_codes_target = batch_output.audio_codes_target
        audio_codes_lens_target = batch_output.audio_codes_lens_target
        context_audio_codes = batch_output.context_audio_codes
        context_audio_codes_lens = batch_output.context_audio_codes_lens

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

        local_transformer_loss = batch_output.local_transformer_loss
        val_output = {
            'val_loss': loss,
            'val_codebook_loss': codebook_loss,
            'val_local_transformer_loss': local_transformer_loss,
        }

        if self.phoneme_tokenizer is not None:
            phoneme_loss = batch_output.phoneme_loss
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

    def _sample_audio_codes(
        self,
        last_hidden: torch.Tensor,
        all_code_logits_t: torch.Tensor,
        temperature: float,
        topk: int,
        use_local_transformer_for_inference: bool,
        use_cfg: bool,
        cfg_scale: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample audio codes from logits using either local transformer or parallel sampling.

        Returns:
            audio_codes_next: Sampled codes with temperature/topk (B, num_codebooks)
            all_codes_next_argmax: Argmax sampled codes for EOS detection (B, num_codebooks)
        """
        if use_local_transformer_for_inference:
            if self.local_transformer_type == LocalTransformerType.AR:
                audio_codes_next = self.local_transformer_sample_autoregressive(
                    dec_output=last_hidden[:, -1, :],
                    temperature=temperature,
                    topk=topk,
                    use_cfg=use_cfg,
                    cfg_scale=cfg_scale,
                )
            else:
                raise ValueError(
                    f"Local transformer inference requested but local transformer type is {self.local_transformer_type}"
                )
            # TODO @rfejgin: should we add argmax sampling for EOS here too?
            all_codes_next_argmax = audio_codes_next
        else:
            # Parallel sampling from all codebook logits
            audio_codes_next = self.sample_codes_from_logits(all_code_logits_t, temperature=temperature, topk=topk)
            # Argmax sampling for reliable EOS detection
            all_codes_next_argmax = self.sample_codes_from_logits(all_code_logits_t, temperature=0.01)

        return audio_codes_next, all_codes_next_argmax

    def streaming_init(
        self,
        context_audio_codes: torch.Tensor,
        context_audio_codes_lens: torch.Tensor,
        context_text_tokens: torch.Tensor,
        context_text_tokens_lens: torch.Tensor,
        inference_mode: Optional[str] = None,
        use_cfg: bool = False,
        cfg_scale: float = 1.0,
        use_local_transformer: bool = False,
        temperature: float = 0.7,
        topk: int = 80,
        phoneme_input_type: str = 'predicted',
        phoneme_sampling_method: str = 'argmax',
        gt_phoneme_tokens: Optional[torch.Tensor] = None,
        gt_phoneme_tokens_lens: Optional[torch.Tensor] = None,
    ) -> StreamingState:
        """
        Initialize streaming TTS inference state.

        This prepares the model for streaming inference by processing the context
        (audio + context text) and returning a StreamingState that can be used
        with streaming_step() to incrementally generate audio.

        Note: This function does NOT take the main text input. Text tokens are
        provided incrementally via streaming_step().

        For batched inference, each batch item can have a different context length.
        This function processes only up to the minimum context length across the batch,
        storing the remaining context to be processed in streaming_step's context phase.

        The streaming inference follows phases (per batch item):
        1. Context phase: Processing remaining context (if any) for items with longer context.
        2. Prompt phase: First `streaming_speech_delay` text tokens are processed
           without generating audio (building up context).
        3. Generation phase: Audio BOS is added and audio codes are generated
           autoregressively, with remaining text tokens added to audio embeddings.

        Args:
            context_audio_codes: Pre-computed audio codes for context audio (B, C, T').
            context_audio_codes_lens: Length of context audio codes (B,).
            context_text_tokens: Context text token IDs for speaker/style conditioning (B, L).
            context_text_tokens_lens: Length of context text (B,).
            inference_mode: Name of the inference mode to use (e.g., "streaming_4_8").
                If None, uses the default inference mode.
            use_cfg: Whether to use classifier-free guidance.
            cfg_scale: CFG scale factor (higher = stronger conditioning).
            use_local_transformer: Whether to use local transformer for AR sampling.
            temperature: Sampling temperature for audio codes.
            topk: Top-k sampling parameter.
            phoneme_input_type: 'gt' or 'predicted' for phoneme tokens (use 'predicted' for streaming).
            phoneme_sampling_method: 'argmax' or 'sample' for phoneme token selection.
            gt_phoneme_tokens: Optional GT phoneme tokens (B, L) with BOS/EOS for teacher forcing.
            gt_phoneme_tokens_lens: Lengths of GT phoneme tokens (B,).

        Returns:
            StreamingState: Initial state for streaming inference.
        """
        with torch.inference_mode():
            batch_size = context_audio_codes.size(0)
            device = context_audio_codes.device

            # Resolve inference mode
            mode_name = inference_mode if inference_mode is not None else self.default_inference_mode
            if mode_name not in self.mode_name_to_mode:
                available_modes = list(self.mode_name_to_mode.keys())
                raise ValueError(f"Unknown inference mode '{mode_name}'. Available modes: {available_modes}")

            selected_training_mode = self.mode_name_to_mode[mode_name]

            # Prepare context embedding using shared helper
            context_embedding, context_lens, context_audio_codes, context_audio_codes_lens = (
                self.prepare_context_tensors(
                    context_text_tokens=context_text_tokens,
                    context_text_tokens_lens=context_text_tokens_lens,
                    context_audio_codes=context_audio_codes,
                    context_audio_codes_lens=context_audio_codes_lens,
                    training_mode=selected_training_mode,
                    dropout_conditional_input=False,
                )
            )

            # Store full context embedding and lens before any CFG manipulation
            full_context_embedding = context_embedding.clone()  # (B, T_max, E)
            full_context_lens = context_lens.clone()  # (B,)

            # Compute min context length - we only process up to this in init
            min_context_len = context_lens.min().item()

            # Setup classifier-free guidance if enabled
            dummy_context_embedding_unconditional = None
            if use_cfg:
                dummy_context_embedding_unconditional = self.decoder.get_input_embeddings()(
                    torch.full((1, 1), self.cfg_unk_token_id, device=device)
                )
                # Create unconditional context (same length as conditional)
                dummy_context_expanded = dummy_context_embedding_unconditional.expand(
                    batch_size, context_embedding.size(1), -1
                )
                # Concatenate conditional and unconditional: (2*B, T, E)
                context_embedding = torch.cat([context_embedding, dummy_context_expanded], dim=0)

            # First forward pass to process context - only up to min_context_len
            cache_position = torch.arange(min_context_len, device=device)
            transformer_out = self.forward(
                inputs_embeds=context_embedding[:, :min_context_len, :],
                attention_mask=None,
                use_cache=True,
                past_key_values=None,
                cache_position=cache_position,
            )

            last_hidden = transformer_out.last_hidden_state
            past_kv = transformer_out.past_key_values
            current_cache_seq_len = min_context_len

            # Process GT phoneme tokens if provided (for teacher forcing)
            gt_phoneme_embeddings = None
            gt_phoneme_lens = None
            if gt_phoneme_tokens is not None and gt_phoneme_tokens_lens is not None:
                gt_phoneme_expanded = gt_phoneme_tokens.unsqueeze(1)  # (B, 1, L)
                gt_phoneme_stacked, gt_phoneme_lens = self.stack_codes(
                    gt_phoneme_expanded,
                    gt_phoneme_tokens_lens,
                    self.phoneme_tokenizer.bos_token_id,
                    self.phoneme_tokenizer.eos_token_id,
                    self.phoneme_stacking_factor,
                    1,
                )
                gt_phoneme_embeddings = self.embed_phoneme_tokens(gt_phoneme_stacked)  # (B, T', E)

            # Initialize streaming state with batch support
            state = StreamingState(
                batch_size=batch_size,
                past_key_values=past_kv,
                cache_seq_len=current_cache_seq_len,
                all_predictions=[],
                all_phoneme_predictions=[],
                context_audio_codes=context_audio_codes,
                context_audio_codes_lens=context_audio_codes_lens,
                context_lens=context_lens,
                full_context_embedding=full_context_embedding,
                full_context_lens=full_context_lens,
                context_position=torch.full((batch_size,), min_context_len, dtype=torch.long, device=device),
                text_tokens_seen=torch.zeros(batch_size, dtype=torch.long, device=device),
                phoneme_steps=torch.zeros(batch_size, dtype=torch.long, device=device),
                audio_steps=torch.zeros(batch_size, dtype=torch.long, device=device),
                phoneme_stream_ended=torch.zeros(batch_size, dtype=torch.bool, device=device),
                finished=torch.zeros(batch_size, dtype=torch.bool, device=device),
                device=device,
                training_mode=selected_training_mode,
                use_cfg=use_cfg,
                cfg_scale=cfg_scale,
                use_local_transformer=use_local_transformer,
                temperature=temperature,
                topk=topk,
                dummy_context_embedding_unconditional=dummy_context_embedding_unconditional,
                last_hidden=last_hidden,
                text_finished=torch.zeros(batch_size, dtype=torch.bool, device=device),
                phoneme_input_type=phoneme_input_type,
                phoneme_sampling_method=phoneme_sampling_method,
                last_phoneme_tokens=None,
                last_audio_codes=None,
                audio_prediction_start_idx=torch.full((batch_size,), -1, dtype=torch.long, device=device),
                audio_prediction_end_idx=torch.full((batch_size,), -1, dtype=torch.long, device=device),
                phoneme_prediction_start_idx=torch.full((batch_size,), -1, dtype=torch.long, device=device),
                phoneme_prediction_end_idx=torch.full((batch_size,), -1, dtype=torch.long, device=device),
                gt_phoneme_embeddings=gt_phoneme_embeddings,
                gt_phoneme_lens=gt_phoneme_lens,
            )

            return state

    def streaming_step(
        self,
        state: StreamingState,
        text_tokens: Optional[torch.Tensor] = None,
        force_dropout_text: bool = False,
    ) -> Tuple[StreamingState, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform one streaming inference step with batch support.

        This function processes one text token per batch item (or signals end of text with None)
        and generates predictions according to the streaming delays. Each batch item can be
        in a different phase.

        The streaming operates in four phases per batch item:
        1. Context phase (context_position < full_context_lens):
           - Still processing remaining context from streaming_init
           - Uses context embedding, ignores text_tokens for this item
        2. Prompt phase (text_tokens_seen < phoneme_delay):
           - Only text tokens are processed, KV cache is extended
           - No phoneme or audio predictions
        3. Phoneme-only phase (phoneme_delay <= text_tokens_seen < speech_delay):
           - Starts with phoneme BOS on first step
           - Only phoneme predictions (no audio)
           - Input: text embedding + phoneme embedding
        4. Audio phase (text_tokens_seen >= speech_delay):
           - Starts with audio BOS on first step
           - Both phoneme and audio predictions
           - Input: text embedding + phoneme embedding + audio embedding

        IMPORTANT: Only ONE forward call to the decoder per streaming_step.

        Args:
            state: Current StreamingState from streaming_init or previous streaming_step.
            text_tokens: Next text token for each batch item, shape (B,), or None if text has finished.
                For items still in context phase, the text_token value is ignored (can be 0).
                When None is passed, the model continues generating until EOS.

        Returns:
            Tuple of:
                - Updated StreamingState
                - Predicted audio codes for this step (B, C, S) unstacked, or None if no items in audio phase
                  where C = num_audio_codebooks and S = frame_stacking_factor
                - Predicted phoneme tokens for this step (B, phoneme_stacking_factor) or None if no items in phoneme phase
        """
        if state.finished.all():
            return state, None, None

        with torch.inference_mode():
            device = state.device
            batch_size = state.batch_size
            streaming_speech_delay = state.training_mode.streaming_speech_delay
            streaming_phonemes_delay = state.training_mode.streaming_phonemes_delay

            # ==================== DETERMINE PHASES PER BATCH ITEM ====================
            needs_context = state.context_position < state.full_context_lens  # (B,) bool
            needs_text = (~needs_context) & (~state.text_finished)
            needs_phoneme = (state.text_tokens_seen >= streaming_phonemes_delay) & (~state.phoneme_stream_ended)
            needs_audio = (state.text_tokens_seen >= streaming_speech_delay) & (~state.finished)

            next_input = torch.zeros(batch_size, 1, self.cfg.embedding_dim, device=device)
            # --- Context phase items: use next context embedding ---
            if needs_context.any():
                # Gather context embeddings at current position for each item
                # context_position: (B,) - position indices
                # full_context_embedding: (B, T_max, E)
                ctx_positions = state.context_position.clone()  # (B,)
                # Clamp positions to valid range for gathering
                ctx_positions = ctx_positions.clamp(max=state.full_context_embedding.size(1) - 1)
                # Gather: need (B, 1, E) from (B, T, E) at positions (B,)
                ctx_emb = state.full_context_embedding[
                    torch.arange(batch_size, device=device), ctx_positions, :
                ].unsqueeze(
                    1
                )  # (B, 1, E)
                # Only apply to items in context phase
                context_mask = needs_context.view(batch_size, 1, 1).float()
                next_input = next_input + ctx_emb * context_mask

            # --- Non-context phase items: handle text embedding ---
            text_embedded = None
            if text_tokens is not None and needs_text.any():
                # Embed text tokens for all items (will be masked later)
                text_tokens_2d = text_tokens.unsqueeze(1)  # (B, 1)
                text_embedded = self.decoder.get_input_embeddings()(text_tokens_2d)  # (B, 1, E)

                # Handle BPE char tokenizer
                if self.use_bpe_char_tokenizer:
                    text_mask = torch.ones_like(text_tokens_2d, dtype=torch.bool)
                    cas_embedding = self.cas_encoder(text_tokens_2d, subword_mask=text_mask)  # (B, 1, E)
                    text_embedded = text_embedded + cas_embedding

                if force_dropout_text:
                    text_embedded = text_embedded * 0

                text_add_mask = needs_text.view(batch_size, 1, 1).float()
                next_input = next_input + text_embedded * text_add_mask
                # Check for EOS tokens - mark those items as text_finished
                # Items that receive EOS should not have their text embedded added after this step
                is_eos_token = text_tokens == self.eos_id  # (B,) bool
                state.text_finished = state.text_finished | is_eos_token

            elif text_tokens is None:
                # Text finished signal for items not in context phase
                state.text_finished = state.text_finished | ~needs_context

            # --- Phoneme embedding for phoneme and audio phase items ---
            if self.phoneme_tokenizer is not None:
                if needs_phoneme.any():
                    phoneme_emb = torch.zeros(batch_size, 1, self.cfg.embedding_dim, device=device)

                    if state.phoneme_input_type == 'gt' and state.gt_phoneme_embeddings is not None:
                        # Teacher forcing: use pre-computed GT phoneme embeddings
                        # Only use GT embedding if within valid length, otherwise zero
                        within_gt_len = state.phoneme_steps < state.gt_phoneme_lens  # (B,)
                        positions = state.phoneme_steps.clamp(max=state.gt_phoneme_embeddings.size(1) - 1)
                        gt_emb = state.gt_phoneme_embeddings[
                            torch.arange(batch_size, device=device), positions, :
                        ].unsqueeze(
                            1
                        )  # (B, 1, E)
                        phoneme_mask = (needs_phoneme & within_gt_len).view(batch_size, 1, 1).float()
                        phoneme_emb = phoneme_emb + gt_emb * phoneme_mask
                    else:
                        # Prediction mode: use BOS or last predicted phoneme
                        first_phoneme_step = needs_phoneme & (state.phoneme_steps == 0)
                        has_last_phoneme = (
                            needs_phoneme & ~first_phoneme_step & (state.last_phoneme_tokens is not None)
                        )

                        if first_phoneme_step.any():
                            phoneme_bos = torch.full(
                                (batch_size, self.phoneme_stacking_factor, 1),
                                self.phoneme_tokenizer.bos_token_id,
                                device=device,
                            ).long()
                            phoneme_bos_emb = self.embed_phoneme_tokens(phoneme_bos)  # (B, 1, E)
                            first_mask = first_phoneme_step.view(batch_size, 1, 1).float()
                            phoneme_emb = phoneme_emb + phoneme_bos_emb * first_mask

                        if has_last_phoneme.any() and state.last_phoneme_tokens is not None:
                            last_phoneme_emb = self.embed_phoneme_tokens(
                                state.last_phoneme_tokens.unsqueeze(2)
                            )  # (B, 1, E)
                            last_mask = has_last_phoneme.view(batch_size, 1, 1).float()
                            phoneme_emb = phoneme_emb + last_phoneme_emb * last_mask

                    next_input = next_input + phoneme_emb

            # --- Audio embedding for audio phase items ---
            if needs_audio.any():
                # Determine which items are at first audio step
                first_audio_step = needs_audio & (state.audio_steps == 0)
                has_last_audio = needs_audio & ~first_audio_step & (state.last_audio_codes is not None)

                audio_emb = torch.zeros(batch_size, 1, self.cfg.embedding_dim, device=device)

                if first_audio_step.any():
                    # Create BOS for items at first audio step
                    audio_bos = torch.full(
                        (batch_size, self.num_audio_codebooks * self.frame_stacking_factor, 1),
                        self.audio_bos_id,
                        device=device,
                    ).long()
                    audio_bos_emb = self.embed_audio_tokens(audio_bos)  # (B, 1, E)
                    first_mask = first_audio_step.view(batch_size, 1, 1).float()
                    audio_emb = audio_emb + audio_bos_emb * first_mask

                if has_last_audio.any() and state.last_audio_codes is not None:
                    # Use last predicted audio
                    last_audio_emb = self.embed_audio_tokens(state.last_audio_codes.unsqueeze(2))  # (B, 1, E)
                    last_mask = has_last_audio.view(batch_size, 1, 1).float()
                    audio_emb = audio_emb + last_audio_emb * last_mask

                next_input = next_input + audio_emb

            # ==================== HANDLE CFG ====================
            if state.use_cfg:
                # For unconditional branch, use dummy embedding for non-audio items
                # and audio-only embedding for audio items
                next_input_unconditional_context = state.dummy_context_embedding_unconditional.expand(
                    batch_size, 1, -1
                )
                # After the context is finished, we use zero embedding for the unconditional branch until audio phase starts
                next_input_unconditional_zeros = torch.zeros_like(next_input_unconditional_context)
                context_mask = needs_context.view(batch_size, 1, 1).float()
                next_input_unconditional = (
                    context_mask * next_input_unconditional_context
                    + (1 - context_mask) * next_input_unconditional_zeros
                )

                # For audio phase items, we use audio embedding for the unconditional branch
                if needs_audio.any():
                    audio_mask = needs_audio.view(batch_size, 1, 1).float()
                    next_input_unconditional = next_input_unconditional * (1 - audio_mask) + audio_emb * audio_mask

                # Concatenate conditional and unconditional: (2*B, 1, E)
                next_input = torch.cat([next_input, next_input_unconditional], dim=0)

            # ==================== FORWARD PASS ====================
            cache_position = torch.tensor([state.cache_seq_len], device=device)
            transformer_out = self.forward(
                inputs_embeds=next_input,
                attention_mask=None,
                use_cache=True,
                past_key_values=state.past_key_values,
                cache_position=cache_position,
            )

            state.last_hidden = transformer_out.last_hidden_state
            state.past_key_values = transformer_out.past_key_values
            state.cache_seq_len += 1

            # ==================== UPDATE STATE ====================
            # Update context_position for items in context phase
            state.context_position = state.context_position + needs_context.long()
            # Keep updating text_tokens_seen for items once the context is finished
            # This is because this counter is used to determine when to start predicting phonemes and audio
            state.text_tokens_seen = state.text_tokens_seen + (~needs_context).long()

            # Update phoneme_steps for items in phoneme or audio phase
            state.phoneme_steps = state.phoneme_steps + needs_phoneme.long()

            # Update audio_steps for items in audio phase
            state.audio_steps = state.audio_steps + needs_audio.long()

            # ==================== PREDICTIONS ====================
            pred_phoneme_tokens = None
            audio_codes_next = None

            # Phoneme predictions for items in phoneme or audio phase
            if needs_phoneme.any() and self.phoneme_tokenizer is not None:
                # Track phoneme prediction start index for items just entering phoneme phase
                first_phoneme_step = needs_phoneme & (state.phoneme_prediction_start_idx == -1)
                if first_phoneme_step.any():
                    current_phoneme_step_idx = len(state.all_phoneme_predictions)  # before append
                    state.phoneme_prediction_start_idx = torch.where(
                        first_phoneme_step,
                        torch.full_like(state.phoneme_prediction_start_idx, current_phoneme_step_idx),
                        state.phoneme_prediction_start_idx,
                    )

                # Check which items should predict phonemes (not ended)
                pred_phoneme_tokens = self._predict_phoneme_tokens(state)  # (B, phoneme_stacking_factor)
                state.last_phoneme_tokens = pred_phoneme_tokens
                state.all_phoneme_predictions.append(pred_phoneme_tokens)

                # Check for phoneme EOS per item
                phoneme_eos_detected = needs_phoneme & (
                    pred_phoneme_tokens == self.phoneme_tokenizer.eos_token_id
                ).any(
                    dim=1
                )  # (B,)
                state.phoneme_stream_ended = state.phoneme_stream_ended | phoneme_eos_detected

                # Track phoneme prediction end index for items that just ended
                newly_ended_phoneme = phoneme_eos_detected & (state.phoneme_prediction_end_idx == -1)
                if newly_ended_phoneme.any():
                    current_phoneme_step_idx = len(state.all_phoneme_predictions)  # after append
                    state.phoneme_prediction_end_idx = torch.where(
                        newly_ended_phoneme,
                        torch.full_like(state.phoneme_prediction_end_idx, current_phoneme_step_idx),
                        state.phoneme_prediction_end_idx,
                    )

            # Audio predictions for items in audio phase
            if needs_audio.any():
                # Track audio prediction start index for items just entering audio phase
                first_audio_step = needs_audio & (state.audio_prediction_start_idx == -1)
                if first_audio_step.any():
                    # Track start in terms of frames (not steps)
                    current_frame_idx = sum(p.size(-1) for p in state.all_predictions)  # total frames so far
                    state.audio_prediction_start_idx = torch.where(
                        first_audio_step,
                        torch.full_like(state.audio_prediction_start_idx, current_frame_idx),
                        state.audio_prediction_start_idx,
                    )

                audio_codes_next_stacked, all_codes_next_argmax = self._predict_audio_codes(state)  # (B, C*S)

                # Unstack immediately: (B, C*S) -> (B, C, S) where S = frame_stacking_factor
                S = self.frame_stacking_factor
                C = self.num_audio_codebooks
                audio_codes_unstacked = audio_codes_next_stacked.view(batch_size, C, S)  # (B, C, S)

                # Update last_audio_codes with stacked format (needed for next step's embedding)
                if state.last_audio_codes is None:
                    state.last_audio_codes = audio_codes_next_stacked
                else:
                    update_mask = needs_audio.view(batch_size, 1).expand_as(audio_codes_next_stacked)
                    state.last_audio_codes = torch.where(update_mask, audio_codes_next_stacked, state.last_audio_codes)

                # Check for EOS in each frame and track exact end position
                # all_codes_next_argmax is also (B, C*S), reshape to (B, C, S)
                all_codes_argmax_unstacked = all_codes_next_argmax.view(batch_size, C, S)

                # For each batch item, find if/where EOS occurs in this step's frames
                eos_in_sampled = audio_codes_unstacked == self.audio_eos_id  # (B, C, S)
                eos_in_argmax = all_codes_argmax_unstacked == self.audio_eos_id  # (B, C, S)
                eos_any_codebook = eos_in_sampled.any(dim=1) | eos_in_argmax.any(dim=1)  # (B, S)

                # Find first frame with EOS per batch item (or S if none)
                eos_frame_idx = torch.where(
                    eos_any_codebook.any(dim=1),
                    eos_any_codebook.int().argmax(dim=1),  # first frame with EOS
                    torch.full((batch_size,), S, device=device),  # no EOS in this step
                )  # (B,)

                audio_eos_detected = eos_any_codebook.any(dim=1)  # (B,)
                state.finished = state.finished | audio_eos_detected

                # Track audio prediction end index (in frames) for items that just ended
                newly_ended_audio = audio_eos_detected & (state.audio_prediction_end_idx == -1)
                if newly_ended_audio.any():
                    # End index = current frame count + frame offset where EOS was found
                    current_frame_count = len(state.all_predictions) * self.frame_stacking_factor
                    end_frame_idx = current_frame_count + eos_frame_idx
                    state.audio_prediction_end_idx = torch.where(
                        newly_ended_audio, end_frame_idx, state.audio_prediction_end_idx
                    )

                # Store unstacked codes
                state.all_predictions.append(audio_codes_unstacked)
                audio_codes_next = audio_codes_unstacked

            return state, audio_codes_next, pred_phoneme_tokens

    def _predict_phoneme_tokens(self, state: StreamingState) -> torch.Tensor:
        """Predict phoneme tokens from the last hidden state."""
        actual_batch_size = state.batch_size
        last_hidden = state.last_hidden

        # Get phoneme logits
        all_code_logits_t_phoneme = self.phoneme_final_proj(last_hidden[:, -1, :])
        all_code_logits_t_phoneme = all_code_logits_t_phoneme[:actual_batch_size]

        # Sample phonemes
        if state.phoneme_sampling_method == 'argmax':
            pred_phoneme_tokens = self.sample_codes_from_logits_phoneme(all_code_logits_t_phoneme, temperature=0.01)
        else:
            pred_phoneme_tokens = self.sample_codes_from_logits_phoneme(
                all_code_logits_t_phoneme, temperature=state.temperature, topk=state.topk
            )
        # (B, phoneme_stacking_factor)
        return pred_phoneme_tokens

    def _predict_audio_codes(self, state: StreamingState) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict audio codes from the last hidden state."""
        actual_batch_size = state.batch_size
        last_hidden = state.last_hidden

        # Compute audio logits
        last_hidden_audio = self.audio_out_projection(last_hidden[:, -1, :])
        all_code_logits_t = self.final_proj(last_hidden_audio)

        # Apply CFG if enabled
        if state.use_cfg:
            conditional_logits = all_code_logits_t[:actual_batch_size]
            unconditional_logits = all_code_logits_t[actual_batch_size:]
            all_code_logits_t = state.cfg_scale * conditional_logits + (1.0 - state.cfg_scale) * unconditional_logits

        # Sample audio codes
        audio_codes_next, all_codes_next_argmax = self._sample_audio_codes(
            last_hidden=last_hidden,
            all_code_logits_t=all_code_logits_t,
            temperature=state.temperature,
            topk=state.topk,
            use_local_transformer_for_inference=state.use_local_transformer,
            use_cfg=state.use_cfg,
            cfg_scale=state.cfg_scale,
        )

        return audio_codes_next, all_codes_next_argmax

    def streaming_decode(
        self,
        state: StreamingState,
        previous_decode_length: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Decode accumulated audio codes to waveform, returning only the new chunk.

        WARNING: This function does not yet support batch_size > 1.
        Do not use with batched streaming inference. Use streaming_finalize instead.

        This function takes all predicted codes so far and decodes them, but only
        returns the newly generated audio portion (after previous_decode_length).

        Args:
            state: Current StreamingState containing all_predictions.
            previous_decode_length: Number of audio samples already decoded and returned
                in previous calls. Use 0 on first call.

        Returns:
            Tuple of:
                - new_audio: Newly generated audio waveform (1, new_samples)
                - new_audio_len: Length of new audio (1,)
                - total_decode_length: Total decoded length so far (use as previous_decode_length
                    for next call)
        """
        if len(state.all_predictions) == 0:
            return (
                torch.zeros(1, 0, device=state.device),
                torch.zeros(1, dtype=torch.long, device=state.device),
                previous_decode_length,
            )

        with torch.inference_mode():
            # Concatenate all predictions - each is (1, C, S), concat gives (1, C, T_total_frames)
            predicted_codes = torch.cat(state.all_predictions, dim=-1)  # (1, C, T_total_frames)
            predicted_codes_lens = torch.tensor([predicted_codes.size(-1)], device=state.device)

            # Decode to audio (codes are already unstacked, no EOS removal needed)
            audio, audio_len, _ = self.codes_to_audio(predicted_codes, predicted_codes_lens)

            # Extract only new audio
            total_decode_length = audio_len[0].item()
            if total_decode_length <= previous_decode_length:
                return (
                    torch.zeros(1, 0, device=state.device),
                    torch.zeros(1, dtype=torch.long, device=state.device),
                    previous_decode_length,
                )

            new_audio = audio[:, previous_decode_length:total_decode_length]
            new_audio_len = torch.tensor([total_decode_length - previous_decode_length], device=state.device)

            return new_audio, new_audio_len, total_decode_length

    def streaming_finalize(
        self,
        state: StreamingState,
    ) -> StreamingFinalizeOutput:
        """
        Finalize streaming and return the complete generated audio and phoneme predictions.

        This function should be called after all streaming_step() calls are complete
        (i.e., when state.finished.all() is True or max steps reached).

        Args:
            state: Final StreamingState after streaming is complete.

        Returns:
            StreamingFinalizeOutput containing audio, codes, and phoneme predictions.
        """
        batch_size = state.batch_size

        # Extract and decode phoneme predictions
        phoneme_tokens_list: List[List[int]] = []
        phoneme_text_list: List[str] = []
        if self.phoneme_tokenizer is not None and len(state.all_phoneme_predictions) > 0:
            # Stack phoneme predictions: each is (B, phoneme_stacking_factor)
            all_phonemes = torch.stack(state.all_phoneme_predictions, dim=-1)  # (B, S, T)
            for i in range(batch_size):
                start = max(0, state.phoneme_prediction_start_idx[i].item())
                end = state.phoneme_prediction_end_idx[i].item()
                if end < 0:
                    end = all_phonemes.size(-1)
                # Flatten stacked phonemes back to sequence
                tokens = all_phonemes[i, :, start:end].T.reshape(-1).tolist()
                # Remove special tokens (BOS, EOS, PAD)
                special = {self.phoneme_tokenizer.bos_token_id, self.phoneme_tokenizer.eos_token_id}
                if hasattr(self.phoneme_tokenizer, 'pad_token_id'):
                    special.add(self.phoneme_tokenizer.pad_token_id)
                tokens = [t for t in tokens if t not in special]
                phoneme_tokens_list.append(tokens)
                phoneme_text_list.append(self.phoneme_tokenizer.decode(tokens))
        else:
            phoneme_tokens_list = [[] for _ in range(batch_size)]
            phoneme_text_list = ["" for _ in range(batch_size)]

        if len(state.all_predictions) == 0:
            return StreamingFinalizeOutput(
                audio=torch.zeros(batch_size, 0, device=state.device),
                audio_len=torch.zeros(batch_size, dtype=torch.long, device=state.device),
                audio_codes=torch.zeros(batch_size, self.num_audio_codebooks, 0, device=state.device),
                audio_codes_len=torch.zeros(batch_size, dtype=torch.long, device=state.device),
                phoneme_tokens=phoneme_tokens_list,
                phoneme_text=phoneme_text_list,
            )

        with torch.inference_mode():
            # Concatenate all predictions - each is (B, C, S), concat gives (B, C, T_total_frames)
            all_codes = torch.cat(state.all_predictions, dim=-1)  # (B, C, T_total_frames)
            total_frames = all_codes.size(-1)
            num_codebooks = all_codes.size(1)

            # Start and end indices are in frames (not steps)
            # If start_idx is -1, item never started audio predictions - use 0
            # If end_idx is -1, item never ended - use total_frames
            start_indices = torch.clamp(state.audio_prediction_start_idx, min=0)
            end_indices = torch.where(
                state.audio_prediction_end_idx >= 0,
                state.audio_prediction_end_idx,
                torch.full_like(state.audio_prediction_end_idx, total_frames),
            )

            # Calculate per-item lengths (in frames)
            predicted_codes_lens = end_indices - start_indices
            max_len = predicted_codes_lens.max().item()

            # Handle case where all items have zero-length predictions
            if max_len == 0:
                return StreamingFinalizeOutput(
                    audio=torch.zeros(batch_size, 0, device=state.device),
                    audio_len=torch.zeros(batch_size, dtype=torch.long, device=state.device),
                    audio_codes=torch.zeros(batch_size, num_codebooks, 0, device=state.device, dtype=all_codes.dtype),
                    audio_codes_len=torch.zeros(batch_size, dtype=torch.long, device=state.device),
                    phoneme_tokens=phoneme_tokens_list,
                    phoneme_text=phoneme_text_list,
                )

            # Create padded output tensor and slice each item's valid predictions
            predicted_codes = torch.zeros(
                batch_size, num_codebooks, max_len, dtype=all_codes.dtype, device=state.device
            )
            for i in range(batch_size):
                start = start_indices[i].item()
                end = end_indices[i].item()
                length = end - start
                if length > 0:
                    predicted_codes[i, :, :length] = all_codes[i, :, start:end]

            # No need to remove EOS - end_indices already point to the frame before EOS
            # Decode to audio (codes are already unstacked: B, C, T)
            audio, audio_len, decoded_codes = self.codes_to_audio(predicted_codes, predicted_codes_lens)

            return StreamingFinalizeOutput(
                audio=audio,
                audio_len=audio_len,
                audio_codes=predicted_codes,
                audio_codes_len=predicted_codes_lens,
                phoneme_tokens=phoneme_tokens_list,
                phoneme_text=phoneme_text_list,
            )

    def infer_batch(
        self,
        batch: Dict[str, torch.Tensor],
        max_decoder_steps: int = 500,
        temperature: float = 0.7,
        topk: int = 80,
        use_cfg: bool = False,
        cfg_scale: float = 1.0,
        use_local_transformer_for_inference: bool = False,
        phoneme_input_type: str = 'pred',
        phoneme_sampling_method: str = 'argmax',
        force_dropout_text: bool = False,
    ) -> InferBatchOutput:
        """
        Batch inference using streaming infrastructure.

        This is a simple wrapper around streaming_init, streaming_step, and streaming_finalize
        that processes a batch dictionary similar to training_step/validation_step.

        Args:
            batch: Dictionary containing:
                - text: Text token IDs (B, L)
                - text_lens: Lengths (B,)
                - context_text_tokens: Context text tokens (B, L')
                - context_text_tokens_lens: Lengths (B,)
                - context_audio_codes: Context audio codes (B, C, T) OR
                - context_audio / context_audio_lens: Raw context audio to encode
                - phoneme_tokens (optional): GT phoneme tokens (B, L'')
                - phoneme_tokens_lens (optional): Lengths (B,)
            max_decoder_steps: Maximum number of decoder steps.
            temperature: Sampling temperature for audio codes.
            topk: Top-k sampling parameter.
            use_cfg: Whether to use classifier-free guidance.
            cfg_scale: CFG scale factor.
            use_local_transformer_for_inference: Whether to use local transformer.
            phoneme_input_type: 'gt' or 'pred' for phoneme tokens.
            phoneme_sampling_method: 'argmax' or 'sample' for phoneme token selection.
            force_dropout_text: Whether to dropout text embeddings.

        Returns:
            InferBatchOutput containing predicted audio, codes, and RTF metrics.
        """
        with torch.inference_mode():
            start_time = time.time()

            # Extract tensors from batch
            text = batch['text']
            text_lens = batch['text_lens']
            context_text_tokens = batch['context_text_tokens']
            context_text_tokens_lens = batch['context_text_tokens_lens']

            # Handle context audio - either use codes directly or encode from audio
            if 'context_audio_codes' in batch:
                context_audio_codes = batch['context_audio_codes']
                context_audio_codes_lens = batch['context_audio_codes_lens']
            else:
                context_audio = batch['context_audio']
                context_audio_lens = batch['context_audio_lens']
                context_audio_codes, context_audio_codes_lens = self.audio_to_codes(context_audio, context_audio_lens)

            # Optional GT phoneme tokens for teacher forcing
            gt_phoneme_tokens = batch.get('phoneme_tokens')
            gt_phoneme_tokens_lens = batch.get('phoneme_tokens_lens')

            batch_size = text.size(0)

            # Initialize streaming state
            state = self.streaming_init(
                context_audio_codes=context_audio_codes,
                context_audio_codes_lens=context_audio_codes_lens,
                context_text_tokens=context_text_tokens,
                context_text_tokens_lens=context_text_tokens_lens,
                use_cfg=use_cfg,
                cfg_scale=cfg_scale,
                use_local_transformer=use_local_transformer_for_inference,
                temperature=temperature,
                topk=topk,
                phoneme_input_type=phoneme_input_type,
                phoneme_sampling_method=phoneme_sampling_method,
                gt_phoneme_tokens=gt_phoneme_tokens,
                gt_phoneme_tokens_lens=gt_phoneme_tokens_lens,
            )

            time_to_first_prediction = None
            generation_start_time = time.time()
            device = text.device

            # Generate until all items are finished or max steps reached
            while not state.finished.all() and len(state.all_predictions) < max_decoder_steps:
                # Gather the correct text token for each batch item based on text_tokens_seen
                # Items in context phase will have their token ignored by streaming_step
                positions = state.text_tokens_seen.clamp(max=text.size(1) - 1)
                current_tokens = text[torch.arange(batch_size, device=device), positions]

                # For items that have exhausted their text, provide EOS token
                text_exhausted = state.text_tokens_seen >= text_lens
                current_tokens = torch.where(
                    text_exhausted, torch.full_like(current_tokens, self.eos_id), current_tokens
                )

                state, audio_codes, phoneme_tokens = self.streaming_step(
                    state=state,
                    text_tokens=current_tokens,
                    force_dropout_text=force_dropout_text,
                )

                # Record time to first audio prediction
                if time_to_first_prediction is None and audio_codes is not None:
                    time_to_first_prediction = time.time() - start_time

            tts_generation_time = time.time() - generation_start_time

            # Finalize and decode audio
            finalize_output = self.streaming_finalize(state)

            end_time = time.time()
            total_time = end_time - start_time

            # Compute RTF metrics
            total_audio_samples = finalize_output.audio_len.sum().item()
            total_audio_duration = total_audio_samples / self.output_sample_rate
            num_frames = len(state.all_predictions)
            tts_generation_time_per_frame = tts_generation_time / num_frames if num_frames > 0 else 0.0

            rtf_metrics = {
                'rtf': total_audio_duration / total_time if total_time > 0 else 0.0,
                'time_to_first_prediction': time_to_first_prediction,
                'tts_generation_time': tts_generation_time,
                'max_frames_generated': num_frames,
                'tts_generation_time_per_frame': tts_generation_time_per_frame,
                'batch_size': batch_size,
            }

            return InferBatchOutput(
                predicted_audio=finalize_output.audio,
                predicted_audio_lens=finalize_output.audio_len,
                predicted_codes=finalize_output.audio_codes,
                predicted_codes_lens=finalize_output.audio_codes_len,
                rtf_metrics=rtf_metrics,
            )

    @classmethod
    def list_available_models(cls) -> List[PretrainedModelInfo]:
        return []
