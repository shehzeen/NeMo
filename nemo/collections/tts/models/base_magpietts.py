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

from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import get_worker_info

from nemo.collections.tts.data.text_to_speech_dataset_lhotse import instantiate_phoneme_tokenizer, setup_tokenizers
from nemo.collections.tts.modules.magpietts_modules import SpecialAudioToken, cosine_schedule
from nemo.collections.tts.parts.utils.helpers import get_mask_from_lengths
from nemo.core.classes import ModelPT
from nemo.utils import logging


def worker_init_fn(worker_id):
    """Per-worker init for DataLoader workers.

    Sets up tokenizers for the dataset (text and optionally phoneme)
    when using multiprocessing.
    """
    logging.info(f"Worker {worker_id} initializing...")
    worker_info = get_worker_info()
    dataset = worker_info.dataset
    tokenizer = setup_tokenizers(dataset.tokenizer_config, mode=dataset.dataset_type)
    dataset.text_tokenizer = tokenizer
    if hasattr(dataset, 'phoneme_tokenizer_config'):
        dataset.phoneme_tokenizer = instantiate_phoneme_tokenizer(dataset.phoneme_tokenizer_config)


class BaseMagpieTTSModel(ModelPT):
    """Base class for MagpieTTS models.

    Contains shared functionality for audio codec helpers, special token
    manipulation, local transformer functions, and state dict handling.
    Subclasses (EasyMagpieTTSModel, MagpieTTSModel) provide their own
    ``__init__``, data loading, training/inference logic, etc.
    """

    # ------------------------------------------------------------------
    # State-dict exclusion – subclasses override
    # ------------------------------------------------------------------

    def _get_state_dict_keys_to_exclude(self) -> List[str]:
        """Return list of key substrings to exclude from checkpoint save/load.

        Subclasses should override to specify model-specific exclusions
        (e.g. codec model, eval models).
        """
        return ['_codec_model']

    # ------------------------------------------------------------------
    # state_dict / load_state_dict / optimizer param groups
    # ------------------------------------------------------------------

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        if hasattr(self, '_no_state_dict') and self._no_state_dict:
            return {}
        state_dict = super().state_dict(destination, prefix, keep_vars)
        keys_substrings_to_exclude = self._get_state_dict_keys_to_exclude()
        for key in list(state_dict.keys()):
            if any(substring in key for substring in keys_substrings_to_exclude):
                del state_dict[key]
        return state_dict

    def load_state_dict(self, state_dict, strict=True):
        if not strict:
            super().load_state_dict(state_dict, strict=False)
        modules_to_skip = self._get_state_dict_keys_to_exclude()
        for name, child in self.named_children():
            if name in modules_to_skip:
                continue
            if any(param.numel() > 0 for param in child.parameters()):
                new_state_dict = {}
                for key in state_dict.keys():
                    name_with_dot = f"{name}."
                    if key.startswith(name_with_dot):
                        new_state_dict[key[len(name_with_dot) :]] = state_dict[key]
                child.load_state_dict(new_state_dict)

    def setup_optimizer_param_groups(self):
        """Exclude frozen eval/inference-only models from the optimizer."""
        modules_to_exclude = set(self._get_state_dict_keys_to_exclude())

        excluded_param_ids = set()
        for name, module in self.named_children():
            if name in modules_to_exclude:
                for param in module.parameters():
                    excluded_param_ids.add(id(param))

        trainable_params = [p for p in self.parameters() if id(p) not in excluded_param_ids]

        logging.info(
            f"setup_optimizer_param_groups: {len(trainable_params)} params in optimizer, "
            f"{len(excluded_param_ids)} params excluded (eval models)"
        )

        self._optimizer_param_groups = [{"params": trainable_params}]

    # ------------------------------------------------------------------
    # Special token helpers
    # ------------------------------------------------------------------

    def add_eos_token(self, codes, codes_len, eos_id, num_eos_tokens=1):
        # codes: (B, C, T')
        codes = torch.nn.functional.pad(input=codes, pad=(0, num_eos_tokens), value=0)
        codes_len = codes_len + num_eos_tokens
        for idx in range(codes.size(0)):
            codes[idx, :, codes_len[idx] - 1] = eos_id
        return codes, codes_len

    def add_special_tokens(self, codes, codes_len, bos_id, eos_id, num_bos_tokens=1, num_eos_tokens=1):
        # codes: (B, C, T')
        codes = torch.nn.functional.pad(input=codes, pad=(num_bos_tokens, 0), value=bos_id)
        codes_len = codes_len + num_bos_tokens
        codes, codes_len = self.add_eos_token(
            codes=codes, codes_len=codes_len, eos_id=eos_id, num_eos_tokens=num_eos_tokens
        )
        return codes, codes_len

    def remove_bos_token(self, codes, codes_len, num_tokens=1):
        codes = codes[:, :, num_tokens:]
        codes_len = codes_len - num_tokens
        return codes, codes_len

    def remove_embedded_bos_token(self, embedded, embedded_len):
        embedded = embedded[:, 1:, :]
        embedded_len = embedded_len - 1
        return embedded, embedded_len

    def remove_eos_token(self, codes, codes_len):
        codes_len = codes_len - 1
        codes = codes[:, :, :-1]
        mask = get_mask_from_lengths(lengths=codes_len)
        codes = codes * mask.unsqueeze(1)
        return codes, codes_len

    def remove_embedded_eos_token(self, embedded, embedded_len):
        # embedded: (B, T', D)
        embedded_len = embedded_len - 1
        embedded = embedded[:, :-1, :]
        mask = get_mask_from_lengths(lengths=embedded_len)
        embedded = embedded * mask.unsqueeze(2)
        return embedded, embedded_len

    def remove_special_tokens(self, codes, codes_len, num_bos_tokens=1):
        codes, codes_len = self.remove_bos_token(codes=codes, codes_len=codes_len, num_tokens=num_bos_tokens)
        codes, codes_len = self.remove_eos_token(codes=codes, codes_len=codes_len)
        return codes, codes_len

    # ------------------------------------------------------------------
    # Audio codec helpers
    # ------------------------------------------------------------------

    def audio_to_codes(self, audio, audio_len, sample_rate=None):
        self._codec_model.eval()
        with torch.no_grad(), torch.autocast(device_type=audio.device.type, dtype=torch.float32):
            codes, codes_len = self._codec_model.encode(audio=audio, audio_len=audio_len, sample_rate=sample_rate)
            return codes, codes_len

    def codes_to_audio(self, codes, codes_len):
        # codes: (B, C, T')
        self._codec_model.eval()
        with torch.no_grad(), torch.autocast(device_type=codes.device.type, dtype=torch.float32):
            if self._codec_converter is not None:
                codes = self._codec_converter.convert_new_to_original(audio_tokens=codes, audio_lens=codes_len)
            audio, audio_len = self._codec_model.decode(tokens=codes, tokens_len=codes_len)
            return audio, audio_len, codes

    # ------------------------------------------------------------------
    # Padding / forbidden-logits helpers
    # ------------------------------------------------------------------

    def pad_audio_codes(self, audio_codes: torch.Tensor):
        """Pads the time dimension of the audio codes to a multiple of the frame stacking factor.

        Args:
            audio_codes: (B, C, T)
        Returns:
            (B, C, T_padded)
        """
        T = audio_codes.size(2)
        T_padded = int(np.ceil(T / self.frame_stacking_factor) * self.frame_stacking_factor)
        num_pad = T_padded - T
        audio_codes = torch.nn.functional.pad(input=audio_codes, pad=(0, num_pad))
        return audio_codes

    def clear_forbidden_logits(self, logits: torch.Tensor, forbid_audio_eos: bool = False) -> torch.Tensor:
        """Sets logits of forbidden tokens to ``-inf`` so they will never be sampled.

        Specifically, we forbid sampling of all special tokens except AUDIO_EOS
        which is allowed by default.

        Args:
            logits: (B, C, num_audio_tokens_per_codebook)
            forbid_audio_eos: If True, also forbid AUDIO_EOS tokens from being sampled.
        """
        logits[
            :,
            :,
            SpecialAudioToken.get_forbidden_tokens(self.codebook_size, forbid_audio_eos=forbid_audio_eos),
        ] = float('-inf')
        return logits

    # ------------------------------------------------------------------
    # MaskGit helpers
    # ------------------------------------------------------------------

    def maskgit_create_random_mask(self, codes):
        """Creates a mask where True indicates positions that should be replaced with MASK_TOKEN."""
        B, C, T = codes.shape
        rand_values = torch.rand(B, T, device=codes.device)
        frac_masked = cosine_schedule(rand_values)
        n_masked = torch.ceil(frac_masked * C).long()
        random_permutations = torch.argsort(torch.rand(B, C, T, device=codes.device), dim=1)
        mask_indices = torch.arange(C, device=codes.device).view(1, C, 1)
        mask = mask_indices < n_masked.view(B, 1, T)
        mask = torch.gather(mask, 1, random_permutations)
        return mask

    def maskgit_apply_random_mask(self, codes):
        """Randomly replaces some codes with MASK_TOKEN following the cosine schedule."""
        mask = self.maskgit_create_random_mask(codes)
        codes_with_mask = torch.where(mask, self.mask_token_id, codes)
        return codes_with_mask, mask

    # ------------------------------------------------------------------
    # Local transformer – training
    # ------------------------------------------------------------------

    def compute_local_transformer_logits(self, dec_out, audio_codes_target, targets_offset_by_one=False):
        """Predicts the logits for all codebooks using the local transformer.

        Used in both autoregressive (AR) and MaskGit (MG) modes during
        training and validation (not inference/sampling).

        The sequence layout is slightly different between AR and MG modes, as shown below
        (using an 8-codebook setup as an example)::

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

        Args:
            dec_out: (B, T', E)
            audio_codes_target: (B, C, T')
            targets_offset_by_one: if False, target for index 0 is codebook 0 (AR);
                if True, target for index 1 is codebook 0 (MaskGit).
        """
        C = self.num_audio_codebooks
        dec_out_all = dec_out.reshape(-1, dec_out.size(-1))  # (B*T', E)
        local_transformer_input = [dec_out_all]
        audio_codes_target = self.pad_audio_codes(audio_codes_target).long()
        for fs_index in range(self.frame_stacking_factor):
            for codebook_num in range(C):
                codes = audio_codes_target[:, codebook_num, fs_index :: self.frame_stacking_factor]
                codes = codes.reshape(-1)
                codebook_embedding = self.audio_embeddings[codebook_num + fs_index * C](codes)
                codebook_embedding = self.audio_in_projection(codebook_embedding)
                local_transformer_input.append(codebook_embedding)

        local_transformer_input = torch.stack(local_transformer_input, dim=1)
        local_transformer_input = self.local_transformer_in_projection(local_transformer_input)
        _mask = torch.ones(
            local_transformer_input.size(0), local_transformer_input.size(1), device=local_transformer_input.device
        )
        local_transformer_output = self.local_transformer(local_transformer_input, _mask)['output']
        if not targets_offset_by_one:
            local_transformer_output = local_transformer_output[:, :-1, :]
        else:
            local_transformer_output = local_transformer_output[:, 1:, :]

        local_transformer_output = self.local_transformer_audio_out_projection(local_transformer_output)

        all_code_logits = []
        for fs_index in range(self.frame_stacking_factor):
            for codebook_num in range(audio_codes_target.size(1)):
                codebook_logits = self.local_transformer_out_projections[codebook_num + fs_index * C](
                    local_transformer_output[:, codebook_num + fs_index * C, :]
                )
                all_code_logits.append(codebook_logits)
        all_code_logits = torch.cat(all_code_logits, dim=1)

        all_code_logits = all_code_logits.view(
            audio_codes_target.size(0), audio_codes_target.size(2) // self.frame_stacking_factor, -1
        )

        return all_code_logits

    # ------------------------------------------------------------------
    # Local transformer – AR sampling
    # ------------------------------------------------------------------

    def local_transformer_sample_autoregressive(
        self,
        dec_output: torch.Tensor,
        temperature: float = 0.7,
        topk: int = 80,
        unfinished_items: Dict[int, bool] = {},
        finished_items: Dict[int, bool] = {},
        use_cfg: bool = False,
        cfg_scale: float = 1.0,
        use_kv_cache: bool = True,
        forbid_audio_eos: bool = False,
    ) -> torch.Tensor:
        """Sample audio codes autoregressively across codebooks using the local transformer.

        Uses multinomial sampling with temperature, top-k, and
        classifier-free guidance (CFG).

        Args:
            dec_output: Decoder output tensor (B, E).
            temperature: Sampling temperature. When <= 0, uses argmax.
            topk: Number of top-probability tokens to consider.
            unfinished_items: Batch indices that have not completed generation (EOS forbidden).
            finished_items: Batch indices that are completed (EOS forced).
            use_cfg: Whether to use classifier-free guidance (doubled batch).
            cfg_scale: Scale factor for CFG.
            use_kv_cache: Whether to use key-value caching in the local transformer.
            forbid_audio_eos: Whether to globally forbid audio EOS.

        Returns:
            Sampled audio codes (B, num_codebooks, frame_stacking_factor).
        """
        self.local_transformer.reset_cache(use_cache=use_kv_cache)
        dec_output = dec_output.unsqueeze(1)  # (B, 1, E)
        local_transformer_input = self.local_transformer_in_projection(dec_output)
        all_preds = []
        for codebook_num in range(self.num_audio_codebooks * self.frame_stacking_factor):
            _mask = torch.ones(
                local_transformer_input.size(0), local_transformer_input.size(1), device=local_transformer_input.device
            )
            local_transformer_output = self.local_transformer(local_transformer_input, _mask)['output']

            lt_out_for_proj = self.local_transformer_audio_out_projection(local_transformer_output[:, -1, :])
            codebook_logits = self.local_transformer_out_projections[codebook_num](lt_out_for_proj)

            if use_cfg:
                actual_batch_size = codebook_logits.size(0) // 2
                conditional_logits = codebook_logits[:actual_batch_size]
                unconditional_logits = codebook_logits[actual_batch_size:]
                cfg_logits = cfg_scale * conditional_logits + (1.0 - cfg_scale) * unconditional_logits
                codebook_logits[:actual_batch_size] = cfg_logits

            codebook_logits = torch.nan_to_num(codebook_logits, nan=0.0, posinf=100.0, neginf=-100.0)
            codebook_logits = codebook_logits.clamp(min=-100.0, max=100.0)

            for item_idx in unfinished_items:
                codebook_logits[item_idx, self.audio_eos_id] = float('-inf')
            for item_idx in finished_items:
                codebook_logits[item_idx, :] = float('-inf')
                codebook_logits[item_idx, self.audio_eos_id] = 0.0

            codebook_logits = self.clear_forbidden_logits(
                codebook_logits.unsqueeze(1), forbid_audio_eos=forbid_audio_eos
            ).squeeze(1)

            codebook_logits_topk = torch.topk(codebook_logits, topk, dim=-1)[0]
            indices_to_remove = codebook_logits < codebook_logits_topk[:, -1].unsqueeze(-1)
            codebook_logits_rescored = codebook_logits.clone()
            codebook_logits_rescored[indices_to_remove] = float('-inf')

            if temperature <= 0.0:
                codebook_preds = codebook_logits_rescored.argmax(dim=-1, keepdim=True)
            else:
                codebook_probs = torch.softmax(codebook_logits_rescored / temperature, dim=-1)
                codebook_preds = torch.multinomial(codebook_probs, 1)

            if use_cfg:
                codebook_preds[actual_batch_size:] = codebook_preds[:actual_batch_size]
            all_preds.append(codebook_preds)

            next_local_transformer_input = self.audio_embeddings[codebook_num](codebook_preds.squeeze(-1)).unsqueeze(1)
            next_local_transformer_input = self.audio_in_projection(next_local_transformer_input)
            next_local_transformer_input = self.local_transformer_in_projection(next_local_transformer_input)
            local_transformer_input = torch.cat([local_transformer_input, next_local_transformer_input], dim=1)

        all_preds = torch.cat(all_preds, dim=1)  # (B, num_codebooks * frame_stacking_factor)
        all_preds = all_preds.reshape(-1, self.frame_stacking_factor, self.num_audio_codebooks).permute(0, 2, 1)
        if use_cfg:
            all_preds = all_preds[:actual_batch_size]

        return all_preds

    # ------------------------------------------------------------------
    # Local transformer – MaskGit sampling
    # ------------------------------------------------------------------

    def local_transformer_sample_maskgit(
        self,
        dec_output: torch.Tensor,
        temperature: float = 0.7,
        topk: int = 80,
        unfinished_items: Dict[int, bool] = {},
        finished_items: Dict[int, bool] = {},
        use_cfg: bool = False,
        cfg_scale: float = 1.0,
        n_steps: int = 3,
        noise_scale: float = 0.0,
        fixed_schedule: Optional[List[int]] = None,
        dynamic_cfg_scale: bool = False,
        sampling_type: Optional[str] = None,
        forbid_audio_eos: bool = False,
    ) -> torch.Tensor:
        """Sample audio codes using MaskGit-like iterative prediction with the local transformer.

        If frame-stacking is enabled, the codes for all frames in the stack
        are sampled, treated as one long sequence.

        Args:
            dec_output: Decoder output tensor (B, E).
            temperature: Sampling temperature.
            topk: Number of top-probability tokens to consider.
            unfinished_items: Batch indices that have not completed generation.
            finished_items: Batch indices that are completed.
            use_cfg: Whether to use classifier-free guidance.
            cfg_scale: Scale factor for CFG.
            n_steps: Number of iterative refinement steps.
            noise_scale: Scale factor for noise added to confidence scores.
            fixed_schedule: Fixed schedule for number of tokens to unmask per step.
            dynamic_cfg_scale: Whether to dynamically adjust CFG scale.
            sampling_type: Sampling strategy (``"default"``, ``"causal"``,
                ``"purity_causal"``, ``"purity_default"``).
            forbid_audio_eos: Whether to globally forbid audio EOS.

        Returns:
            Sampled audio codes (B, num_codebooks, frame_stacking_factor).
        """
        device = dec_output.device
        self.local_transformer.reset_cache(use_cache=False)
        dec_output = dec_output.unsqueeze(1)
        local_transformer_input_init = self.local_transformer_in_projection(dec_output)
        codebook_seq_len = self.num_audio_codebooks * self.frame_stacking_factor
        B = dec_output.size(0)

        min_confidence = 0
        max_confidence = 5
        confidences = min_confidence * torch.ones(B, codebook_seq_len, device=device)
        codes = self.mask_token_id * torch.ones((B, codebook_seq_len), device=device, dtype=torch.long)
        sampled_codes = codes.clone()
        if fixed_schedule is not None:
            n_steps = len(fixed_schedule)
        for step in range(n_steps):
            progress = step / n_steps
            frac_masked = cosine_schedule(torch.tensor(progress))
            if sampling_type == "causal" or sampling_type == "purity_causal":
                frac_masked = torch.ones_like(frac_masked) * (1.0 - progress)
            if fixed_schedule is None:
                n_masked = torch.ceil(codebook_seq_len * frac_masked).long()
            else:
                n_masked = codebook_seq_len - fixed_schedule[step]
            n_unmasked = codebook_seq_len - n_masked

            if sampling_type == "causal" or sampling_type == "purity_causal":
                n_frames_to_allow = int(np.floor(progress * self.frame_stacking_factor + 1))
                confidences[:, n_frames_to_allow * self.num_audio_codebooks :] = min_confidence - 1

            _, topk_indices = torch.topk(confidences, k=n_unmasked, dim=1)
            if use_cfg:
                actual_batch_size = topk_indices.size(0) // 2
                assert (
                    topk_indices[actual_batch_size:] == topk_indices[:actual_batch_size]
                ).all(), "Topk indices are not the same for conditional and unconditional codes"

            unmasked_codes = torch.gather(sampled_codes, dim=1, index=topk_indices)
            codes.scatter_(dim=1, index=topk_indices, src=unmasked_codes)

            local_transformer_input = local_transformer_input_init
            for codebook_num in range(codebook_seq_len):
                next_local_transformer_input = self.audio_embeddings[codebook_num](codes[:, codebook_num]).unsqueeze(1)
                next_local_transformer_input = self.local_transformer_in_projection(next_local_transformer_input)
                local_transformer_input = torch.cat([local_transformer_input, next_local_transformer_input], dim=1)

            _mask = torch.ones(B, codebook_seq_len + 1, device=device)
            local_transformer_output = self.local_transformer(local_transformer_input, _mask)['output']

            logits = []
            for codebook_num in range(codebook_seq_len):
                codebook_logits = self.local_transformer_out_projections[codebook_num](
                    local_transformer_output[:, codebook_num + 1, :]
                )
                logits.append(codebook_logits)
            logits = torch.stack(logits, dim=1)

            if use_cfg:
                actual_batch_size = logits.size(0) // 2
                conditional_logits = logits[:actual_batch_size]
                unconditional_logits = logits[actual_batch_size:]
                if not dynamic_cfg_scale:
                    current_cfg_scale = cfg_scale
                else:
                    progress = step / (n_steps - 1)
                    interp = progress
                    current_cfg_scale = (cfg_scale - 1) * interp + 1.0
                cfg_logits = current_cfg_scale * conditional_logits + (1.0 - current_cfg_scale) * unconditional_logits
                logits[:actual_batch_size] = cfg_logits

            logits = self.clear_forbidden_logits(logits, forbid_audio_eos=forbid_audio_eos)

            for item_idx in unfinished_items:
                logits[item_idx, self.audio_eos_id] = float('-inf')
            for item_idx in finished_items:
                logits[item_idx, :, :] = float('-inf')
                logits[item_idx, :, self.audio_eos_id] = 0.0

            logits_topk = torch.topk(logits, topk, dim=-1)[0]
            indices_to_remove = logits < logits_topk[:, :, -1].unsqueeze(-1)
            logits_rescored = logits.clone()
            logits_rescored[indices_to_remove] = float('-inf')
            probs = torch.softmax(logits_rescored / temperature, dim=-1)
            sampled_codes = torch.multinomial(probs.view(B * codebook_seq_len, -1), 1).view(B, codebook_seq_len)
            if use_cfg:
                sampled_codes[actual_batch_size:] = sampled_codes[:actual_batch_size]
                probs[actual_batch_size:] = probs[:actual_batch_size]
            if sampling_type != "purity_causal" and sampling_type != "purity_default":
                confidences = torch.gather(probs, dim=2, index=sampled_codes.unsqueeze(-1)).squeeze(-1)
            else:
                confidences = probs.max(dim=2)[0]
            sampled_codes.scatter_(dim=1, index=topk_indices, src=unmasked_codes)
            if noise_scale > 0.0:
                noise = (torch.rand_like(confidences) - 0.5) * noise_scale * (1 - (step + 2) / n_steps)
                confidences += noise
                confidences[actual_batch_size:] = confidences[:actual_batch_size]
            confidence_eps = 0.1
            assert (
                confidences.max() + confidence_eps < max_confidence
            ), f"Predicted confidence is approaching max_confidence: {confidences.max()}"
            confidences.scatter_(
                index=topk_indices, dim=1, src=max_confidence * torch.ones_like(topk_indices, dtype=torch.float)
            )
        codes = sampled_codes
        assert not (
            codes == self.mask_token_id
        ).any(), "Codes contain mask tokens after completion of MaskGit sampling"

        codes = codes.reshape(B, self.frame_stacking_factor, self.num_audio_codebooks).permute(0, 2, 1)

        if use_cfg:
            codes = codes[:actual_batch_size]
        return codes
