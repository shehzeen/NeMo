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

import copy
import os
import random
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf
import torch
from lightning.pytorch import Trainer
from omegaconf import DictConfig, open_dict

import nemo.collections.asr as nemo_asr
from nemo.collections.asr.metrics.wer import word_error_rate
from nemo.collections.asr.parts.mixins.transcription import TranscribeConfig
from nemo.collections.tts.models.easy_magpietts import EasyMagpieTTSModel
from nemo.collections.tts.parts.utils.helpers import (
    get_mask_from_lengths,
    get_speaker_embeddings_from_filepaths,
    process_text_for_cer,
    transcribe_with_whisper,
)
from nemo.utils import logging

try:
    import torchaudio
    from torchaudio.pipelines import SQUIM_OBJECTIVE

    HAVE_TORCHAUDIO = True
except ImportError:
    HAVE_TORCHAUDIO = False

try:
    from nemo_text_processing.text_normalization.normalize import Normalizer

    PYNINI_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    Normalizer = None
    PYNINI_AVAILABLE = False


class EasyMagpieTTSModelOnlinePO(EasyMagpieTTSModel):
    """
    EasyMagpie-TTS online preference optimization model (GRPO / DR-GRPO).

    Training flow:
    1. Sample multiple generations per prompt.
    2. Compute rewards (CER/SSIM/PESQ).
    3. Compute group-normalized advantages.
    4. Run teacher-forced policy forward on generated codes and optimize GRPO objective.
    5. Add auxiliary phoneme loss from the same forward pass with GT phoneme tokens.
    """

    def __init__(self, cfg: DictConfig, trainer: 'Trainer' = None):
        super().__init__(cfg, trainer)
        
        self.run_val_inference = True # Always run validation inference in PO.
        self.automatic_optimization = False

        ref_model_cfg = copy.deepcopy(cfg)
        with open_dict(ref_model_cfg):
            ref_model_cfg.train_ds = None
            ref_model_cfg.validation_ds = None

        self.reference_free = self.cfg.get('reference_free', False)
        if not self.reference_free:
            self._reference_model = EasyMagpieTTSModel(cfg=ref_model_cfg)
            logging.info("Loading EasyMagpie reference model from checkpoint")
            self._reference_model.load_state_dict(
                torch.load(cfg.reference_model_ckpt_path, map_location="cpu", weights_only=False)['state_dict']
            )
            self._reference_model.freeze()
            self._reference_model._no_state_dict = True
            logging.info("Reference model loaded and frozen")

        reward_asr_model = cfg.get('reward_asr_model', 'nemo')
        if reward_asr_model == 'nemo':
            self._eval_asr_model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained(
                model_name=cfg.get('reward_asr_model_name', "nvidia/parakeet-ctc-0.6b")
            )
            self._eval_asr_model.freeze()
            self.whisper_processor = None
            self.whisper_model = None
        elif reward_asr_model == 'whisper':
            from transformers import WhisperForConditionalGeneration, WhisperProcessor

            self._eval_asr_model = None
            self.whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3")
            self.whisper_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3")
            self.whisper_model.eval()
            for param in self.whisper_model.parameters():
                param.requires_grad = False
            self.use_multilingual_asr = True
        else:
            raise ValueError(f"Unknown reward_asr_model: {reward_asr_model}")

        self._eval_speaker_verification_model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(
            model_name=cfg.get('speaker_verification_model_name', 'titanet_large')
        )
        self._eval_speaker_verification_model.freeze()

        use_pesq = self.cfg.get('use_pesq', False)
        if use_pesq:
            assert HAVE_TORCHAUDIO, "torchaudio is required for PESQ reward."
            self.squim_objective_model = SQUIM_OBJECTIVE.get_model()

        self.loss_type = self.cfg.get('loss_type', 'grpo')
        if self.loss_type not in ['grpo', 'dr_grpo']:
            raise ValueError(
                f"Received loss_type={self.loss_type}. Supported values: ['grpo', 'dr_grpo']."
            )
        self.scale_rewards = self.cfg.get('scale_rewards', True)
        self.max_decoder_steps = self.cfg.get('max_decoder_steps', 220)
        self.aux_phoneme_loss_weight = self.cfg.get('aux_phoneme_loss_weight', 1.0)
        self.po_groups_per_subbatch = max(int(self.cfg.get('po_groups_per_subbatch', 1)), 1)

        self._normalize_whisper_transcript = self.cfg.get('normalize_whisper_transcript', True)
        if reward_asr_model == 'whisper' and self._normalize_whisper_transcript:
            self._normalizer_cache = {}

        # Filter out poor groups for stable optimization.
        self.best_cer_threshold = self.cfg.get('best_cer_threshold', 1.0)
        self.worst_cer_threshold = self.cfg.get('worst_cer_threshold', 1.0)

    def _get_trainable_module_groups(self) -> Dict[str, List[torch.nn.Parameter]]:
        """Return a dict mapping module-group name → list of trainable parameters.
        Used for per-module gradient / weight diagnostics."""
        modules_to_exclude = {
            '_speaker_verification_model', '_codec_model', '_eval_asr_model',
            '_eval_speaker_verification_model', '_reference_model',
            'whisper_model', 'whisper_processor', 'squim_objective_model',
        }
        groups: Dict[str, List[torch.nn.Parameter]] = {}
        for name, module in self.named_children():
            if name in modules_to_exclude:
                continue
            params = [p for p in module.parameters() if p.requires_grad]
            if params:
                groups[name] = params
        return groups

    @torch.no_grad()
    def _compute_grad_and_weight_metrics(self) -> Dict[str, float]:
        """Compute per-module and global gradient / weight statistics."""
        module_groups = self._get_trainable_module_groups()
        metrics: Dict[str, float] = {}

        all_grad_norms = []
        all_weight_norms = []
        total_params = 0
        zero_grad_params = 0
        nan_grad_params = 0
        none_grad_params = 0  # params where p.grad is None

        for group_name, params in module_groups.items():
            grad_norms = []
            weight_norms = []
            n_params_total = len(params)
            n_params_with_grad = 0
            n_params_none_grad = 0
            for p in params:
                w_norm = p.data.norm(2).item()
                weight_norms.append(w_norm)
                if p.grad is not None:
                    g_norm = p.grad.data.norm(2).item()
                    grad_norms.append(g_norm)
                    total_params += 1
                    n_params_with_grad += 1
                    if g_norm == 0.0:
                        zero_grad_params += 1
                    if not np.isfinite(g_norm):
                        nan_grad_params += 1
                else:
                    none_grad_params += 1
                    n_params_none_grad += 1

            # Always record weight norms for every module
            if weight_norms:
                module_weight_norm = float(np.sqrt(sum(w**2 for w in weight_norms)))
                metrics[f'weight_norm/{group_name}'] = module_weight_norm
                all_weight_norms.extend(weight_norms)

            # Record grad norms (may be empty if module gets no gradients)
            if grad_norms:
                module_grad_norm = float(np.sqrt(sum(g**2 for g in grad_norms)))
                metrics[f'grad_norm/{group_name}'] = module_grad_norm
                metrics[f'grad_norm_mean/{group_name}'] = float(np.mean(grad_norms))
                metrics[f'grad_norm_max/{group_name}'] = float(np.max(grad_norms))
                all_grad_norms.extend(grad_norms)
            else:
                # Explicitly record zero grad norm for modules with no gradients
                metrics[f'grad_norm/{group_name}'] = 0.0
                metrics[f'grad_norm_mean/{group_name}'] = 0.0
                metrics[f'grad_norm_max/{group_name}'] = 0.0

            # Per-module param counts
            metrics[f'grad_diagnostics/params_with_grad/{group_name}'] = float(n_params_with_grad)
            metrics[f'grad_diagnostics/params_none_grad/{group_name}'] = float(n_params_none_grad)
            metrics[f'grad_diagnostics/params_total/{group_name}'] = float(n_params_total)

        # Global aggregates
        if all_grad_norms:
            metrics['grad_norm/global'] = float(np.sqrt(sum(g**2 for g in all_grad_norms)))
            metrics['grad_norm_mean/global'] = float(np.mean(all_grad_norms))
            metrics['grad_norm_max/global'] = float(np.max(all_grad_norms))
        if all_weight_norms:
            metrics['weight_norm/global'] = float(np.sqrt(sum(w**2 for w in all_weight_norms)))
        metrics['grad_diagnostics/total_params_with_grad'] = float(total_params)
        metrics['grad_diagnostics/zero_grad_params'] = float(zero_grad_params)
        metrics['grad_diagnostics/nan_grad_params'] = float(nan_grad_params)
        metrics['grad_diagnostics/none_grad_params'] = float(none_grad_params)

        return metrics

    @torch.no_grad()
    def _compute_weight_update_metrics(
        self, prev_weights: Dict[int, torch.Tensor]
    ) -> Dict[str, float]:
        """Compute per-module weight delta norms (how much weights actually changed)."""
        metrics: Dict[str, float] = {}
        module_groups = self._get_trainable_module_groups()
        all_deltas = []
        for group_name, params in module_groups.items():
            deltas = []
            for p in params:
                pid = id(p)
                if pid in prev_weights:
                    delta = (p.data - prev_weights[pid]).norm(2).item()
                    deltas.append(delta)
            if deltas:
                module_delta_norm = float(np.sqrt(sum(d**2 for d in deltas)))
                metrics[f'weight_delta/{group_name}'] = module_delta_norm
                all_deltas.extend(deltas)
        if all_deltas:
            metrics['weight_delta/global'] = float(np.sqrt(sum(d**2 for d in all_deltas)))
        return metrics

    @torch.no_grad()
    def _snapshot_trainable_weights(self) -> Dict[int, torch.Tensor]:
        """Take a snapshot of all trainable parameter values (by param id)."""
        snapshot = {}
        module_groups = self._get_trainable_module_groups()
        for params in module_groups.values():
            for p in params:
                snapshot[id(p)] = p.data.clone()
        return snapshot

    def _print_grad_weight_summary(self, metrics: Dict[str, float], step: int) -> None:
        """Print a concise summary of gradient and weight diagnostics."""
        if not getattr(self.trainer, "is_global_zero", True):
            return

        lines = [f"\n[grad/weight diagnostics] step={step}"]

        # Global summary
        lines.append(
            f"  global grad_norm={metrics.get('grad_norm/global', 0.0):.6f}  "
            f"weight_norm={metrics.get('weight_norm/global', 0.0):.4f}  "
            f"weight_delta={metrics.get('weight_delta/global', 0.0):.8f}"
        )
        lines.append(
            f"  zero_grad_params={int(metrics.get('grad_diagnostics/zero_grad_params', 0))} / "
            f"{int(metrics.get('grad_diagnostics/total_params_with_grad', 0))}  "
            f"nan_grad_params={int(metrics.get('grad_diagnostics/nan_grad_params', 0))}  "
            f"none_grad_params={int(metrics.get('grad_diagnostics/none_grad_params', 0))}"
        )

        # Per-module summary — show ALL modules (keyed by weight_norm which is always recorded)
        module_names = sorted(
            set(
                k.split('/')[1]
                for k in metrics
                if '/' in k and k.startswith('weight_norm/') and k != 'weight_norm/global'
            )
        )
        if module_names:
            lines.append("  per-module:")
            for name in module_names:
                gn = metrics.get(f'grad_norm/{name}', 0.0)
                wn = metrics.get(f'weight_norm/{name}', 0.0)
                wd = metrics.get(f'weight_delta/{name}', 0.0)
                gm = metrics.get(f'grad_norm_max/{name}', 0.0)
                n_with = int(metrics.get(f'grad_diagnostics/params_with_grad/{name}', 0))
                n_none = int(metrics.get(f'grad_diagnostics/params_none_grad/{name}', 0))
                n_total = int(metrics.get(f'grad_diagnostics/params_total/{name}', 0))
                grad_status = "NO_GRAD" if n_with == 0 else f"{n_with}/{n_total}"
                lines.append(
                    f"    {name:40s}  grad_norm={gn:.6f}  grad_max={gm:.6f}  "
                    f"weight_norm={wn:.4f}  weight_delta={wd:.8f}  "
                    f"grad_params={grad_status}"
                )

        summary = "\n".join(lines)
        print(summary)
        logging.info(summary)

    def setup_optimizer_param_groups(self):
        """
        Exclude frozen eval/reference modules AND modules that receive no gradients
        from the PO loss (final_proj, lm_text_head, phoneme_final_proj) from the
        optimizer. Including them would subject their weights to weight decay without
        any learning signal, slowly degrading them.
        """
        modules_to_exclude = {
            '_speaker_verification_model',
            '_codec_model',
            '_eval_asr_model',
            '_eval_speaker_verification_model',
            '_reference_model',
            'whisper_model',
            'whisper_processor',
            # These modules are not used by the PO loss and receive no gradients.
            # Including them would only apply weight decay, degrading their weights.
            'final_proj',
            'lm_text_head',
            'phoneme_final_proj',
        }

        excluded_param_ids = set()
        for name, module in self.named_children():
            if name in modules_to_exclude and hasattr(module, "parameters"):
                for param in module.parameters():
                    excluded_param_ids.add(id(param))

        trainable_params = [p for p in self.parameters() if id(p) not in excluded_param_ids]
        self._optimizer_param_groups = [{"params": trainable_params}]

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        state_dict = super().state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
        keys_substrings_to_exclude = ['_reference_model']
        for key in list(state_dict.keys()):
            if any(substring in key for substring in keys_substrings_to_exclude):
                del state_dict[key]
        return state_dict

    def _get_cached_normalizer(self, lang_key: Optional[str]):
        if not PYNINI_AVAILABLE:
            return None
        lang_key = lang_key if lang_key else "en"
        if lang_key not in self._normalizer_cache:
            logging.info(f"Creating normalizer for language: {lang_key}")
            try:
                self._normalizer_cache[lang_key] = Normalizer(input_case="cased", lang=lang_key)
            except Exception as e:
                logging.warning(f"Failed to create normalizer for language: {lang_key}. Error: {e}")
                self._normalizer_cache[lang_key] = None
        return self._normalizer_cache[lang_key]

    def _get_per_token_logps(self, logits: torch.Tensor, labels: torch.Tensor, loss_mask: torch.Tensor) -> torch.Tensor:
        # Force fp32 for log_softmax to avoid bf16 precision issues that sever the
        # gradient path through the GRPO "exp(logps - logps.detach())" trick.
        # Under bf16 autocast, the tiny gradient signal through this identity-like
        # expression gets rounded to zero, disconnecting local_transformer_out_projections.
        with torch.cuda.amp.autocast(enabled=False):
            logits_fp32 = logits.float()
            per_token_logps = torch.gather(logits_fp32.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)
            per_token_logps = per_token_logps * loss_mask.float()
        return per_token_logps


    def compute_local_transformer_logits(self, dec_out, audio_codes_target, targets_offset_by_one=False):
        """
        Override parent to force fp32 computation for the entire local transformer logits path.

        Under bf16-mixed autocast, the nn.Linear out_projections execute in bf16 and insert
        ToCopyBackward0 nodes in the autograd graph. The GRPO loss formula
        ``exp(logps - logps.detach())`` produces an identity in the forward pass, but the
        gradient signal through this expression is extremely small. The bf16 ToCopyBackward0
        nodes round these tiny gradients to zero, completely severing the gradient path to
        local_transformer_out_projections. Running the full computation in fp32 preserves
        the gradient fidelity.
        """
        with torch.cuda.amp.autocast(enabled=False):
            # Cast dec_out to fp32 if it's in a lower precision (e.g. bf16 from autocast)
            dec_out_fp32 = dec_out.float()
            return super().compute_local_transformer_logits(
                dec_out_fp32, audio_codes_target, targets_offset_by_one=targets_offset_by_one
            )

    def repeat_items_in_batch(self, batch: Dict, num_repeats: int) -> Dict:
        repeated_batch = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                repeated_batch[key] = value.repeat_interleave(num_repeats, dim=0)
            elif isinstance(value, list):
                repeated_value = []
                for item in value:
                    repeated_value.extend([item] * num_repeats)
                repeated_batch[key] = repeated_value
            else:
                repeated_batch[key] = value
        return repeated_batch

    def _get_audio_dir(self) -> str:
        if self.logger is not None and hasattr(self.logger, "log_dir") and self.logger.log_dir is not None:
            log_dir = self.logger.log_dir
        elif self.trainer is not None and self.trainer.log_dir is not None:
            log_dir = self.trainer.log_dir
        else:
            log_dir = "."
        audio_dir = os.path.join(log_dir, 'online_po_audios')
        os.makedirs(audio_dir, exist_ok=True)
        return audio_dir

    def _save_waveforms_to_paths(
        self,
        waveforms: torch.Tensor,
        waveform_lens: torch.Tensor,
        prefix: str,
        sample_rate: int,
    ) -> List[str]:
        audio_dir = self._get_audio_dir()
        time_id = time.time_ns()
        paths = []
        for idx in range(waveforms.size(0)):
            wav = waveforms[idx].float().detach().cpu().numpy()
            wav = wav[: int(waveform_lens[idx].item())]
            # path = os.path.join(audio_dir, f'{prefix}_rank{self.global_rank}_{time_id}_{idx}.wav')
            path = os.path.join(audio_dir, f'{prefix}_rank{self.global_rank}_{idx}.wav')
            sf.write(path, wav, sample_rate)
            paths.append(path)
        return paths

    def _get_reference_audio_paths(self, batch_repeated: Dict) -> List[str]:
        """
        Build per-item reference audio paths for speaker similarity reward.
        Priority: audio_filepaths -> context_audio -> context_audio_codes.
        """
        if 'context_audio' in batch_repeated and 'context_audio_lens' in batch_repeated:
            # TODO: Handle text context here support here.
            return self._save_waveforms_to_paths(
                waveforms=batch_repeated['context_audio'],
                waveform_lens=batch_repeated['context_audio_lens'],
                prefix='reference_context_audio',
                sample_rate=self.sample_rate,
            )

        if 'context_audio_codes' in batch_repeated and 'context_audio_codes_lens' in batch_repeated:
            context_codes = batch_repeated['context_audio_codes'].clone()
            context_lens = batch_repeated['context_audio_codes_lens'].clone()

            target_codes = batch_repeated['audio_codes'].clone()
            target_lens = batch_repeated['audio_codes_lens'].clone()

            # For items where context_lens < 3, fall back to target_codes/target_lens
            # This is for items with text context
            short_context_mask = context_lens < 3
            if short_context_mask.any():
                # Pad the shorter tensor along the time dimension if needed
                max_len = max(context_codes.shape[-1], target_codes.shape[-1])
                if context_codes.shape[-1] < max_len:
                    pad_size = max_len - context_codes.shape[-1]
                    context_codes = torch.nn.functional.pad(context_codes, (0, pad_size), value=0)
                if target_codes.shape[-1] < max_len:
                    pad_size = max_len - target_codes.shape[-1]
                    target_codes = torch.nn.functional.pad(target_codes, (0, pad_size), value=0)
                context_codes[short_context_mask] = target_codes[short_context_mask]
                context_lens[short_context_mask] = target_lens[short_context_mask]
                # Slice to the actual max length needed
                context_codes = context_codes[..., :context_lens.max()]

            if self._codec_converter is not None:
                context_codes = self._codec_converter.convert_original_to_new(
                    audio_tokens=context_codes, audio_lens=context_lens
                ).long()
            context_audio, context_audio_lens, _ = self.codes_to_audio(context_codes, context_lens)
            return self._save_waveforms_to_paths(
                waveforms=context_audio,
                waveform_lens=context_audio_lens,
                prefix='reference_context_codes_decoded',
                sample_rate=self.output_sample_rate,
            )

        raise ValueError(
            "Could not construct reference audio for speaker similarity. Need one of: "
            "context_audio/context_audio_lens, or context_audio_codes/context_audio_codes_lens."
        )

    def _run_easy_process_batch(
        self,
        model: EasyMagpieTTSModel,
        batch: Dict,
        audio_codes: torch.Tensor,
        audio_codes_lens: torch.Tensor,
        mode: str,
    ):
        if 'context_audio_codes' in batch:
            context_audio_codes = batch['context_audio_codes']
            context_audio_codes_lens = batch['context_audio_codes_lens']
        else:
            context_audio_codes, context_audio_codes_lens = model.audio_to_codes(
                batch['context_audio'], batch['context_audio_lens']
            )

        return model.process_batch(
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
            mode=mode,
        )

    def _format_text_table(self, headers: List[str], rows: List[List[str]]) -> str:
        col_widths = [len(h) for h in headers]
        for row in rows:
            for col_idx, value in enumerate(row):
                col_widths[col_idx] = max(col_widths[col_idx], len(value))

        header_line = " | ".join(headers[col_idx].ljust(col_widths[col_idx]) for col_idx in range(len(headers)))
        separator = "-+-".join("-" * col_widths[col_idx] for col_idx in range(len(headers)))
        row_lines = [
            " | ".join(row[col_idx].ljust(col_widths[col_idx]) for col_idx in range(len(headers))) for row in rows
        ]
        return "\n".join([header_line, separator] + row_lines)

    def _print_group_cer_wer_table(
        self,
        batch: Dict,
        batch_metrics: List[Dict],
        group_idx: int,
        group_start_idx: int,
        group_end_idx: int,
        is_group_valid: bool,
        mean_reward: float,
        std_reward: float,
    ) -> None:
        if not getattr(self.trainer, "is_global_zero", True):
            return

        prompt_text = str(batch['raw_texts'][group_idx]).replace("\n", " ")
        if len(prompt_text) > 120:
            prompt_text = f"{prompt_text[:117]}..."

        rows = []
        for local_idx, metric_idx in enumerate(range(group_start_idx, group_end_idx)):
            item_metrics = batch_metrics[metric_idx]
            rows.append(
                [
                    str(local_idx),
                    f"{item_metrics['cer_gt']:.4f}",
                    f"{item_metrics['wer_gt']:.4f}",
                    f"{item_metrics['spk_similarity']:.4f}",
                    f"{item_metrics['reward']:.4f}",
                    f"{item_metrics.get('advantage', 0.0):.4f}",
                ]
            )

        table = self._format_text_table(headers=["item", "cer", "wer", "ssim", "reward", "advantage"], rows=rows)
        print(
            f"[generate_and_reward] group={group_idx} valid={is_group_valid} "
            f"mean_reward={mean_reward:.4f} std_reward={std_reward:.4f}\n"
            f"prompt: {prompt_text}\n{table}\n"
        )

    def generate_and_reward(
        self,
        batch: Dict,
        num_generations_per_item: int,
        mode: str = 'train',
        use_local_transformer_for_inference: bool = False,
    ):
        batch_repeated = self.repeat_items_in_batch(batch, num_generations_per_item)
        reward_asr_model = self.cfg.get('reward_asr_model', 'nemo')
        use_pesq = self.cfg.get('use_pesq', False)

        use_cfg = False
        cfg_scale = 1.0
        inference_cfg_prob = self.cfg.get('inference_cfg_prob', 0.0)
        if (inference_cfg_prob == 1.0) or (inference_cfg_prob > 0.0 and mode == 'train'):
            use_cfg = random.random() < inference_cfg_prob
            cfg_scale = self.cfg.get('inference_cfg_scale', 1.0)

        phoneme_input_type = 'pred'
        gt_phoneme_input_prob = self.cfg.get('gt_phoneme_input_prob', 0.0)
        can_use_gt_phonemes = ('phoneme_tokens' in batch_repeated) and ('phoneme_tokens_lens' in batch_repeated)
        if can_use_gt_phonemes and gt_phoneme_input_prob > 0.0 and mode == 'train':
            phoneme_input_type = 'gt' if random.random() < gt_phoneme_input_prob else 'pred'

        generation_start_time = time.perf_counter()
        print("Inference started")
        output = self.infer_batch(
            batch=batch_repeated,
            max_decoder_steps=self.max_decoder_steps,
            temperature=self.cfg.get('inference_temperature', 0.7),
            topk=self.cfg.get('inference_topk', 80),
            use_cfg=use_cfg,
            cfg_scale=cfg_scale,
            use_local_transformer_for_inference=use_local_transformer_for_inference,
            phoneme_input_type=phoneme_input_type,
            phoneme_sampling_method=self.cfg.get('inference_phoneme_sampling_method', 'argmax'),
            force_dropout_text=False,
            use_teacher_forced=False,
            use_inference_mode=False,
        )
        print("Inference ended")
        audio_generation_time_sec = time.perf_counter() - generation_start_time

        predicted_audio = output.predicted_audio
        predicted_audio_lens = output.predicted_audio_lens
        predicted_codes = output.predicted_codes
        predicted_codes_lens = output.predicted_codes_lens
        save_start_time = time.perf_counter()
        predicted_audio_paths = self._save_waveforms_to_paths(
            waveforms=predicted_audio,
            waveform_lens=predicted_audio_lens,
            prefix='generated',
            sample_rate=self.output_sample_rate,
        )
        audio_save_time_sec = time.perf_counter() - save_start_time
        audio_durations = [int(predicted_audio_lens[idx].item()) / self.output_sample_rate for idx in range(predicted_audio.size(0))]

        rewarding_start_time = time.perf_counter()
        if reward_asr_model == 'nemo':
            pred_transcripts = self._eval_asr_model.transcribe(
                predicted_audio_paths,
                batch_size=len(predicted_audio_paths),
                override_config=TranscribeConfig(use_lhotse=False, batch_size=len(predicted_audio_paths), num_workers=0),
            )
            pred_transcripts = [process_text_for_cer(transcript.text) for transcript in pred_transcripts]
        else:
            self.whisper_model.to(self.device)
            pred_transcripts = []
            langs = batch_repeated.get('languages', ['en'] * len(predicted_audio_paths))
            for item_idx, audio_path in enumerate(predicted_audio_paths):
                language = langs[item_idx] if item_idx < len(langs) else 'en'
                normalizer = self._get_cached_normalizer(language) if self._normalize_whisper_transcript else None
                print(f"Transcribing audio {audio_path} with language {language}")
                transcript = transcribe_with_whisper(
                    audio_filepath=audio_path,
                    language=language,
                    whisper_processor=self.whisper_processor,
                    whisper_model=self.whisper_model,
                    device=self.device,
                    normalizer=normalizer,
                )
                print(f"Pred Transcript: {transcript}")
                print(f"Normalized Pred Text: {process_text_for_cer(transcript)}")
                print(f"Raw Text: {batch_repeated['raw_texts'][item_idx]}")
                print("--------------------------------")
                pred_transcripts.append(process_text_for_cer(transcript))

        reference_audio_paths = self._get_reference_audio_paths(batch_repeated)
        try:
            pred_speaker_embeddings = get_speaker_embeddings_from_filepaths(
                predicted_audio_paths, self._eval_speaker_verification_model, self.device
            )
            gt_speaker_embeddings = get_speaker_embeddings_from_filepaths(
                reference_audio_paths, self._eval_speaker_verification_model, self.device
            )
        except Exception as e:
            logging.warning(f"Speaker-embedding reward failed. Falling back to zero SSIM reward. Error: {e}")
            pred_speaker_embeddings = None
            gt_speaker_embeddings = None

        batch_metrics = []
        cer_reward_weight = self.cfg.get('cer_reward_weight', 0.5)
        ssim_reward_weight = self.cfg.get('ssim_reward_weight', 0.5)
        pesq_reward_weight = self.cfg.get('pesq_reward_weight', 0.0)
        min_valid_codes_len = self.cfg.get('min_valid_codes_len', 4)
        max_valid_codes_len = self.cfg.get(
            'max_valid_codes_len', self.max_decoder_steps * self.frame_stacking_factor - 1
        )

        for idx in range(predicted_audio.size(0)):
            pred_transcript = pred_transcripts[idx]
            gt_transcript = process_text_for_cer(batch_repeated['raw_texts'][idx])
            cer_gt = min(max(word_error_rate([pred_transcript], [gt_transcript], use_cer=True), 0.0), 1.0)
            wer_gt = min(max(word_error_rate([pred_transcript], [gt_transcript], use_cer=False), 0.0), 1.0)

            if pred_speaker_embeddings is not None and gt_speaker_embeddings is not None:
                spk_embedding_pred = pred_speaker_embeddings[idx].cpu().float().numpy()
                spk_embedding_gt = gt_speaker_embeddings[idx].cpu().float().numpy()
                denom = max(np.linalg.norm(spk_embedding_pred) * np.linalg.norm(spk_embedding_gt), 1e-8)
                spk_similarity = float(np.dot(spk_embedding_pred, spk_embedding_gt) / denom)
            else:
                spk_similarity = 0.0

            if use_pesq:
                sample_audio, sr = torchaudio.load(predicted_audio_paths[idx])
                sample_audio = sample_audio.to(self.device)
                if sr != 16000:
                    sample_audio = torchaudio.functional.resample(sample_audio, sr, 16000)
                _, pesq_hyp, _ = self.squim_objective_model(sample_audio)
                pesq_hyp = float(pesq_hyp.item())
            else:
                pesq_hyp = 0.0

            item_metrics = {
                'cer_gt': float(cer_gt),
                'wer_gt': float(wer_gt),
                'duration': float(audio_durations[idx]),
                'spk_similarity': float(spk_similarity),
                'pred_transcript': pred_transcript,
                'gt_transcript': gt_transcript,
                'codes_len': int(predicted_codes_lens[idx].item()),
                'pesq': float(pesq_hyp),
            }

            best_ssim_achievable = self.cfg.get('best_ssim_achievable', 0.9)
            mean_cer_dataset = self.cfg.get('mean_cer_dataset', 0.1)
            mean_ssim_dataset = self.cfg.get('mean_ssim_dataset', 0.6)

            item_cer = item_metrics['cer_gt']
            item_ssim = max(min(item_metrics['spk_similarity'], best_ssim_achievable), 0.0)
            if item_cer <= mean_cer_dataset:
                cer_reward = 0.5 + 0.5 * (mean_cer_dataset - item_cer) / max(mean_cer_dataset, 1e-8)
            else:
                cer_reward = 0.5 - 0.5 * (item_cer - mean_cer_dataset) / max(1.0 - mean_cer_dataset, 1e-8)

            if item_ssim >= mean_ssim_dataset:
                spk_similarity_reward = 0.5 + 0.5 * (item_ssim - mean_ssim_dataset) / max(
                    best_ssim_achievable - mean_ssim_dataset, 1e-8
                )
            else:
                spk_similarity_reward = 0.5 - 0.5 * (mean_ssim_dataset - item_ssim) / max(mean_ssim_dataset, 1e-8)

            pesq_reward = item_metrics['pesq'] / 4.5 if use_pesq else 0.0
            reward = (
                cer_reward * cer_reward_weight
                + spk_similarity_reward * ssim_reward_weight
                + pesq_reward * pesq_reward_weight
            )
            if (item_metrics['codes_len'] >= max_valid_codes_len) or (item_metrics['codes_len'] <= min_valid_codes_len):
                # reward = 0.0
                pass

            item_metrics['cer_reward'] = float(cer_reward)
            item_metrics['spk_similarity_reward'] = float(spk_similarity_reward)
            item_metrics['pesq_reward'] = float(pesq_reward)
            item_metrics['reward'] = float(reward)
            batch_metrics.append(item_metrics)

        num_groups = len(batch['raw_texts'])
        all_groups_mean_reward = 0.0
        all_groups_std_reward = 0.0
        group_validities = []
        for group_idx in range(num_groups):
            group_start_idx = group_idx * num_generations_per_item
            group_end_idx = group_start_idx + num_generations_per_item
            group_rewards = [batch_metrics[idx]['reward'] for idx in range(group_start_idx, group_end_idx)]
            group_cers = [batch_metrics[idx]['cer_gt'] for idx in range(group_start_idx, group_end_idx)]
            mean_reward = float(np.mean(group_rewards))
            std_reward = float(np.std(group_rewards))
            is_group_valid = True
            if min(group_cers) > self.best_cer_threshold:
                is_group_valid = False
            if max(group_cers) > self.worst_cer_threshold:
                is_group_valid = False

            for idx in range(group_start_idx, group_end_idx):
                advantage = batch_metrics[idx]['reward'] - mean_reward
                if self.scale_rewards:
                    advantage = advantage / (std_reward + 1e-4)
                batch_metrics[idx]['advantage'] = float(advantage)
                group_validities.append(is_group_valid)

            self._print_group_cer_wer_table(
                batch=batch,
                batch_metrics=batch_metrics,
                group_idx=group_idx,
                group_start_idx=group_start_idx,
                group_end_idx=group_end_idx,
                is_group_valid=is_group_valid,
                mean_reward=mean_reward,
                std_reward=std_reward,
            )

            all_groups_mean_reward += mean_reward
            all_groups_std_reward += std_reward

        all_groups_mean_reward = all_groups_mean_reward / max(num_groups, 1)
        all_groups_std_reward = all_groups_std_reward / max(num_groups, 1)
        advantages = torch.tensor([x['advantage'] for x in batch_metrics], device=self.device, dtype=torch.float32)
        group_validities = torch.tensor(group_validities, device=self.device, dtype=torch.float32)
        rewarding_time_sec = time.perf_counter() - rewarding_start_time

        return {
            'mean_reward': torch.tensor(all_groups_mean_reward, device=self.device, dtype=torch.float32),
            'std_reward': torch.tensor(all_groups_std_reward, device=self.device, dtype=torch.float32),
            'batch_repeated': batch_repeated,
            'metrics': batch_metrics,
            'predicted_codes': predicted_codes,
            'predicted_codes_lens': predicted_codes_lens,
            'advantages': advantages,
            'group_validities': group_validities,
            'rollout_phoneme_input_type': phoneme_input_type,
            'timings': {
                'audio_generation_time_sec': float(audio_generation_time_sec),
                'audio_save_time_sec': float(audio_save_time_sec),
                'rewarding_time_sec': float(rewarding_time_sec),
            },
        }

    def process_batch_online_po(self, batch: Dict, n_generations_per_item: int, mode: str = 'train'):
        generated_codes_and_metrics, batch_repeated, predicted_codes, predicted_codes_lens = self._prepare_online_po_inputs(
            batch=batch,
            n_generations_per_item=n_generations_per_item,
            mode=mode,
        )
        chunked_outputs = self._run_teacher_forced_chunked_po(
            generated_codes_and_metrics=generated_codes_and_metrics,
            batch_repeated=batch_repeated,
            predicted_codes=predicted_codes,
            predicted_codes_lens=predicted_codes_lens,
            n_generations_per_item=n_generations_per_item,
            do_backward=False,
        )
        return {
            'mean_reward': generated_codes_and_metrics['mean_reward'],
            'std_reward': generated_codes_and_metrics['std_reward'],
            'loss': chunked_outputs['loss'],
            'po_loss': chunked_outputs['po_loss'],
            'phoneme_aux_loss': chunked_outputs['phoneme_aux_loss'],
            'kl_loss': chunked_outputs['kl_loss'],
            'used_gt_phoneme_input': chunked_outputs['used_gt_phoneme_input'],
            'batch_metrics': generated_codes_and_metrics['metrics'],
        }

    def _slice_batch_range(self, batch: Dict, start_idx: int, end_idx: int) -> Dict:
        sliced_batch = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                sliced_batch[key] = value[start_idx:end_idx]
            elif isinstance(value, list):
                sliced_batch[key] = value[start_idx:end_idx]
            else:
                sliced_batch[key] = value

        # Keep explicit keys only to avoid accidental slicing of non-temporal tensors.
        temporal_key_pairs = [
            ('text', 'text_lens'),
            ('context_text_tokens', 'context_text_tokens_lens'),
            ('audio_codes', 'audio_codes_lens'),
            ('context_audio_codes', 'context_audio_codes_lens'),
            ('phoneme_tokens', 'phoneme_tokens_lens'),
            ('context_audio', 'context_audio_lens'),
            ('audio', 'audio_lens'),
        ]
        for tensor_key, lens_key in temporal_key_pairs:
            tensor_value = sliced_batch.get(tensor_key)
            lens = sliced_batch.get(lens_key)
            if not isinstance(tensor_value, torch.Tensor) or not isinstance(lens, torch.Tensor):
                continue
            if tensor_value.dim() < 2 or tensor_value.size(0) != lens.size(0):
                continue

            local_max_len = int(lens.max().item()) if lens.numel() > 0 else 0
            local_max_len = min(local_max_len, tensor_value.size(-1))
            sliced_batch[tensor_key] = tensor_value[..., :local_max_len]

        return sliced_batch

    def _iter_group_ranges(self, num_groups: int, groups_per_subbatch: int):
        for group_start in range(0, num_groups, groups_per_subbatch):
            yield group_start, min(group_start + groups_per_subbatch, num_groups)

    def _prepare_online_po_inputs(self, batch: Dict, n_generations_per_item: int, mode: str):
        use_local_transformer_for_inference = False
        use_local_transformer_prob = self.cfg.get('use_local_transformer_prob', 0.0)
        if use_local_transformer_prob > 0.0 and mode == 'train':
            use_local_transformer_for_inference = random.random() < use_local_transformer_prob

        with torch.no_grad():
            self.eval()
            generated_codes_and_metrics = self.generate_and_reward(
                batch=batch,
                num_generations_per_item=n_generations_per_item,
                mode=mode,
                use_local_transformer_for_inference=use_local_transformer_for_inference,
            )
            self.train()

        batch_repeated = generated_codes_and_metrics['batch_repeated']
        predicted_codes = generated_codes_and_metrics['predicted_codes']
        predicted_codes_lens = generated_codes_and_metrics['predicted_codes_lens']
        predicted_codes = predicted_codes[:, :, : predicted_codes_lens.max()]
        predicted_codes = self._codec_converter.convert_new_to_original(
            audio_tokens=predicted_codes, audio_lens=predicted_codes_lens
        )
        batch_repeated['audio_codes'] = predicted_codes
        batch_repeated['audio_codes_lens'] = predicted_codes_lens
        if 'audio' in batch_repeated:
            del batch_repeated['audio']
        if 'audio_lens' in batch_repeated:
            del batch_repeated['audio_lens']

        return generated_codes_and_metrics, batch_repeated, predicted_codes, predicted_codes_lens

    def _compute_po_losses_from_outputs(
        self,
        policy_output,
        reference_output,
        advantages: torch.Tensor,
        group_validities: torch.Tensor,
        rollout_phoneme_input_type: str,
    ):
        logits = policy_output.local_transformer_logits
        if logits is None:
            logits = policy_output.logits
        ref_logits = None
        if reference_output is not None:
            ref_logits = reference_output.local_transformer_logits
            if ref_logits is None:
                ref_logits = reference_output.logits

        audio_codes_target = policy_output.audio_codes_target.long()
        audio_codes_lens_target = policy_output.audio_codes_lens_target
        audio_loss_mask = get_mask_from_lengths(audio_codes_lens_target).float()

        n_codebooks = audio_codes_target.size(1)
        total_loss = None
        total_kl = None
        for codebook_idx in range(n_codebooks):
            si = codebook_idx * self.num_all_tokens_per_codebook
            ei = si + self.num_all_tokens_per_codebook
            codebook_logits = logits[:, :, si:ei]
            codebook_labels = audio_codes_target[:, codebook_idx, :]
            per_token_logps = self._get_per_token_logps(codebook_logits, codebook_labels, audio_loss_mask)
            # Ensure the GRPO policy gradient trick stays in fp32 to preserve gradient signal
            with torch.cuda.amp.autocast(enabled=False):
                per_token_loss = -(torch.exp(per_token_logps.float() - per_token_logps.float().detach()) * advantages.float().unsqueeze(1))
                per_token_loss = per_token_loss * group_validities.float().unsqueeze(1)

            if not self.reference_free and ref_logits is not None:
                with torch.no_grad():
                    ref_codebook_logits = ref_logits[:, :, si:ei]
                    per_token_ref_logps = self._get_per_token_logps(
                        ref_codebook_logits, codebook_labels, audio_loss_mask
                    )
                with torch.cuda.amp.autocast(enabled=False):
                    per_token_kl = (
                        torch.exp(per_token_ref_logps.float() - per_token_logps.float()) - (per_token_ref_logps.float() - per_token_logps.float()) - 1
                    )
                    per_token_loss = per_token_loss + self.cfg.get('grpo_beta', 0.0) * per_token_kl
                codebook_kl_loss_mean = (
                    (per_token_kl * audio_loss_mask).sum(dim=1) / audio_loss_mask.sum(dim=1).clamp_min(1e-8)
                ).mean()
            else:
                codebook_kl_loss_mean = torch.tensor(0.0, device=self.device)

            if self.loss_type == "grpo":
                codebook_loss = (
                    (per_token_loss * audio_loss_mask).sum(dim=1) / audio_loss_mask.sum(dim=1).clamp_min(1e-8)
                ).mean()
            elif self.loss_type == "dr_grpo":
                total_tokens = per_token_loss.shape[0] * self.max_decoder_steps
                codebook_loss = (per_token_loss * audio_loss_mask).sum() / max(total_tokens, 1)
            else:
                raise ValueError(f"Unknown loss function: {self.loss_type}")

            if total_loss is None:
                total_loss = codebook_loss
                total_kl = codebook_kl_loss_mean
            else:
                total_loss += codebook_loss
                total_kl += codebook_kl_loss_mean

        total_po_loss = total_loss / n_codebooks
        total_kl = total_kl / n_codebooks

        phoneme_aux_loss = policy_output.phoneme_loss if rollout_phoneme_input_type == 'gt' else None
        if phoneme_aux_loss is None:
            phoneme_aux_loss = torch.tensor(0.0, device=self.device)
        total_loss = total_po_loss + self.aux_phoneme_loss_weight * phoneme_aux_loss

        return {
            'loss': total_loss,
            'po_loss': total_po_loss,
            'phoneme_aux_loss': phoneme_aux_loss,
            'kl_loss': total_kl,
            'used_gt_phoneme_input': float(rollout_phoneme_input_type == 'gt'),
        }

    @staticmethod
    def _trace_grad_graph(tensor, target_param_ids: set, max_depth: int = 500) -> List[str]:
        """Walk the autograd graph from `tensor` and report whether any target param ids are found."""
        visited = set()
        found_params = []
        path_info = []
        depth_limited_count = 0
        node_type_counts = {}  # Track node types for debugging
        max_depth_reached = 0

        def _walk(node, depth):
            nonlocal depth_limited_count, max_depth_reached
            if node is None or id(node) in visited:
                return
            if depth > max_depth:
                depth_limited_count += 1
                return
            visited.add(id(node))
            max_depth_reached = max(max_depth_reached, depth)

            node_type = type(node).__name__
            node_type_counts[node_type] = node_type_counts.get(node_type, 0) + 1

            # AccumulateGrad nodes hold the .variable (leaf parameter)
            if hasattr(node, 'variable'):
                var = node.variable
                pid = id(var)
                tag = "TARGET" if pid in target_param_ids else "other"
                found_params.append((tag, pid, var.shape, var.dtype))
                if len(found_params) <= 20:  # cap output
                    path_info.append(f"    depth={depth} AccumulateGrad param shape={list(var.shape)} dtype={var.dtype} [{tag}]")
            for child_fn, _ in node.next_functions:
                _walk(child_fn, depth + 1)

        if tensor is not None and tensor.grad_fn is not None:
            _walk(tensor.grad_fn, 0)

        n_target = sum(1 for t in found_params if t[0] == "TARGET")
        n_other = sum(1 for t in found_params if t[0] == "other")
        summary = (
            f"  grad_graph: visited {len(visited)} nodes, max_depth_reached={max_depth_reached}, "
            f"depth_limited={depth_limited_count}, "
            f"found {n_target} TARGET params, {n_other} other params"
        )
        # Show top node types
        top_types = sorted(node_type_counts.items(), key=lambda x: -x[1])[:15]
        type_str = f"  node_types (top 15): {dict(top_types)}"
        return [summary, type_str] + path_info

    def _diagnose_local_transformer_out_projections(self, loss_tensor, policy_output, advantages, step_info: str):
        """Deep diagnostic for local_transformer_out_projections gradient flow.
        
        Called BEFORE backward to inspect the computation graph, and can be called
        AFTER backward to inspect populated gradients.
        """
        if not getattr(self.trainer, "is_global_zero", True):
            return

        lines = [f"\n[LOCAL_TRANSFORMER_OUT_PROJ DIAGNOSTIC] {step_info}"]

        # 1. Check if local_transformer_out_projections exists and has params
        if not hasattr(self, 'local_transformer_out_projections'):
            lines.append("  ERROR: self.local_transformer_out_projections does not exist!")
            msg = "\n".join(lines)
            print(msg)
            logging.info(msg)
            return

        lt_out_projs = self.local_transformer_out_projections
        lines.append(f"  num projection layers: {len(lt_out_projs)}")

        # Collect target param ids for graph tracing
        target_param_ids = set()
        for proj in lt_out_projs:
            target_param_ids.add(id(proj.weight))
            if proj.bias is not None:
                target_param_ids.add(id(proj.bias))

        # 2. Check requires_grad, dtype, and grad on each projection layer
        for i, proj in enumerate(lt_out_projs):
            w = proj.weight
            b = proj.bias if proj.bias is not None else None
            w_rg = w.requires_grad
            b_rg = b.requires_grad if b is not None else "N/A"
            w_grad = "None" if w.grad is None else f"norm={w.grad.data.norm(2).item():.8f}, dtype={w.grad.dtype}"
            b_grad = "None" if (b is None or b.grad is None) else f"norm={b.grad.data.norm(2).item():.8f}, dtype={b.grad.dtype}"
            lines.append(
                f"  proj[{i}] weight: requires_grad={w_rg}, dtype={w.dtype}, grad={w_grad}, "
                f"weight_norm={w.data.norm(2).item():.6f} | "
                f"bias: requires_grad={b_rg}, dtype={b.dtype if b is not None else 'N/A'}, grad={b_grad}"
            )

        # 3. Check the logits tensor from policy_output (shape, dtype, grad_fn)
        lt_logits = policy_output.local_transformer_logits if policy_output is not None else None
        if lt_logits is not None:
            lines.append(
                f"  local_transformer_logits: shape={list(lt_logits.shape)}, dtype={lt_logits.dtype}, "
                f"requires_grad={lt_logits.requires_grad}, "
                f"grad_fn={lt_logits.grad_fn}, "
                f"is_leaf={lt_logits.is_leaf}"
            )
        else:
            lines.append("  local_transformer_logits: None (USING FALLBACK logits!)")

        # 4. Check the loss tensor
        if loss_tensor is not None:
            lines.append(
                f"  loss: value={loss_tensor.item():.8f}, dtype={loss_tensor.dtype}, "
                f"requires_grad={loss_tensor.requires_grad}, "
                f"grad_fn={loss_tensor.grad_fn}"
            )
            # Trace the autograd graph from loss to see if out_proj params are reachable
            lines.append("  --- Tracing autograd graph from loss ---")
            lines.extend(self._trace_grad_graph(loss_tensor, target_param_ids))

        # 5. Check advantages
        if advantages is not None:
            lines.append(
                f"  advantages: mean={advantages.mean().item():.6f}, std={advantages.std().item():.6f}, "
                f"abs_mean={advantages.abs().mean().item():.6f}, all_zero={bool((advantages == 0).all())}"
            )

        # 6. Check other modules in the local transformer path (grad status + dtype)
        for attr_name in ['local_transformer', 'local_transformer_in_projection',
                          'local_transformer_audio_out_projection']:
            if hasattr(self, attr_name):
                mod = getattr(self, attr_name)
                params = list(mod.parameters())
                n_with_grad = sum(1 for p in params if p.grad is not None)
                n_total = len(params)
                grad_norms = [p.grad.data.norm(2).item() for p in params if p.grad is not None]
                max_grad = max(grad_norms) if grad_norms else 0.0
                dtypes = set(str(p.dtype) for p in params)
                lines.append(
                    f"  {attr_name}: {n_with_grad}/{n_total} params have grad, "
                    f"max_grad_norm={max_grad:.8f}, param_dtypes={dtypes}"
                )

        # 7. Check if torch.is_grad_enabled()
        lines.append(f"  torch.is_grad_enabled()={torch.is_grad_enabled()}")
        lines.append(f"  torch.is_autocast_enabled('cuda')={torch.is_autocast_enabled('cuda')}")

        msg = "\n".join(lines)
        print(msg)
        logging.info(msg)

    def _run_teacher_forced_chunked_po(
        self,
        generated_codes_and_metrics: Dict,
        batch_repeated: Dict,
        predicted_codes: torch.Tensor,
        predicted_codes_lens: torch.Tensor,
        n_generations_per_item: int,
        do_backward: bool,
    ):
        num_groups = len(batch_repeated['raw_texts']) // n_generations_per_item
        groups_per_subbatch = max(self.po_groups_per_subbatch, 1)

        accumulated_loss = torch.tensor(0.0, device=self.device)
        accumulated_po_loss = torch.tensor(0.0, device=self.device)
        accumulated_phoneme_aux_loss = torch.tensor(0.0, device=self.device)
        accumulated_kl_loss = torch.tensor(0.0, device=self.device)
        used_gt_phoneme_input = 0.0

        is_first_chunk = True  # Only run deep diagnostic on first chunk to avoid spam

        for group_start_idx, group_end_idx in self._iter_group_ranges(num_groups, groups_per_subbatch):
            item_start_idx = group_start_idx * n_generations_per_item
            item_end_idx = group_end_idx * n_generations_per_item
            group_weight = float(group_end_idx - group_start_idx) / max(float(num_groups), 1.0)

            batch_sub = self._slice_batch_range(batch_repeated, item_start_idx, item_end_idx)
            predicted_codes_sub = predicted_codes[item_start_idx:item_end_idx]
            predicted_codes_lens_sub = predicted_codes_lens[item_start_idx:item_end_idx]
            predicted_codes_sub = predicted_codes_sub[:, :, : predicted_codes_lens_sub.max()]
            advantages_sub = generated_codes_and_metrics['advantages'][item_start_idx:item_end_idx]
            group_validities_sub = generated_codes_and_metrics['group_validities'][item_start_idx:item_end_idx]
            rollout_phoneme_input_type = generated_codes_and_metrics.get('rollout_phoneme_input_type', 'pred')

            # Use mode='val' intentionally for stable PO optimization:
            # no random input dropout, no CFG unconditional dropout, no random phoneme corruption.
            policy_output = self._run_easy_process_batch(
                model=self,
                batch=batch_sub,
                audio_codes=predicted_codes_sub,
                audio_codes_lens=predicted_codes_lens_sub,
                mode='val',
            )

            reference_output = None
            if not self.reference_free:
                with torch.no_grad():
                    reference_output = self._run_easy_process_batch(
                        model=self._reference_model,
                        batch=batch_sub,
                        audio_codes=predicted_codes_sub,
                        audio_codes_lens=predicted_codes_lens_sub,
                        mode='val',
                    )

            chunk_outputs = self._compute_po_losses_from_outputs(
                policy_output=policy_output,
                reference_output=reference_output,
                advantages=advantages_sub,
                group_validities=group_validities_sub,
                rollout_phoneme_input_type=rollout_phoneme_input_type,
            )

            # Deep diagnostic BEFORE backward (check computation graph)
            if is_first_chunk and do_backward:
                self._diagnose_local_transformer_out_projections(
                    loss_tensor=chunk_outputs['loss'] * group_weight,
                    policy_output=policy_output,
                    advantages=advantages_sub,
                    step_info=f"BEFORE backward, step={self.global_step}",
                )

                # --- DEFINITIVE GRADIENT CONNECTIVITY TEST ---
                # Test 1: Check if the PO loss connects to out_proj params
                loss_for_test = chunk_outputs['loss'] * group_weight
                test_params = []
                test_names = []
                for i, proj in enumerate(self.local_transformer_out_projections):
                    test_params.append(proj.weight)
                    test_names.append(f"out_proj[{i}].weight")
                try:
                    test_grads = torch.autograd.grad(
                        loss_for_test, test_params,
                        retain_graph=True, allow_unused=True,
                    )
                    grad_test_lines = [f"\n[AUTOGRAD.GRAD TEST 1: PO loss → out_proj] step={self.global_step}"]
                    for name, g in zip(test_names, test_grads):
                        if g is None:
                            grad_test_lines.append(f"  {name}: grad=None (NOT CONNECTED)")
                        else:
                            grad_test_lines.append(
                                f"  {name}: grad_norm={g.norm(2).item():.8f}, dtype={g.dtype}"
                            )
                    grad_test_msg = "\n".join(grad_test_lines)
                    print(grad_test_msg)
                    logging.info(grad_test_msg)
                except Exception as e:
                    err_msg = f"\n[AUTOGRAD.GRAD TEST 1] ERROR: {e}"
                    print(err_msg)
                    logging.info(err_msg)

                # Test 2: Check if a SIMPLE CE loss on local_transformer_logits connects to out_proj params
                # This isolates whether the problem is in the GRPO formula or in the logits tensor itself
                try:
                    lt_logits = policy_output.local_transformer_logits
                    dummy_targets = policy_output.audio_codes_target[:, 0, :].long()  # first codebook
                    simple_ce_loss = torch.nn.functional.cross_entropy(
                        lt_logits[:, :, :self.num_all_tokens_per_codebook].reshape(-1, self.num_all_tokens_per_codebook),
                        dummy_targets.reshape(-1),
                    )
                    test_grads_ce = torch.autograd.grad(
                        simple_ce_loss, [self.local_transformer_out_projections[0].weight],
                        retain_graph=True, allow_unused=True,
                    )
                    g = test_grads_ce[0]
                    if g is None:
                        ce_msg = f"\n[AUTOGRAD.GRAD TEST 2: CE loss → out_proj[0]] grad=None (NOT CONNECTED — logits tensor is detached!)"
                    else:
                        ce_msg = f"\n[AUTOGRAD.GRAD TEST 2: CE loss → out_proj[0]] grad_norm={g.norm(2).item():.8f} (CONNECTED!)"
                    print(ce_msg)
                    logging.info(ce_msg)
                except Exception as e:
                    err_msg = f"\n[AUTOGRAD.GRAD TEST 2] ERROR: {e}"
                    print(err_msg)
                    logging.info(err_msg)

                # Test 3: Check if PO loss connects to the decoder (main transformer)
                try:
                    decoder_param = next(self.decoder.parameters())
                    test_grads_dec = torch.autograd.grad(
                        loss_for_test, [decoder_param],
                        retain_graph=True, allow_unused=True,
                    )
                    g = test_grads_dec[0]
                    if g is None:
                        dec_msg = f"\n[AUTOGRAD.GRAD TEST 3: PO loss → decoder] grad=None (NOT CONNECTED)"
                    else:
                        dec_msg = f"\n[AUTOGRAD.GRAD TEST 3: PO loss → decoder] grad_norm={g.norm(2).item():.8f} (CONNECTED)"
                    print(dec_msg)
                    logging.info(dec_msg)
                except Exception as e:
                    err_msg = f"\n[AUTOGRAD.GRAD TEST 3] ERROR: {e}"
                    print(err_msg)
                    logging.info(err_msg)
                # --- END CONNECTIVITY TESTS ---

            if do_backward:
                self.manual_backward(chunk_outputs['loss'] * group_weight)

            # Deep diagnostic AFTER backward (check populated gradients)
            if is_first_chunk and do_backward:
                self._diagnose_local_transformer_out_projections(
                    loss_tensor=None,
                    policy_output=policy_output,
                    advantages=advantages_sub,
                    step_info=f"AFTER backward, step={self.global_step}",
                )
                is_first_chunk = False

            accumulated_loss = accumulated_loss + chunk_outputs['loss'].detach() * group_weight
            accumulated_po_loss = accumulated_po_loss + chunk_outputs['po_loss'].detach() * group_weight
            accumulated_phoneme_aux_loss = (
                accumulated_phoneme_aux_loss + chunk_outputs['phoneme_aux_loss'].detach() * group_weight
            )
            accumulated_kl_loss = accumulated_kl_loss + chunk_outputs['kl_loss'].detach() * group_weight
            used_gt_phoneme_input = max(used_gt_phoneme_input, chunk_outputs['used_gt_phoneme_input'])

        return {
            'loss': accumulated_loss,
            'po_loss': accumulated_po_loss,
            'phoneme_aux_loss': accumulated_phoneme_aux_loss,
            'kl_loss': accumulated_kl_loss,
            'used_gt_phoneme_input': used_gt_phoneme_input,
        }

    def training_step(self, batch, batch_idx):
        n_generations_per_item = self.cfg.get('n_generations_per_item', 6)
        optimizer = self.optimizers()
        if isinstance(optimizer, (list, tuple)):
            if len(optimizer) != 1:
                raise ValueError(f"Expected a single optimizer, got {len(optimizer)}.")
            optimizer = optimizer[0]
        optimizer.zero_grad(set_to_none=True)

        # Snapshot weights before optimizer step to measure weight deltas.
        prev_weights = self._snapshot_trainable_weights()

        generated_codes_and_metrics, batch_repeated, predicted_codes, predicted_codes_lens = self._prepare_online_po_inputs(
            batch=batch,
            n_generations_per_item=n_generations_per_item,
            mode='train',
        )
        teacher_forced_start_time = time.perf_counter()
        po_outputs = self._run_teacher_forced_chunked_po(
            generated_codes_and_metrics=generated_codes_and_metrics,
            batch_repeated=batch_repeated,
            predicted_codes=predicted_codes,
            predicted_codes_lens=predicted_codes_lens,
            n_generations_per_item=n_generations_per_item,
            do_backward=True,
        )
        teacher_forced_time_sec = time.perf_counter() - teacher_forced_start_time

        # Compute gradient metrics BEFORE optimizer.step() clears them.
        grad_weight_metrics = self._compute_grad_and_weight_metrics()

        optimizer.step()

        # Step the LR scheduler (required in manual optimization mode).
        lr_schedulers = self.lr_schedulers()
        if lr_schedulers is not None:
            if isinstance(lr_schedulers, (list, tuple)):
                for sched in lr_schedulers:
                    sched.step()
            else:
                lr_schedulers.step()

        # Compute weight update (delta) metrics AFTER optimizer.step().
        weight_delta_metrics = self._compute_weight_update_metrics(prev_weights)
        grad_weight_metrics.update(weight_delta_metrics)

        # Also log advantage statistics for diagnosing flat rewards.
        advantages = generated_codes_and_metrics['advantages']
        grad_weight_metrics['advantages/mean'] = float(advantages.mean().item())
        grad_weight_metrics['advantages/std'] = float(advantages.std().item())
        grad_weight_metrics['advantages/max'] = float(advantages.max().item())
        grad_weight_metrics['advantages/min'] = float(advantages.min().item())
        grad_weight_metrics['advantages/abs_mean'] = float(advantages.abs().mean().item())
        valid_frac = float(generated_codes_and_metrics['group_validities'].mean().item())
        grad_weight_metrics['group_validity_fraction'] = valid_frac

        # Log learning rate
        current_lr = optimizer.param_groups[0]['lr']
        self.log('learning_rate', current_lr, prog_bar=False, sync_dist=True)

        self.log('train_loss', po_outputs['loss'], prog_bar=True, sync_dist=True)
        self.log('train_po_loss', po_outputs['po_loss'], prog_bar=True, sync_dist=True)
        self.log('train_phoneme_aux_loss', po_outputs['phoneme_aux_loss'], prog_bar=True, sync_dist=True)
        self.log('train_kl_loss', po_outputs['kl_loss'], prog_bar=True, sync_dist=True)
        self.log('train_used_gt_phoneme_input', po_outputs['used_gt_phoneme_input'], prog_bar=True, sync_dist=True)
        self.log('train_mean_reward', generated_codes_and_metrics['mean_reward'], prog_bar=True, sync_dist=True)
        self.log('train_std_reward', generated_codes_and_metrics['std_reward'], prog_bar=True, sync_dist=True)

        # Log all gradient / weight / advantage diagnostics to wandb.
        for metric_name, metric_value in grad_weight_metrics.items():
            self.log(f'train_{metric_name}', metric_value, prog_bar=False, sync_dist=True)

        # Print human-readable summary to stdout / log file.
        self._print_grad_weight_summary(grad_weight_metrics, step=self.global_step)

        timings = generated_codes_and_metrics.get('timings', {})
        audio_generation_time_sec = float(timings.get('audio_generation_time_sec', 0.0))
        audio_save_time_sec = float(timings.get('audio_save_time_sec', 0.0))
        rewarding_time_sec = float(timings.get('rewarding_time_sec', 0.0))
        self.log('train_audio_generation_time_sec', audio_generation_time_sec, prog_bar=False, sync_dist=True)
        self.log('train_audio_save_time_sec', audio_save_time_sec, prog_bar=False, sync_dist=True)
        self.log('train_rewarding_time_sec', rewarding_time_sec, prog_bar=False, sync_dist=True)
        self.log('train_teacher_forced_time_sec', teacher_forced_time_sec, prog_bar=False, sync_dist=True)
        timing_msg = (
            f"[training_step_timing] step={self.global_step} batch_idx={batch_idx} "
            f"audio_gen={audio_generation_time_sec:.4f}s "
            f"audio_save={audio_save_time_sec:.4f}s "
            f"rewarding={rewarding_time_sec:.4f}s "
            f"teacher_forced={teacher_forced_time_sec:.4f}s"
        )
        print(timing_msg)
        logging.info(timing_msg)

    # def validation_step(self, batch, batch_idx):
    #     val_n_generations_per_item = self.cfg.get('val_n_generations_per_item', 1)
    #     po_outputs = self.process_batch_online_po(
    #         batch=batch,
    #         n_generations_per_item=val_n_generations_per_item,
    #         mode='val',
    #     )
    #     self.validation_step_outputs.append(
    #         {
    #             'mean_reward': po_outputs['mean_reward'],
    #             'std_reward': po_outputs['std_reward'],
    #             'val_loss': po_outputs['loss'],
    #             'val_po_loss': po_outputs['po_loss'],
    #             'val_phoneme_aux_loss': po_outputs['phoneme_aux_loss'],
    #             'val_kl_loss': po_outputs['kl_loss'],
    #             'val_used_gt_phoneme_input': torch.tensor(
    #                 po_outputs['used_gt_phoneme_input'], device=self.device, dtype=torch.float32
    #             ),
    #             'batch_metrics': po_outputs['batch_metrics'],
    #         }
    #     )

    # def on_validation_epoch_end(self):
    #     def collect(key: str):
    #         values = []
    #         for x in self.validation_step_outputs:
    #             if x[key] is not None:
    #                 values.append(x[key])
    #             else:
    #                 values.append(torch.tensor(0.0, device=self.device))
    #         return torch.stack(values).mean() if len(values) > 0 else torch.tensor(0.0, device=self.device)

    #     val_loss = collect("val_loss")
    #     val_po_loss = collect("val_po_loss")
    #     val_phoneme_aux_loss = collect("val_phoneme_aux_loss")
    #     val_kl_loss = collect("val_kl_loss")
    #     val_used_gt_phoneme_input = collect("val_used_gt_phoneme_input")
    #     mean_reward = collect("mean_reward")
    #     std_reward = collect("std_reward")

    #     self.log("val_loss", val_loss, prog_bar=True, sync_dist=True)
    #     self.log("val_po_loss", val_po_loss, prog_bar=True, sync_dist=True)
    #     self.log("val_phoneme_aux_loss", val_phoneme_aux_loss, prog_bar=True, sync_dist=True)
    #     self.log("val_kl_loss", val_kl_loss, prog_bar=True, sync_dist=True)
    #     self.log("val_used_gt_phoneme_input", val_used_gt_phoneme_input, prog_bar=True, sync_dist=True)
    #     self.log("val_mean_reward", mean_reward, prog_bar=True, sync_dist=True)
    #     self.log("val_std_reward", std_reward, prog_bar=True, sync_dist=True)

    #     mean_metrics = {}
    #     for val_output in self.validation_step_outputs:
    #         for item_metrics in val_output['batch_metrics']:
    #             for key, value in item_metrics.items():
    #                 if "transcript" not in key:
    #                     mean_metrics.setdefault(key, []).append(value)
    #     for key, values in mean_metrics.items():
    #         self.log(f"val_{key}", float(np.mean(values)), prog_bar=True, sync_dist=True)

    #     self.validation_step_outputs.clear()
