import numpy as np
import torch
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from torch import nn
import os
import json
from nemo.utils import logging
import nemo.collections.asr as nemo_asr
import soundfile as sf
import librosa
import copy
from omegaconf import open_dict
import string
from nemo.collections.asr.metrics.wer import word_error_rate
from nemo.collections.tts.parts.utils.tts_dataset_utils import stack_tensors
import random
from nemo.collections.tts.models import T5TTS_Model

class T5TTS_Model_PrefDataGen(T5TTS_Model):
    """Small override to save inference metrics, used for datagen in DPO"""
    def __init__(self, cfg: DictConfig, trainer: 'Trainer' = None):
        super().__init__(cfg, trainer)
        if cfg.get('pref_set_language', "en") == "en":
            self.eval_asr_model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained(model_name="nvidia/parakeet-ctc-0.6b")
            self.eval_asr_model.freeze()
            self.eval_asr_model.eval()

        self.eval_speaker_verification_model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(model_name='titanet_large')
        self.eval_speaker_verification_model.freeze()
        self.eval_speaker_verification_model.eval()

        if cfg.get('load_whisper_model', False):
            from transformers import WhisperProcessor, WhisperForConditionalGeneration
            self.whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3")
            self.whisper_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3")
            self.whisper_model.eval()

    def test_step(self, batch, batch_idx):
        with torch.no_grad():
            test_dl_batch_size = self._test_dl.batch_size
            temperature = self.cfg.get('inference_temperature', 0.7)
            topk = self.cfg.get('inference_topk', 80)
            use_cfg = self.cfg.get('inference_use_cfg', False)
            cfg_scale = self.cfg.get('inference_cfg_scale', 1.0)
            predicted_audio, predicted_audio_lens, predicted_codes, predicted_codes_lens = self.infer_batch(
                batch,
                max_decoder_steps=self.cfg.get('max_decoder_steps', 500),
                temperature=temperature,
                topk=topk,
                use_cfg=use_cfg,
                cfg_scale=cfg_scale
            )
            predicted_audio_paths = []
            audio_durations = []
            batch_invalid = False
            for idx in range(predicted_audio.size(0)):
                predicted_audio_np = predicted_audio[idx].float().detach().cpu().numpy()
                predicted_audio_np = predicted_audio_np[:predicted_audio_lens[idx]]
                item_idx = batch_idx * test_dl_batch_size + idx
                # Save the predicted audio
                log_dir = self.logger.log_dir
                audio_dir = os.path.join(log_dir, 'audios')
                if not os.path.exists(audio_dir):
                    os.makedirs(audio_dir)
                audio_path = os.path.join(audio_dir, f'predicted_audioRank{self.global_rank}_{item_idx}.wav')
                audio_durations.append(len(predicted_audio_np) / self.cfg.sample_rate)
                sf.write(audio_path, predicted_audio_np, self.cfg.sample_rate)

                predicted_codes_torch = predicted_codes[idx].cpu().type(torch.int16)
                predicted_codes_torch = predicted_codes_torch[:, :predicted_codes_lens[idx]]
                torch.save(predicted_codes_torch, os.path.join(audio_dir, f'predicted_audioRank{self.global_rank}_{item_idx}_codes.pt'))
                predicted_audio_paths.append(audio_path)
                
                if not batch_invalid:
                    with torch.no_grad():
                        try:
                            if self.cfg.get("pref_set_language", "en") == "en":
                                pred_transcripts = self.eval_asr_model.transcribe(predicted_audio_paths, batch_size=len(predicted_audio_paths))[0]
                                pred_transcripts = [ process_text_for_cer(transcript) for transcript in pred_transcripts ]
                            else:
                                pred_transcripts = []
                                for audio_path in predicted_audio_paths:
                                    transcript = transcribe_with_whisper(audio_path, self.cfg.pref_set_language, self.whisper_processor, self.whisper_model, self.device)
                                    pred_transcripts.append(transcript)

                                pred_transcripts = [process_text_for_cer(transcript) for transcript in pred_transcripts]
                        except Exception as e:
                            assert (predicted_audio_lens[idx] < 1000).any(), f"Expected short audio file to be the only cause of ASR errors, but got error with lengths {predicted_audio_lens}"
                            logging.warning(f"Exception during ASR transcription: {e}")
                            logging.warning(f"Skipping processing of the batch; generating metrics indicating a WER of 100% and Speaker Similarity of 0.0")
                            batch_invalid = True
                            continue # don't break since we want to continue building audio durations list
                        pred_speaker_embeddings = get_speaker_embeddings_from_filepaths(predicted_audio_paths, self.eval_speaker_verification_model, self.device)
                        gt_speaker_embeddings = get_speaker_embeddings_from_filepaths(batch['audio_filepaths'], self.eval_speaker_verification_model, self.device)

            for idx in range(predicted_audio.size(0)):
                if not batch_invalid:
                    audio_path = predicted_audio_paths[idx]
                    item_idx = batch_idx * test_dl_batch_size + idx
                    pred_transcript = pred_transcripts[idx]
                    gt_transcript = process_text_for_cer(batch['raw_texts'][idx])

                    cer_gt = word_error_rate([pred_transcript], [gt_transcript], use_cer=True)
                    wer_gt = word_error_rate([pred_transcript], [gt_transcript], use_cer=False)

                    spk_embedding_pred = pred_speaker_embeddings[idx].cpu().numpy()
                    spk_embedding_gt = gt_speaker_embeddings[idx].cpu().numpy()
                    
                    spk_similarity = np.dot(spk_embedding_pred, spk_embedding_gt) / (
                        np.linalg.norm(spk_embedding_pred) * np.linalg.norm(spk_embedding_gt)
                    )
                else:
                    # Create an entry indicating invalid metrics
                    cer_gt = 1.0
                    wer_gt = 1.0
                    spk_similarity = 0.0
                    pred_transcript = "<INVALID>" # do not change this string; subsequent processing relies on it
                    gt_transcript = process_text_for_cer(batch['raw_texts'][idx])

                item_metrics = {
                    'cer_gt': float(cer_gt),
                    'wer_gt': float(wer_gt),
                    'duration' : audio_durations[idx],
                    'spk_similarity': float(spk_similarity),
                    'pred_transcript': pred_transcript,
                    'gt_transcript': gt_transcript,
                }

                with open(os.path.join(audio_dir, f'predicted_audioRank{self.global_rank}_{item_idx}_metrics.json'), 'w') as f:
                    json.dump(item_metrics, f)

class T5TTS_Model_OfflinePO(T5TTS_Model):
    def __init__(self, cfg: DictConfig, trainer: 'Trainer' = None):
        super().__init__(cfg, trainer)
        # Copy cfg
        ref_model_cfg = copy.deepcopy(cfg)
        with open_dict(ref_model_cfg):
            ref_model_cfg.train_ds = None
            ref_model_cfg.validation_ds = None
        self._reference_model = T5TTS_Model(cfg=ref_model_cfg)
        print("Loading reference model from checkpoint")
        self._reference_model.load_state_dict(torch.load(cfg.reference_model_ckpt_path, map_location="cpu")['state_dict'])
        self.freeze_model(self._reference_model)
        self._reference_model.eval()
        self._reference_model._no_state_dict = True
        print("Reference model loaded and frozen")
    
    def state_dict(self, destination=None, prefix='', keep_vars=False):
        state_dict = super().state_dict(destination, prefix, keep_vars)
        keys_substrings_to_exclude = ['_speaker_verification_model', '_codec_model', '_reference_model']
        for key in list(state_dict.keys()):
            if any([substring in key for substring in keys_substrings_to_exclude]):
                del state_dict[key]
        return state_dict
        

    def _get_batch_logps(self, logits, labels, loss_mask, average_log_prob=False):
        """Compute the log probabilities of the given labels under the given logits.

        Args:
            logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
            labels: Labels for which to compute the log probabilities. Label tokens with a value of -100 are ignored. Shape: (batch_size, sequence_length)
            average_log_prob: If True, return the average log probability per (non-masked) token. Otherwise, return the sum of the log probabilities of the (non-masked) tokens.

        Returns:
            A tensor of shape (batch_size,) containing the average/sum log probabilities of the given labels under the given logits.
        """
        per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)

        if average_log_prob:
            return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
        else:
            return (per_token_logps * loss_mask).sum(-1)

    # https://github.com/eric-mitchell/direct-preference-optimization/blob/main/trainers.py
    def preference_loss(self, policy_chosen_logps,
                    policy_rejected_logps,
                    reference_chosen_logps,
                    reference_rejected_logps,
                    chosen_gt_rewards=None,
                    rejected_gt_rewards=None,
                    beta=0.2,
                    gt_reward_scale=1.0,
                    label_smoothing=0,
                    loss_type="dpo",
                    reference_free=False):
        """Compute the DPO loss for a batch of policy and reference model log probabilities.

        Args:
            policy_chosen_logps: Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
            policy_rejected_logps: Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)
            reference_chosen_logps: Log probabilities of the reference model for the chosen responses. Shape: (batch_size,)
            reference_rejected_logps: Log probabilities of the reference model for the rejected responses. Shape: (batch_size,)
            beta: Temperature parameter for the DPO loss, typically something in the range of 0.1 to 0.5. We ignore the reference model as beta -> 0.
            label_smoothing: conservativeness for DPO loss, which assumes that preferences are noisy (flipped with probability label_smoothing)
            ipo: If True, use the IPO loss instead of the DPO loss.
            reference_free: If True, we ignore the _provided_ reference model and implicitly use a reference model that assigns equal probability to all responses.

        Returns:
            A tuple of three tensors: (losses, chosen_rewards, rejected_rewards).
            The losses tensor contains the DPO loss for each example in the batch.
            The chosen_rewards and rejected_rewards tensors contain the rewards for the chosen and rejected responses, respectively.
        """
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = reference_chosen_logps - reference_rejected_logps

        if reference_free:
            ref_logratios = 0

        logits = pi_logratios - ref_logratios  # also known as h_{\pi_\theta}^{y_w,y_l}
        # logits = (policy_chosen_logps - policy_rejected_logps) - (reference_chosen_logps - reference_rejected_logps)
        # logits = (policy_chosen_logps - reference_chosen_logps) - (policy_rejected_logps - reference_rejected_logps)
        # logits is the same as rewards_delta in NeMo aligner: https://github.com/NVIDIA/NeMo-Aligner/blob/0b5bffeb78a8316dd57e0816a2a9544540f0c8dd/nemo_aligner/models/nlp/gpt/megatron_gpt_dpo_model.py#L241

        if loss_type == "ipo":
            losses = (logits - 1/(2 * beta)) ** 2  # Eq. 17 of https://arxiv.org/pdf/2310.12036v2.pdf
        elif loss_type == "rpo":
            # https://github.com/NVIDIA/NeMo-Aligner/blob/0b5bffeb78a8316dd57e0816a2a9544540f0c8dd/nemo_aligner/models/nlp/gpt/megatron_gpt_dpo_model.py#L241
            logbeta_hat_chosen = torch.nn.functional.logsigmoid(beta * logits)
            logbeta_hat_rejected = torch.nn.functional.logsigmoid(-beta * logits)
            gt_rewards_delta = gt_reward_scale * (chosen_gt_rewards - rejected_gt_rewards)
            logalpha_hat_chosen = torch.nn.functional.logsigmoid(gt_rewards_delta)
            logalpha_hat_rejected = torch.nn.functional.logsigmoid(-gt_rewards_delta)
            losses = (
                torch.exp(logalpha_hat_chosen) * (logalpha_hat_chosen - logbeta_hat_chosen)
                + torch.exp(logalpha_hat_rejected) * (logalpha_hat_rejected - logbeta_hat_rejected)
            )
        elif loss_type == "rpo_sq":
            gt_rewards_delta = gt_reward_scale * (chosen_gt_rewards - rejected_gt_rewards)
            losses = (beta * logits - gt_rewards_delta) ** 2
        elif loss_type == "dpo":
            # Eq. 3 https://ericmitchell.ai/cdpo.pdf; label_smoothing=0 gives original DPO (Eq. 7 of https://arxiv.org/pdf/2305.18290.pdf)
            F = torch.nn.functional
            losses = -F.logsigmoid(beta * logits) * (1 - label_smoothing) - F.logsigmoid(-beta * logits) * label_smoothing
        else:
            raise NotImplementedError("loss type {} is not implemented".format(loss_type))

        chosen_rewards = beta * (policy_chosen_logps - reference_chosen_logps).detach()
        rejected_rewards = beta * (policy_rejected_logps - reference_rejected_logps).detach()

        return losses, chosen_rewards, rejected_rewards

    def process_batch_dpo(self, batch_chosen_rejected):
        batch_chosen = batch_chosen_rejected['chosen']
        batch_rejected = batch_chosen_rejected['rejected']
        
        model_output_chosen = self.process_batch(batch_chosen)
        model_output_rejected = self.process_batch(batch_rejected)
        with torch.no_grad():
            reference_model_output_chosen = self._reference_model.process_batch(batch_chosen)
            reference_model_output_rejected = self._reference_model.process_batch(batch_rejected)
        
        chosen_policy_logprobs = None
        rejected_policy_logprobs = None
        chosen_ref_logprobs = None
        rejected_ref_logprobs = None
        for codebook_idx in range(self.cfg.num_audio_codebooks):
            si = codebook_idx * self.cfg.num_audio_tokens_per_codebook
            ei = si + self.cfg.num_audio_tokens_per_codebook
            codebook_logits_chosen = model_output_chosen['logits'][:, :, si:ei]
            codebook_logits_rejected = model_output_rejected['logits'][:, :, si:ei]

            ref_codebook_logits_chosen = reference_model_output_chosen['logits'][:, :, si:ei]
            ref_codebook_logits_rejected = reference_model_output_rejected['logits'][:, :, si:ei]

            codebook_labels_chosen = model_output_chosen['audio_codes_target'][:,codebook_idx]
            codebook_labels_rejected = model_output_rejected['audio_codes_target'][:,codebook_idx]

            codebook_log_probs_chosen = self._get_batch_logps(codebook_logits_chosen, codebook_labels_chosen, model_output_chosen['loss_mask'])
            codebook_log_probs_rejected = self._get_batch_logps(codebook_logits_rejected, codebook_labels_rejected, model_output_rejected['loss_mask'])
            with torch.no_grad():
                ref_codebook_log_probs_chosen = self._get_batch_logps(ref_codebook_logits_chosen, codebook_labels_chosen, reference_model_output_chosen['loss_mask'])
                ref_codebook_log_probs_rejected = self._get_batch_logps(ref_codebook_logits_rejected, codebook_labels_rejected, reference_model_output_rejected['loss_mask'])
            
            if chosen_policy_logprobs is None:
                chosen_policy_logprobs = codebook_log_probs_chosen
                rejected_policy_logprobs = codebook_log_probs_rejected
                chosen_ref_logprobs = ref_codebook_log_probs_chosen
                rejected_ref_logprobs = ref_codebook_log_probs_rejected
            else:
                chosen_policy_logprobs += codebook_log_probs_chosen
                rejected_policy_logprobs += codebook_log_probs_rejected
                chosen_ref_logprobs += ref_codebook_log_probs_chosen
                rejected_ref_logprobs += ref_codebook_log_probs_rejected
        
        rewards_chosen = batch_chosen['rewards']
        rewards_rejected = batch_rejected['rewards']
        
        assert torch.all(rewards_chosen == 1)
        assert torch.all(rewards_rejected < 1)

        pref_loss, chosen_rewards, rejected_rewards = self.preference_loss(
            chosen_policy_logprobs,
            rejected_policy_logprobs,
            chosen_ref_logprobs,
            rejected_ref_logprobs,
            chosen_gt_rewards=rewards_chosen,
            rejected_gt_rewards=rewards_rejected,
            beta=self.cfg.get('dpo_beta', 0.01),
            loss_type=self.cfg.get('dpo_loss_type', 'dpo'),
        )

        pref_loss = pref_loss.mean()
        sft_loss = -chosen_policy_logprobs.mean()
        
        pref_loss_weight = self.cfg.get('dpo_pref_loss_weight', 1.0)
        sft_loss_weight = self.cfg.get('dpo_sft_loss_weight', 0.0)
        loss = pref_loss_weight * pref_loss + sft_loss * sft_loss_weight

        alignment_loss = model_output_chosen['alignment_loss']
        if alignment_loss is not None:
            loss += alignment_loss
        
        return {
            'loss': loss,
            'pref_loss': pref_loss,
            'sft_loss': sft_loss,
            'alignment_loss': alignment_loss,
        }

    def training_step(self, batch, batch_idx):
        dpo_outputs = self.process_batch_dpo(batch)
        self.log('train_loss', dpo_outputs['loss'], prog_bar=True, sync_dist=True)
        self.log('train_pref_loss', dpo_outputs['pref_loss'], prog_bar=True, sync_dist=True)
        self.log('train_sft_loss', dpo_outputs['sft_loss'], prog_bar=True, sync_dist=True)
        return dpo_outputs['loss']
    
    def validation_step(self, batch, batch_idx):
        dpo_outputs = self.process_batch_dpo(batch)
        
        val_loss = dpo_outputs['loss']
        val_pref_loss = dpo_outputs['pref_loss']
        val_sft_loss = dpo_outputs['sft_loss']
        val_alignment_loss = dpo_outputs['alignment_loss']
        
        self.validation_step_outputs.append({
            'val_loss': val_loss,
            'val_pref_loss': val_pref_loss,
            'val_sft_loss': val_sft_loss,
            'val_alignment_loss': val_alignment_loss,
        })
    
    def on_validation_epoch_end(self):
        def collect(key):
            values = []
            for x in self.validation_step_outputs:
                if x[key] is not None:
                    values.append(x[key])
                else:
                    values.append(torch.tensor(0.0, device=self.device))
            stacked_values = torch.stack(values)
            return stacked_values.mean()

        val_loss = collect("val_loss")
        val_pref_loss = collect("val_pref_loss")
        val_sft_loss = collect("val_sft_loss")
        val_alignment_loss = collect("val_alignment_loss")
        self.log("val_loss", val_loss, prog_bar=True, sync_dist=True)
        self.log("val_pref_loss", val_pref_loss, prog_bar=True, sync_dist=True)
        self.log("val_sft_loss", val_sft_loss, prog_bar=True, sync_dist=True)
        if val_alignment_loss is not None:
            self.log("val_alignment_loss", val_alignment_loss, prog_bar=True, sync_dist=True)
        self.validation_step_outputs.clear()

class T5TTS_Model_OnlinePO(T5TTS_Model):
    def __init__(self, cfg: DictConfig, trainer: 'Trainer' = None):
        super().__init__(cfg, trainer)
        # Copy cfg
        ref_model_cfg = copy.deepcopy(cfg)
        with open_dict(ref_model_cfg):
            ref_model_cfg.train_ds = None
            ref_model_cfg.validation_ds = None
        self._reference_model = T5TTS_Model(cfg=ref_model_cfg)
        print("Loading reference model from checkpoint")
        self._reference_model.load_state_dict(torch.load(cfg.reference_model_ckpt_path, map_location="cpu")['state_dict'])
        self.freeze_model(self._reference_model)
        self._reference_model.eval()
        self._reference_model._no_state_dict = True
        print("Reference model loaded and frozen")

        if cfg.get('pref_set_language', "en") == "en":
            self.eval_asr_model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained(model_name="nvidia/parakeet-ctc-0.6b")
            self.eval_asr_model.freeze()
            self.eval_asr_model.eval()

        self.eval_speaker_verification_model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(model_name='titanet_large')
        self.eval_speaker_verification_model.freeze()
        self.eval_speaker_verification_model.eval()

        if cfg.get('load_whisper_model', False):
            from transformers import WhisperProcessor, WhisperForConditionalGeneration
            self.whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3")
            self.whisper_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3")
            self.whisper_model.eval()
    
    def state_dict(self, destination=None, prefix='', keep_vars=False):
        state_dict = super().state_dict(destination, prefix, keep_vars)
        keys_substrings_to_exclude = ['_speaker_verification_model', '_codec_model', '_reference_model', 
                                      'eval_asr_model',  'eval_speaker_verification_model', 'whisper_model']
        for key in list(state_dict.keys()):
            if any([substring in key for substring in keys_substrings_to_exclude]):
                del state_dict[key]
        return state_dict
        

    def _get_per_token_logps(self, logits, labels, loss_mask):
        """Compute the log probabilities of the given labels under the given logits.

        Args:
            logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
            labels: Labels for which to compute the log probabilities.
        """
        per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)
        per_token_logps = per_token_logps * loss_mask
        return per_token_logps

    def repeat_items_in_batch(self, batch, num_repeats):
        repeated_batch = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                repeated_value = value.repeat_interleave(num_repeats, dim=0)
            elif isinstance(value, list):
                repeated_value = []
                for item in value:
                    repeated_value.extend([item] * num_repeats)
            else:
                repeated_value = value
            repeated_batch[key] = repeated_value
        return repeated_batch

    def generate_and_reward(self, batch, num_generations_per_item, mode='train'):
        batch_repeated = self.repeat_items_in_batch(batch, num_generations_per_item)
        temperature = self.cfg.get('inference_temperature', 0.7)
        topk = self.cfg.get('inference_topk', 80)
        use_cfg = False
        cfg_scale = 1.0
        inference_cfg_prob = self.cfg.get('inference_cfg_prob', 0.0)
        if (inference_cfg_prob == 1.0) or (inference_cfg_prob > 0.0 and mode == 'train'):
            # Randomly set use_cfg based on the given probability
            use_cfg = random.random() < self.cfg.inference_cfg_prob
            cfg_scale = self.cfg.get('inference_cfg_scale', 1.0)
        print("use_cfg", use_cfg)
        predicted_audio, predicted_audio_lens, predicted_codes, predicted_codes_lens = self.infer_batch(
            batch_repeated,
            max_decoder_steps=self.cfg.get('max_decoder_steps', 430),
            temperature=temperature,
            topk=topk,
            use_cfg=use_cfg,
            cfg_scale=cfg_scale
        )
        predicted_audio_paths = []
        audio_durations = []
        for idx in range(predicted_audio.size(0)):
            predicted_audio_np = predicted_audio[idx].float().detach().cpu().numpy()
            predicted_audio_np = predicted_audio_np[:predicted_audio_lens[idx]]
            if predicted_audio_np.shape[0] < 1000:
                # Corner case to handle short audio files
                predicted_audio_np = np.pad(predicted_audio_np, (0, 1000 - predicted_audio_np.shape[0]))
            item_idx = idx
            # Save the predicted audio
            log_dir = self.logger.log_dir
            audio_dir = os.path.join(log_dir, 'audios')
            os.makedirs(audio_dir, exist_ok=True)
            audio_path = os.path.join(audio_dir, f'predicted_audioRank{self.global_rank}_{item_idx}.wav')
            audio_durations.append(len(predicted_audio_np) / self.cfg.sample_rate)
            sf.write(audio_path, predicted_audio_np, self.cfg.sample_rate)

            predicted_codes_torch = predicted_codes[idx].cpu().type(torch.int16)
            predicted_codes_torch = predicted_codes_torch[:, :predicted_codes_lens[idx]] # C, T
            torch.save(predicted_codes_torch, os.path.join(audio_dir, f'predicted_audioRank{self.global_rank}_{item_idx}_codes.pt'))
            predicted_audio_paths.append(audio_path)
            
        with torch.no_grad():
            if self.cfg.get("pref_set_language", "en") == "en":
                pred_transcripts = self.eval_asr_model.transcribe(predicted_audio_paths, batch_size=len(predicted_audio_paths))
                pred_transcripts = [ process_text_for_cer(transcript) for transcript in pred_transcripts ]
            else:
                pred_transcripts = []
                for audio_path in predicted_audio_paths:
                    transcript = transcribe_with_whisper(audio_path, self.cfg.pref_set_language, self.whisper_processor, self.whisper_model, self.device)
                    pred_transcripts.append(transcript)
                pred_transcripts = [process_text_for_cer(transcript) for transcript in pred_transcripts]
            
            pred_speaker_embeddings = get_speaker_embeddings_from_filepaths(predicted_audio_paths, self.eval_speaker_verification_model, self.device)
            gt_speaker_embeddings = get_speaker_embeddings_from_filepaths(batch_repeated['audio_filepaths'], self.eval_speaker_verification_model, self.device)

        batch_metrics = []
        cer_reward_weight = self.cfg.get('cer_reward_weight', 0.5)
        ssim_reward_weight = 1.0 - cer_reward_weight
        for idx in range(predicted_audio.size(0)):
            audio_path = predicted_audio_paths[idx]
            item_idx = idx
            pred_transcript = pred_transcripts[idx]
            gt_transcript = process_text_for_cer(batch_repeated['raw_texts'][idx])
            cer_gt = word_error_rate([pred_transcript], [gt_transcript], use_cer=True)
            wer_gt = word_error_rate([pred_transcript], [gt_transcript], use_cer=False)
            spk_embedding_pred = pred_speaker_embeddings[idx].cpu().numpy()
            spk_embedding_gt = gt_speaker_embeddings[idx].cpu().numpy()
            spk_similarity = np.dot(spk_embedding_pred, spk_embedding_gt) / (
                np.linalg.norm(spk_embedding_pred) * np.linalg.norm(spk_embedding_gt)
            )

            item_metrics = {
                'cer_gt': float(cer_gt),
                'wer_gt': float(wer_gt),
                'duration' : audio_durations[idx],
                'spk_similarity': float(spk_similarity),
                'pred_transcript': pred_transcript,
                'gt_transcript': gt_transcript,
                'codes_len': predicted_codes_lens[idx].item(),
            }
            with open(os.path.join(audio_dir, f'predicted_audioRank{self.global_rank}_{item_idx}_metrics.json'), 'w') as f:
                json.dump(item_metrics, f)

            batch_metrics.append(item_metrics)

        num_groups = len(batch['audio_filepaths'])
        
        best_ssim_achievable = self.cfg.get("best_ssim_achievable", 0.9) # Examples with this speaker similarity or higher will have SSIM reward of 1
        mean_cer_dataset = self.cfg.get("mean_cer_dataset", 0.1) # CER equal to this value will have reward of 0.5
        mean_ssim_dataset = self.cfg.get("mean_ssim_dataset", 0.6) # SSIM equal to this value will have reward of 0.5
        for group_idx in range(num_groups):
            group_start_idx = group_idx * num_generations_per_item
            group_end_idx = group_start_idx + num_generations_per_item
            group_rewards = []
            mean_reward = 0
            eps = 0.0001
            for idx in range(group_start_idx, group_end_idx):
                # Lower CER and higher speaker similarity is better, means high reward
                # Reward for best CER and best speaker similarity should be 1
                item_cer = batch_metrics[idx]['cer_gt']
                item_ssim = batch_metrics[idx]['spk_similarity']
                item_cer = min( max(item_cer, 0.0), 1.0)
                item_ssim = max( min(item_ssim, best_ssim_achievable), 0.0)
                if item_cer <= mean_cer_dataset:
                    cer_reward = 0.5 + 0.5 * (mean_cer_dataset - item_cer) / mean_cer_dataset # 0.5 to 1
                else:
                    cer_reward = 0.5 - 0.5 * (item_cer - mean_cer_dataset) / (1 - mean_cer_dataset) # 0 to 0.5
                if item_ssim >= mean_ssim_dataset:
                    spk_similarity_reward = 0.5 + 0.5 * (item_ssim - mean_ssim_dataset) / (best_ssim_achievable - mean_ssim_dataset)
                else:
                    spk_similarity_reward = 0.5 - 0.5 * (mean_ssim_dataset - item_ssim) / (mean_ssim_dataset)

                batch_metrics[idx]['reward'] = cer_reward * cer_reward_weight + spk_similarity_reward * ssim_reward_weight
                
                if (batch_metrics[idx]['codes_len'] >= 425) or (batch_metrics[idx]['codes_len'] <= 3): # TODO: Remove hardcoded lengths
                    # This means it did not complete the sentence or generated an extremely short sentence
                    batch_metrics[idx]['reward'] = 0.0
                print("Item idx: ", idx, " CER: ", item_cer, " SSIM: ", item_ssim, " Reward: ", batch_metrics[idx]['reward'], " Codes len: ", batch_metrics[idx]['codes_len'])
                batch_metrics[idx]['cer_reward'] = cer_reward
                batch_metrics[idx]['spk_similarity_reward'] = spk_similarity_reward
                mean_reward += batch_metrics[idx]['reward']
                group_rewards.append(batch_metrics[idx]['reward'])
            
            mean_reward /= num_generations_per_item
            std_reward = np.std(group_rewards)
            for idx in range(group_start_idx, group_end_idx):
                batch_metrics[idx]['advantage'] = (batch_metrics[idx]['reward'] - mean_reward) / (std_reward + 1e-6)


        advantages = [x['advantage'] for x in batch_metrics]
        advantages = torch.tensor(advantages, device=self.device)
        print("Mean reward: ", mean_reward)
        return {
            'mean_reward': torch.tensor(mean_reward, device=self.device),
            'batch_repeated': batch_repeated,
            'metrics': batch_metrics,
            'predicted_codes': predicted_codes,
            'predicted_codes_lens': predicted_codes_lens,
            'advantages': advantages,
        }
    
    def process_batch_online_po(self, batch, n_generations_per_item, mode='train'):
        use_kv_cache_during_online_po = self.cfg.get("use_kv_cache_during_online_po", False)
        if use_kv_cache_during_online_po:
            self.use_kv_cache_for_inference = True
            self.t5_decoder.reset_cache(use_cache=True)
        
        with torch.no_grad():
            generated_codes_and_metrics = self.generate_and_reward(batch, n_generations_per_item, mode)
            
        if use_kv_cache_during_online_po:
            self.use_kv_cache_for_inference = False
            self.t5_decoder.reset_cache(use_cache=False)

        batch_repeated = generated_codes_and_metrics['batch_repeated']
        predicted_codes = generated_codes_and_metrics['predicted_codes'] # B, 8, T
        predicted_codes_lens = generated_codes_and_metrics['predicted_codes_lens'] # B
        predicted_codes = predicted_codes[:,:,:predicted_codes_lens.max()]

        advantages = generated_codes_and_metrics['advantages'] # B
        # Add extra tokens for BOS and EOS
        bos_tensor = torch.full((predicted_codes.size(0), predicted_codes.size(1), 1), self.audio_bos_id, dtype=predicted_codes.dtype, device=predicted_codes.device)
        padding_tensor = torch.full((predicted_codes.size(0), predicted_codes.size(1), 1), 0, dtype=predicted_codes.dtype, device=predicted_codes.device)
        predicted_codes = torch.cat([bos_tensor, predicted_codes, padding_tensor], dim=2)
        for idx in range(predicted_codes.size(0)):
            predicted_codes[idx, :, predicted_codes_lens[idx]+1] = self.audio_eos_id # Accounts for BOS
        batch_repeated['audio_codes'] = predicted_codes
        batch_repeated['audio_codes_lens'] = predicted_codes_lens + 2 # Accounts for BOS and EOS
        if 'audio' in batch_repeated:
            del batch_repeated['audio']
        if 'audio_lens' in batch_repeated:
            del batch_repeated['audio_lens']
        
        policy_model_outputs = self.process_batch(batch_repeated)
        with torch.no_grad():
            reference_model_output = self._reference_model.process_batch(batch_repeated)
        
        total_loss = None
        total_kl = None
        for codebook_idx in range(self.cfg.num_audio_codebooks):
            si = codebook_idx * self.cfg.num_audio_tokens_per_codebook
            ei = si + self.cfg.num_audio_tokens_per_codebook
            codebook_logits = policy_model_outputs['logits'][:, :, si:ei] # B, T, C
            ref_codebook_logits = reference_model_output['logits'][:, :, si:ei]

            codebook_labels = batch_repeated['audio_codes'][:,codebook_idx,1:]
            per_token_codebook_log_probs = self._get_per_token_logps(codebook_logits, codebook_labels, policy_model_outputs['loss_mask'])
            with torch.no_grad():
                per_token_ref_codebook_log_probs = self._get_per_token_logps(ref_codebook_logits, codebook_labels, reference_model_output['loss_mask'])
            
            # https://github.com/huggingface/trl/blob/ffcb9f4aee725a2bd072d0387afe68a4b1c7967c/trl/trainer/grpo_trainer.py#L703
            per_token_codebook_kl = torch.exp(per_token_ref_codebook_log_probs - per_token_codebook_log_probs) - (per_token_ref_codebook_log_probs - per_token_codebook_log_probs) - 1
            per_token_loss = torch.exp(per_token_codebook_log_probs - per_token_codebook_log_probs.detach()) * advantages.unsqueeze(1)
            per_token_loss = -(per_token_loss - self.cfg.grpo_beta * per_token_codebook_kl)
            codebook_loss = ((per_token_loss * policy_model_outputs['loss_mask']).sum(dim=1) / policy_model_outputs['loss_mask'].sum(dim=1)).mean()
            codebook_kl_loss_mean = ((per_token_codebook_kl * policy_model_outputs['loss_mask']).sum(dim=1) / policy_model_outputs['loss_mask'].sum(dim=1)).mean()
            if total_loss is None:
                total_loss = codebook_loss
                total_kl = codebook_kl_loss_mean
            else:
                total_loss += codebook_loss
                total_kl += codebook_kl_loss_mean

        
        total_loss /= self.cfg.num_audio_codebooks
        print("Total kl", total_kl, n_generations_per_item)
        return {
            'mean_reward': generated_codes_and_metrics['mean_reward'],
            'loss': total_loss,
            'kl_loss': total_kl,
            'batch_metrics': generated_codes_and_metrics['metrics'],
        }

    def training_step(self, batch, batch_idx):
        n_generations_per_item = self.cfg.get('n_generations_per_item', 6)
        po_outputs = self.process_batch_online_po(batch, n_generations_per_item)
        self.log('train_loss', po_outputs['loss'], prog_bar=True, sync_dist=True)
        self.log('train_kl_loss', po_outputs['kl_loss'], prog_bar=True, sync_dist=True)
        self.log('train_mean_reward', po_outputs['mean_reward'], prog_bar=True, sync_dist=True)
        return po_outputs['loss']
    
    def validation_step(self, batch, batch_idx):
        po_outputs = self.process_batch_online_po(batch, 1, mode='val')
        batch_metrics = po_outputs['batch_metrics']
        mean_reward = po_outputs['mean_reward']
        val_loss = po_outputs['loss']
        val_kl_loss = po_outputs['kl_loss']
        
        self.validation_step_outputs.append({
            'mean_reward': mean_reward,
            'val_loss': val_loss,
            'val_kl_loss': val_kl_loss,
            'batch_metrics': batch_metrics,
        })
    
    def on_validation_epoch_end(self):
        def collect(key):
            values = []
            for x in self.validation_step_outputs:
                if x[key] is not None:
                    values.append(x[key])
                else:
                    values.append(torch.tensor(0.0, device=self.device))
            stacked_values = torch.stack(values)
            return stacked_values.mean()

        val_loss = collect("val_loss")
        val_kl_loss = collect("val_kl_loss")
        mean_reward = collect("mean_reward")

        self.log("val_loss", val_loss, prog_bar=True, sync_dist=True)
        self.log("val_kl_loss", val_kl_loss, prog_bar=True, sync_dist=True)
        self.log("val_mean_reward", mean_reward, prog_bar=True, sync_dist=True)

        mean_metrics = {}
        for val_output in self.validation_step_outputs:
            batch_metrics = val_output['batch_metrics']
            for item_metrics in batch_metrics:
                for key, value in item_metrics.items():
                    if "transcript" not in key:
                        if key not in mean_metrics:
                            mean_metrics[key] = []
                        mean_metrics[key].append(value)
        
        for key, values in mean_metrics.items():
            mean_metrics[key] = np.mean(values)
            self.log(f"val_{key}", mean_metrics[key], prog_bar=True, sync_dist=True)

        self.validation_step_outputs.clear()



# Utility functions
def process_text_for_cer(input_text):
    """
    Normalizes text for CER/WER calculation.
    Taken from hallucination_eval.py
    """
    # Convert text to lowercase
    lower_case_text = input_text.lower()
    
    # Remove commas from text
    no_comma_text = lower_case_text.replace(",", "")
    # Replace "-" with spaces
    no_dash_text = no_comma_text.replace("-", " ")
    no_dash_text = no_dash_text.replace("'", "")
    no_dash_text = no_dash_text.replace(";", "")
    no_dash_text = no_dash_text.replace(".", "")
    
    # Replace double spaces with single space
    single_space_text = " ".join(no_dash_text.split())

    single_space_text = single_space_text.translate(str.maketrans('', '', string.punctuation))

    # @shehzeen: Added this to handle some common errors in ASR transcripts
    single_space_text.replace("h t t p", "http")
    single_space_text.replace("w w w", "www")

    return single_space_text

def get_speaker_embeddings_from_filepaths(filepaths, speaker_verification_model, device):
        audio_batch = []
        audio_lengths = []
        for filepath in filepaths:
            audio, sr = sf.read(filepath)
            if sr != 16000:
                audio = librosa.core.resample(audio, orig_sr=sr, target_sr=16000)
            audio_tensor = torch.tensor(audio, dtype=torch.float32, device=device)
            audio_batch.append(audio_tensor)
            audio_lengths.append(audio_tensor.size(0))
        
        batch_audio_lens = torch.tensor(audio_lengths, device=device).long()
        max_audio_len = int(batch_audio_lens.max().item())
        audio_batch = stack_tensors(audio_batch, max_lens=[max_audio_len])

        _, speaker_embeddings = speaker_verification_model.forward(
            input_signal=audio_batch, 
            input_signal_length=batch_audio_lens
        )
        
        return speaker_embeddings

def transcribe_with_whisper(audio_filepath, language, whisper_processor, whisper_model, device):
    speech_array, sampling_rate = librosa.load(audio_filepath, sr=16000)
    forced_decoder_ids = whisper_processor.get_decoder_prompt_ids(language=language) if language else None
    inputs = whisper_processor(speech_array, sampling_rate=sampling_rate, return_tensors="pt").input_features
    inputs = inputs.to(device)
    with torch.no_grad():
        predicted_ids = whisper_model.generate(inputs, forced_decoder_ids=forced_decoder_ids)
    transcription = whisper_processor.batch_decode(predicted_ids, skip_special_tokens=True)
    result = transcription[0]
    return result