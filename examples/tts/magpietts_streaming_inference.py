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
"""
MagpieTTS Streaming Inference Test Script.

This script tests the streaming TTS inference functionality, supporting both
single sample (batch_size=1) and batched inference (batch_size>1).

For batched inference, each item in the batch can have different context lengths
and be in different processing phases (context, prompt, phoneme-only, audio).

Example usage:
    # Single sample inference from checkpoint
    python examples/tts/magpietts_streaming_inference.py \
        --hparams_file /path/to/hparams.yaml \
        --checkpoint_file /path/to/model.ckpt \
        --codecmodel_path /path/to/codec.nemo \
        --context_audio /path/to/context.wav \
        --text "Hello, this is a test of streaming TTS inference." \
        --output_path /path/to/output.wav

    # Batched inference with multiple context audios
    python examples/tts/magpietts_streaming_inference.py \
        --nemo_file /path/to/model.nemo \
        --codecmodel_path /path/to/codec.nemo \
        --context_audio /path/to/context1.wav /path/to/context2.wav \
        --context_duration 3.0 5.0 \
        --text "First text to synthesize." "Second text to synthesize." \
        --output_path /path/to/output.wav
"""
from __future__ import annotations

import argparse
import os
import time
from typing import Optional

import numpy as np
import soundfile as sf
import torch
from omegaconf import OmegaConf, open_dict

from nemo.collections.tts.models import EasyMagpieTTSModel
from nemo.utils import logging


def load_model(
    hparams_file: Optional[str],
    checkpoint_file: Optional[str],
    nemo_file: Optional[str],
    codecmodel_path: str,
    device: str = "cuda",
) -> EasyMagpieTTSModel:
    """
    Load an EasyMagpieTTSModel from checkpoint or .nemo file.

    Args:
        hparams_file: Path to hparams.yaml (required with checkpoint_file).
        checkpoint_file: Path to .ckpt file (required with hparams_file).
        nemo_file: Path to .nemo file (alternative to hparams + checkpoint).
        codecmodel_path: Path to the audio codec model.
        device: Device to load model on.

    Returns:
        Loaded model ready for inference.
    """
    if hparams_file is not None and checkpoint_file is not None:
        # Load from hparams + checkpoint
        logging.info(f"Loading model from checkpoint: {checkpoint_file}")
        model_cfg = OmegaConf.load(hparams_file)

        # Handle different config structures
        if "cfg" in model_cfg:
            model_cfg = model_cfg.cfg

        with open_dict(model_cfg):
            # Override codec model path
            model_cfg.codecmodel_path = codecmodel_path

            # Disable training datasets
            model_cfg.train_ds = None
            model_cfg.validation_ds = None

        model = EasyMagpieTTSModel(cfg=model_cfg)

        # Load weights
        ckpt = torch.load(checkpoint_file, weights_only=False)
        state_dict = ckpt['state_dict']
        model.load_state_dict(state_dict)

    elif nemo_file is not None:
        # Load from .nemo file
        logging.info(f"Loading model from NeMo archive: {nemo_file}")
        model_cfg = EasyMagpieTTSModel.restore_from(nemo_file, return_config=True)

        with open_dict(model_cfg):
            model_cfg.codecmodel_path = codecmodel_path
            model_cfg.train_ds = None
            model_cfg.validation_ds = None

        model = EasyMagpieTTSModel.restore_from(nemo_file, override_config_path=model_cfg)

    else:
        raise ValueError("Must provide either (hparams_file + checkpoint_file) or nemo_file")

    model.to(device)
    model.eval()
    logging.info("Model loaded and ready for streaming inference.")

    return model


def load_audio(audio_path: str, target_sample_rate: int) -> torch.Tensor:
    """
    Load audio file and resample if needed.

    Args:
        audio_path: Path to audio file.
        target_sample_rate: Target sample rate.

    Returns:
        Audio tensor of shape (1, num_samples).
    """
    audio, sr = sf.read(audio_path, dtype='float32')

    # Convert to mono if stereo
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)

    # Resample if needed
    if sr != target_sample_rate:
        import librosa

        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sample_rate)

    return torch.from_numpy(audio).unsqueeze(0)  # (1, num_samples)


def adjust_audio_to_duration(
    audio: torch.Tensor,
    sample_rate: int,
    target_duration: float,
    codec_model_samples_per_frame: int,
) -> torch.Tensor:
    """
    Adjust audio to target_duration seconds, aligned to codec frame boundaries.

    The target number of samples is calculated to align with codec frame boundaries:
    1. Convert target_duration to number of codec frames
    2. Convert codec frames back to samples

    If audio is longer than target, take the first target_duration seconds.
    If audio is shorter, repeat it until it reaches target_duration seconds.

    Args:
        audio: Audio tensor of shape (1, num_samples).
        sample_rate: Sample rate of the audio.
        target_duration: Target duration in seconds.
        codec_model_samples_per_frame: Number of audio samples per codec frame
            (codec downsampling factor).

    Returns:
        Audio tensor of shape (1, target_num_samples) where target_num_samples
        is aligned to codec frame boundaries.
    """
    # Calculate target samples aligned to codec frame boundaries
    # Same logic as text_to_speech_dataset.py
    num_codec_frames = int(target_duration * sample_rate / codec_model_samples_per_frame)
    target_num_samples = num_codec_frames * codec_model_samples_per_frame
    current_num_samples = audio.size(1)

    if current_num_samples >= target_num_samples:
        # Audio is longer than target - take the first target_duration seconds
        audio = audio[:, :target_num_samples]
    else:
        # Audio is shorter - repeat until we have enough samples
        num_repeats = int(np.ceil(target_num_samples / current_num_samples))
        audio_repeated = audio.repeat(1, num_repeats)
        audio = audio_repeated[:, :target_num_samples]

    return audio


def run_streaming_inference(
    model: EasyMagpieTTSModel,
    context_audio: torch.Tensor,
    context_audio_lens: torch.Tensor,
    context_text: str,
    text: str,
    phoneme_text: Optional[str] = None,
    use_gt_phonemes: bool = False,
    inference_mode: Optional[str] = None,
    use_cfg: bool = False,
    cfg_scale: float = 1.5,
    use_local_transformer: bool = False,
    temperature: float = 0.7,
    topk: int = 80,
    max_steps: int = 500,
    verbose: bool = True,
    force_dropout_text: bool = False,
) -> tuple:
    """
    Run streaming TTS inference.

    Args:
        model: The loaded EasyMagpieTTSModel.
        context_audio: Context audio tensor (1, num_samples).
        context_audio_lens: Length of context audio (1,).
        context_text: Context text for speaker conditioning.
        text: Main text to synthesize.
        phoneme_text: Optional phoneme text for GT conditioning. If None, uses text.
        use_gt_phonemes: If True, use GT phonemes as decoder input (teacher forcing).
        inference_mode: Inference mode name (e.g., "streaming_4_8").
        use_cfg: Whether to use classifier-free guidance.
        cfg_scale: CFG scale factor.
        use_local_transformer: Whether to use local transformer.
        temperature: Sampling temperature.
        topk: Top-k sampling parameter.
        max_steps: Maximum generation steps.
        verbose: Whether to print progress.

    Returns:
        Tuple of (output, timing_info, context_audio_decoded, context_audio_decoded_lens).
        output is StreamingFinalizeOutput with audio, codes, and phoneme predictions.
        context_audio_decoded is the decoded context audio from the model's internal codes (for sanity checking).
    """
    device = next(model.parameters()).device

    # Encode context audio to codes
    context_audio = context_audio.to(device)
    context_audio_lens = context_audio_lens.to(device)

    with torch.inference_mode():
        context_audio_codes, context_audio_codes_lens = model.audio_to_codes(context_audio, context_audio_lens)

    # Tokenize context text
    # Use the text conditioning tokenizer
    tokenizer_name = model.text_conditioning_tokenizer_name
    context_text_tokens = model.tokenizer.encode(context_text, tokenizer_name=tokenizer_name)
    context_text_tokens = torch.tensor([context_text_tokens], dtype=torch.long, device=device)
    context_text_tokens_lens = torch.tensor([context_text_tokens.size(1)], dtype=torch.long, device=device)

    # Tokenize main text
    # Get the appropriate tokenizer name for main text
    if hasattr(model.tokenizer, 'tokenizers') and 'english_phoneme' in model.tokenizer.tokenizers:
        main_tokenizer_name = 'english_phoneme'
    else:
        main_tokenizer_name = tokenizer_name

    text_tokens = model.tokenizer.encode(text, tokenizer_name=main_tokenizer_name)
    text_tokens = text_tokens + [model.eos_id]
    text_tokens = torch.tensor(text_tokens, dtype=torch.long, device=device)

    # Tokenize phoneme text if provided (for GT phoneme conditioning)
    gt_phoneme_tokens = None
    gt_phoneme_tokens_lens = None
    if model.phoneme_tokenizer is not None:
        phoneme_source = phoneme_text if phoneme_text is not None else text
        phoneme_tokens_list = model.phoneme_tokenizer.encode(phoneme_source)
        # Add BOS and EOS
        bos_id = model.phoneme_tokenizer.bos_token_id
        eos_id = model.phoneme_tokenizer.eos_token_id
        phoneme_tokens_list = [bos_id] + phoneme_tokens_list + [eos_id]
        gt_phoneme_tokens = torch.tensor([phoneme_tokens_list], dtype=torch.long, device=device)
        gt_phoneme_tokens_lens = torch.tensor([len(phoneme_tokens_list)], dtype=torch.long, device=device)

    phoneme_input_type = 'gt' if use_gt_phonemes else 'pred'

    # Get streaming delays for logging
    mode_name = inference_mode or model.default_inference_mode
    training_mode = model.mode_name_to_mode.get(mode_name, model.training_modes[0])
    phoneme_delay = training_mode.streaming_phonemes_delay
    speech_delay = training_mode.streaming_speech_delay

    if verbose:
        logging.info(f"Context audio codes shape: {context_audio_codes.shape}")
        logging.info(f"Context text tokens: {context_text_tokens.shape}")
        logging.info(f"Main text tokens: {text_tokens.shape} ({len(text_tokens)} tokens)")
        if gt_phoneme_tokens is not None:
            logging.info(f"GT phoneme tokens: {gt_phoneme_tokens.shape} ({gt_phoneme_tokens_lens[0].item()} tokens)")
        logging.info(f"Phoneme input type: {phoneme_input_type}")
        logging.info(f"Using inference mode: {mode_name}")
        logging.info(f"Phoneme delay: {phoneme_delay}, Speech delay: {speech_delay}")
        logging.info("Phases: Prompt (0 to phoneme_delay) -> Phoneme-only (phoneme_delay to speech_delay) -> Audio")

    # Initialize streaming state
    start_time = time.time()

    state = model.streaming_init(
        context_audio_codes=context_audio_codes,
        context_audio_codes_lens=context_audio_codes_lens,
        context_text_tokens=context_text_tokens,
        context_text_tokens_lens=context_text_tokens_lens,
        inference_mode=inference_mode,
        use_cfg=use_cfg,
        cfg_scale=cfg_scale,
        use_local_transformer=use_local_transformer,
        temperature=temperature,
        topk=topk,
        phoneme_input_type=phoneme_input_type,
        gt_phoneme_tokens=gt_phoneme_tokens,
        gt_phoneme_tokens_lens=gt_phoneme_tokens_lens,
    )

    init_time = time.time() - start_time
    if verbose:
        logging.info(f"Streaming init completed in {init_time:.3f}s")

    # Decode and return context audio for sanity check
    # The context_audio_codes in state have special tokens and are stacked
    # We need to remove special tokens and decode them
    with torch.inference_mode():
        ctx_codes = state.context_audio_codes.clone()
        ctx_codes_lens = state.context_audio_codes_lens.clone()
        # Remove special tokens (BOS and EOS)
        ctx_codes, ctx_codes_lens = model.remove_special_tokens(
            codes=ctx_codes,
            codes_len=ctx_codes_lens,
        )
        # codes_to_audio will handle unstacking internally
        context_audio_decoded, context_audio_decoded_lens, _ = model.codes_to_audio(ctx_codes, ctx_codes_lens)

    # Feed text tokens one at a time
    generation_start = time.time()
    num_audio_frames = 0
    num_phoneme_frames = 0
    prompt_phase_tokens = 0
    phoneme_only_phase_tokens = 0

    for i, token in enumerate(text_tokens):
        state, audio_codes, phoneme_tokens = model.streaming_step(
            state, text_tokens=token.unsqueeze(0), force_dropout_text=force_dropout_text
        )

        # Track which phase we're in
        if audio_codes is None and phoneme_tokens is None:
            prompt_phase_tokens += 1
        elif audio_codes is None and phoneme_tokens is not None:
            phoneme_only_phase_tokens += 1
            num_phoneme_frames += 1
        else:
            if audio_codes is not None:
                num_audio_frames += 1
            if phoneme_tokens is not None:
                num_phoneme_frames += 1

        if verbose and (i + 1) % 10 == 0:
            phase = (
                "prompt"
                if audio_codes is None and phoneme_tokens is None
                else ("phoneme-only" if audio_codes is None else "audio")
            )
            logging.info(
                f"Processed {i + 1}/{len(text_tokens)} text tokens (phase: {phase}), "
                f"audio frames: {num_audio_frames}, phoneme frames: {num_phoneme_frames}"
            )

        if state.finished:
            if verbose:
                logging.info(f"EOS detected at text token {i + 1}")
            break

    # Continue generating until finished (text has ended)
    continuation_steps = 0
    while not state.finished and continuation_steps < max_steps:
        state, audio_codes, phoneme_tokens = model.streaming_step(
            state, text_tokens=None, force_dropout_text=force_dropout_text
        )

        if audio_codes is not None:
            num_audio_frames += 1
        if phoneme_tokens is not None:
            num_phoneme_frames += 1

        continuation_steps += 1

        if verbose and continuation_steps % 20 == 0:
            logging.info(
                f"Continuation step {continuation_steps}, "
                f"audio frames: {num_audio_frames}, phoneme frames: {num_phoneme_frames}"
            )

    generation_time = time.time() - generation_start

    if verbose:
        logging.info(f"Generation completed in {generation_time:.3f}s")
        logging.info(f"Prompt phase tokens: {prompt_phase_tokens}")
        logging.info(f"Phoneme-only phase tokens: {phoneme_only_phase_tokens}")
        logging.info(f"Audio frames generated: {num_audio_frames}")
        logging.info(f"Phoneme frames generated: {num_phoneme_frames}")
        logging.info(f"Continuation steps: {continuation_steps}")

    # Finalize and get complete audio
    output = model.streaming_finalize(state)

    total_time = time.time() - start_time

    if verbose and output.phoneme_text:
        logging.info(f"Predicted phoneme text: {output.phoneme_text[0]}")

    timing_info = {
        'init_time': init_time,
        'generation_time': generation_time,
        'total_time': total_time,
        'num_text_tokens': len(text_tokens),
        'prompt_phase_tokens': prompt_phase_tokens,
        'phoneme_only_phase_tokens': phoneme_only_phase_tokens,
        'num_audio_frames': num_audio_frames,
        'num_phoneme_frames': num_phoneme_frames,
        'continuation_steps': continuation_steps,
    }

    return output, timing_info, context_audio_decoded, context_audio_decoded_lens


def run_batched_streaming_inference(
    model: EasyMagpieTTSModel,
    context_audios: list[torch.Tensor],
    context_audio_lens_list: list[torch.Tensor],
    context_texts: list[str],
    texts: list[str],
    phoneme_texts: Optional[list[str]] = None,
    use_gt_phonemes: bool = False,
    inference_mode: Optional[str] = None,
    use_cfg: bool = False,
    cfg_scale: float = 1.5,
    use_local_transformer: bool = False,
    temperature: float = 0.7,
    topk: int = 80,
    max_steps: int = 500,
    verbose: bool = True,
    force_dropout_text: bool = False,
) -> tuple:
    """
    Run batched streaming TTS inference.

    Each batch item can have different context lengths. The streaming processes
    only the minimum context length initially, then continues processing remaining
    context per-item in the "context phase" before moving to prompt/audio phases.

    Args:
        model: The loaded EasyMagpieTTSModel.
        context_audios: List of context audio tensors, each (1, num_samples).
        context_audio_lens_list: List of context audio lengths, each (1,).
        context_texts: List of context texts for speaker conditioning.
        texts: List of main texts to synthesize.
        phoneme_texts: Optional list of phoneme texts for GT conditioning. If None, uses texts.
        use_gt_phonemes: If True, use GT phonemes as decoder input (teacher forcing).
        inference_mode: Inference mode name (e.g., "streaming_4_8").
        use_cfg: Whether to use classifier-free guidance.
        cfg_scale: CFG scale factor.
        use_local_transformer: Whether to use local transformer.
        temperature: Sampling temperature.
        topk: Top-k sampling parameter.
        max_steps: Maximum generation steps.
        verbose: Whether to print progress.

    Returns:
        Tuple of (output, timing_info) where output is StreamingFinalizeOutput.
    """
    device = next(model.parameters()).device
    batch_size = len(context_audios)

    assert len(context_texts) == batch_size, "Number of context texts must match batch size"
    assert len(texts) == batch_size, "Number of texts must match batch size"

    # Encode context audio to codes for each item
    context_audio_codes_list = []
    context_audio_codes_lens_list = []

    with torch.inference_mode():
        for i in range(batch_size):
            context_audio = context_audios[i].to(device)
            context_audio_lens = context_audio_lens_list[i].to(device)
            codes, codes_lens = model.audio_to_codes(context_audio, context_audio_lens)
            context_audio_codes_list.append(codes)
            context_audio_codes_lens_list.append(codes_lens)

    # Pad and batch context audio codes
    max_context_len = max(c.size(-1) for c in context_audio_codes_list)
    num_codebooks = context_audio_codes_list[0].size(1)

    context_audio_codes = torch.zeros(batch_size, num_codebooks, max_context_len, dtype=torch.long, device=device)
    context_audio_codes_lens = torch.zeros(batch_size, dtype=torch.long, device=device)

    for i in range(batch_size):
        codes = context_audio_codes_list[i]
        codes_len = context_audio_codes_lens_list[i]
        context_audio_codes[i, :, : codes.size(-1)] = codes[0]
        context_audio_codes_lens[i] = codes_len[0]

    # Tokenize context texts
    tokenizer_name = model.text_conditioning_tokenizer_name
    context_text_tokens_list = []
    for ctx_text in context_texts:
        tokens = model.tokenizer.encode(ctx_text, tokenizer_name=tokenizer_name)
        context_text_tokens_list.append(tokens)

    # Pad and batch context text tokens
    max_context_text_len = max(len(t) for t in context_text_tokens_list)
    context_text_tokens = torch.zeros(batch_size, max_context_text_len, dtype=torch.long, device=device)
    context_text_tokens_lens = torch.zeros(batch_size, dtype=torch.long, device=device)

    for i, tokens in enumerate(context_text_tokens_list):
        context_text_tokens[i, : len(tokens)] = torch.tensor(tokens, dtype=torch.long, device=device)
        context_text_tokens_lens[i] = len(tokens)

    # Tokenize main texts
    if hasattr(model.tokenizer, 'tokenizers') and 'english_phoneme' in model.tokenizer.tokenizers:
        main_tokenizer_name = 'english_phoneme'
    else:
        main_tokenizer_name = tokenizer_name

    text_tokens_list = []
    for text in texts:
        tokens = model.tokenizer.encode(text, tokenizer_name=main_tokenizer_name)
        tokens = tokens + [model.eos_id]
        text_tokens_list.append(torch.tensor(tokens, dtype=torch.long, device=device))

    max_text_len = max(len(t) for t in text_tokens_list)

    # Tokenize phoneme texts if model has phoneme tokenizer
    gt_phoneme_tokens = None
    gt_phoneme_tokens_lens = None
    if model.phoneme_tokenizer is not None:
        phoneme_sources = phoneme_texts if phoneme_texts is not None else texts
        bos_id = model.phoneme_tokenizer.bos_token_id
        eos_id = model.phoneme_tokenizer.eos_token_id
        phoneme_tokens_lists = []
        for ptext in phoneme_sources:
            tokens = model.phoneme_tokenizer.encode(ptext)
            tokens = [bos_id] + tokens + [eos_id]
            phoneme_tokens_lists.append(tokens)
        max_phoneme_len = max(len(t) for t in phoneme_tokens_lists)
        gt_phoneme_tokens = torch.zeros(batch_size, max_phoneme_len, dtype=torch.long, device=device)
        gt_phoneme_tokens_lens = torch.zeros(batch_size, dtype=torch.long, device=device)
        for i, tokens in enumerate(phoneme_tokens_lists):
            gt_phoneme_tokens[i, : len(tokens)] = torch.tensor(tokens, dtype=torch.long, device=device)
            gt_phoneme_tokens_lens[i] = len(tokens)

    phoneme_input_type = 'gt' if use_gt_phonemes else 'pred'

    # Get streaming delays for logging
    mode_name = inference_mode or model.default_inference_mode
    training_mode = model.mode_name_to_mode.get(mode_name, model.training_modes[0])
    phoneme_delay = training_mode.streaming_phonemes_delay
    speech_delay = training_mode.streaming_speech_delay

    if verbose:
        logging.info(f"Batch size: {batch_size}")
        logging.info(f"Context audio codes shape: {context_audio_codes.shape}")
        logging.info(f"Context audio codes lens: {context_audio_codes_lens.tolist()}")
        logging.info(f"Context text tokens shape: {context_text_tokens.shape}")
        logging.info(f"Context text tokens lens: {context_text_tokens_lens.tolist()}")
        logging.info(f"Max text tokens: {max_text_len}")
        logging.info(f"Text tokens per item: {[len(t) for t in text_tokens_list]}")
        if gt_phoneme_tokens is not None:
            logging.info(f"GT phoneme tokens shape: {gt_phoneme_tokens.shape}")
            logging.info(f"GT phoneme tokens lens: {gt_phoneme_tokens_lens.tolist()}")
        logging.info(f"Phoneme input type: {phoneme_input_type}")
        logging.info(f"Using inference mode: {mode_name}")
        logging.info(f"Phoneme delay: {phoneme_delay}, Speech delay: {speech_delay}")

    # Initialize streaming state
    start_time = time.time()

    state = model.streaming_init(
        context_audio_codes=context_audio_codes,
        context_audio_codes_lens=context_audio_codes_lens,
        context_text_tokens=context_text_tokens,
        context_text_tokens_lens=context_text_tokens_lens,
        inference_mode=inference_mode,
        use_cfg=use_cfg,
        cfg_scale=cfg_scale,
        use_local_transformer=use_local_transformer,
        temperature=temperature,
        topk=topk,
        phoneme_input_type=phoneme_input_type,
        gt_phoneme_tokens=gt_phoneme_tokens,
        gt_phoneme_tokens_lens=gt_phoneme_tokens_lens,
    )

    init_time = time.time() - start_time
    if verbose:
        logging.info(f"Streaming init completed in {init_time:.3f}s")
        logging.info(f"Initial context_position: {state.context_position.tolist()}")
        logging.info(f"Full context lens: {state.full_context_lens.tolist()}")

    # Feed text tokens one at a time
    generation_start = time.time()
    step_count = 0
    num_audio_frames = 0

    # Track which items have finished their text
    text_positions = torch.zeros(batch_size, dtype=torch.long, device=device)
    text_finished_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)

    # Main streaming loop
    while not state.finished.all() and step_count < max_steps + max_text_len:
        # Determine which items are in context phase
        in_context_phase = state.context_position < state.full_context_lens

        # Prepare text tokens for this step
        # Items in context phase: use 0 (will be ignored)
        # Items not in context phase: use their next text token or 0 if text finished
        text_tokens_batch = torch.zeros(batch_size, dtype=torch.long, device=device)

        for i in range(batch_size):
            if not in_context_phase[i] and not text_finished_mask[i]:
                if text_positions[i] < len(text_tokens_list[i]):
                    text_tokens_batch[i] = text_tokens_list[i][text_positions[i]]
                    text_positions[i] += 1
                else:
                    text_finished_mask[i] = True

        # Determine if we should pass None (all items have finished text and exited context)
        all_text_done = text_finished_mask.all() and not in_context_phase.any()

        if all_text_done:
            state, audio_codes, phoneme_tokens = model.streaming_step(
                state, text_tokens=None, force_dropout_text=force_dropout_text
            )
        else:
            state, audio_codes, phoneme_tokens = model.streaming_step(
                state, text_tokens=text_tokens_batch, force_dropout_text=force_dropout_text
            )

        if audio_codes is not None:
            num_audio_frames += 1

        step_count += 1

        if verbose and step_count % 20 == 0:
            in_ctx = state.context_position < state.full_context_lens
            logging.info(
                f"Step {step_count}: "
                f"in_context_phase={in_ctx.tolist()}, "
                f"text_positions={text_positions.tolist()}, "
                f"audio_frames={num_audio_frames}, "
                f"finished={state.finished.tolist()}"
            )

    generation_time = time.time() - generation_start

    if verbose:
        logging.info(f"Generation completed in {generation_time:.3f}s")
        logging.info(f"Total steps: {step_count}")
        logging.info(f"Audio frames generated: {num_audio_frames}")

    # Finalize and get complete audio
    output = model.streaming_finalize(state)

    total_time = time.time() - start_time

    if verbose and output.phoneme_text:
        for i, ptext in enumerate(output.phoneme_text):
            logging.info(f"Predicted phoneme text [{i}]: {ptext}")

    timing_info = {
        'init_time': init_time,
        'generation_time': generation_time,
        'total_time': total_time,
        'num_text_tokens': [len(t) for t in text_tokens_list],
        'num_audio_frames': num_audio_frames,
        'total_steps': step_count,
    }

    return output, timing_info


def main():
    parser = argparse.ArgumentParser(
        description="MagpieTTS Streaming Inference Test Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Model loading arguments
    model_group = parser.add_argument_group('Model Loading')
    model_group.add_argument(
        '--hparams_file',
        type=str,
        default=None,
        help='Path to hparams.yaml file',
    )
    model_group.add_argument(
        '--checkpoint_file',
        type=str,
        default=None,
        help='Path to .ckpt checkpoint file',
    )
    model_group.add_argument(
        '--nemo_file',
        type=str,
        default=None,
        help='Path to .nemo model file',
    )
    model_group.add_argument(
        '--codecmodel_path',
        type=str,
        required=True,
        help='Path to audio codec model (.nemo)',
    )

    # Input arguments
    input_group = parser.add_argument_group('Input')
    input_group.add_argument(
        '--context_audio',
        type=str,
        nargs='+',
        required=True,
        help='Path(s) to context audio file(s) for speaker cloning. ' 'Multiple files enable batched inference.',
    )
    input_group.add_argument(
        '--context_text',
        type=str,
        nargs='+',
        default=["[NO TEXT CONTEXT]"],
        help='Context text(s) for speaker conditioning. Provide one per context audio, '
        'or a single value to use for all. (default: "[NO TEXT CONTEXT]")',
    )
    input_group.add_argument(
        '--context_duration',
        type=float,
        nargs='+',
        default=[5.0],
        help='Target duration(s) for context audio in seconds. Provide one per context audio, '
        'or a single value to use for all. If audio is longer, '
        'first N seconds are used. If shorter, audio is repeated. (default: 5.0)',
    )
    input_group.add_argument(
        '--text',
        type=str,
        nargs='+',
        required=True,
        help='Text(s) to synthesize. Provide one per context audio for batched inference.',
    )
    input_group.add_argument(
        '--phoneme_text',
        type=str,
        nargs='+',
        default=None,
        help='Phoneme text(s) for GT phoneme conditioning. If not provided, uses --text. '
        'Provide one per context audio for batched inference.',
    )
    input_group.add_argument(
        '--use_gt_phonemes',
        action='store_true',
        help='Use ground-truth phonemes as decoder input (teacher forcing). '
        'If not set, uses model-predicted phonemes.',
    )

    # Output arguments
    output_group = parser.add_argument_group('Output')
    output_group.add_argument(
        '--output_path',
        type=str,
        default='streaming_output.wav',
        help='Path for output audio file',
    )

    # Inference arguments
    infer_group = parser.add_argument_group('Inference Parameters')
    infer_group.add_argument(
        '--inference_mode',
        type=str,
        default=None,
        help='Inference mode name (e.g., "streaming_4_8"). Uses model default if not specified.',
    )
    infer_group.add_argument(
        '--use_cfg',
        action='store_true',
        help='Enable classifier-free guidance',
    )
    infer_group.add_argument(
        '--cfg_scale',
        type=float,
        default=1.5,
        help='CFG scale factor (higher = stronger conditioning)',
    )
    infer_group.add_argument(
        '--use_local_transformer',
        action='store_true',
        help='Use local transformer for inference',
    )
    infer_group.add_argument(
        '--temperature',
        type=float,
        default=0.7,
        help='Sampling temperature',
    )
    infer_group.add_argument(
        '--topk',
        type=int,
        default=80,
        help='Top-k sampling parameter',
    )
    infer_group.add_argument(
        '--max_steps',
        type=int,
        default=500,
        help='Maximum generation steps after text ends',
    )
    infer_group.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device to run inference on',
    )
    infer_group.add_argument(
        '--verbose',
        action='store_true',
        help='Print detailed progress information',
    )
    infer_group.add_argument(
        '--force_dropout_text',
        action='store_true',
        help='Force dropout of text embeddings (pass zeros) to test phoneme-only inference',
    )

    args = parser.parse_args()

    # Validate arguments
    has_ckpt_mode = args.hparams_file is not None and args.checkpoint_file is not None
    has_nemo_mode = args.nemo_file is not None

    if not (has_ckpt_mode or has_nemo_mode):
        parser.error("Must provide either (--hparams_file and --checkpoint_file) or --nemo_file")

    # Load model
    model = load_model(
        hparams_file=args.hparams_file,
        checkpoint_file=args.checkpoint_file,
        nemo_file=args.nemo_file,
        codecmodel_path=args.codecmodel_path,
        device=args.device,
    )

    model = model.float()

    # Determine batch size from number of context audios
    batch_size = len(args.context_audio)

    # Expand context_text, context_duration, and text to match batch_size
    context_texts = args.context_text
    if len(context_texts) == 1 and batch_size > 1:
        context_texts = context_texts * batch_size
    elif len(context_texts) != batch_size:
        parser.error(
            f"Number of context_texts ({len(context_texts)}) must match number of context_audios ({batch_size}) or be 1"
        )

    context_durations = args.context_duration
    if len(context_durations) == 1 and batch_size > 1:
        context_durations = context_durations * batch_size
    elif len(context_durations) != batch_size:
        parser.error(
            f"Number of context_durations ({len(context_durations)}) must match number of context_audios ({batch_size}) or be 1"
        )

    texts = args.text
    if len(texts) == 1 and batch_size > 1:
        texts = texts * batch_size
    elif len(texts) != batch_size:
        parser.error(f"Number of texts ({len(texts)}) must match number of context_audios ({batch_size}) or be 1")

    # Handle phoneme_text - default to text if not provided
    phoneme_texts = args.phoneme_text
    if phoneme_texts is None:
        phoneme_texts = texts
    elif len(phoneme_texts) == 1 and batch_size > 1:
        phoneme_texts = phoneme_texts * batch_size
    elif len(phoneme_texts) != batch_size:
        parser.error(
            f"Number of phoneme_texts ({len(phoneme_texts)}) must match number of context_audios ({batch_size}) or be 1"
        )

    # Load and process context audios
    context_audios = []
    context_audio_lens_list = []

    for i, (audio_path, duration) in enumerate(zip(args.context_audio, context_durations)):
        logging.info(f"Loading context audio {i+1}/{batch_size} from: {audio_path}")
        audio = load_audio(audio_path, model.sample_rate)
        original_duration = audio.size(1) / model.sample_rate
        logging.info(f"  Original duration: {original_duration:.2f}s")

        # Adjust to target duration (aligned to codec frame boundaries)
        audio = adjust_audio_to_duration(audio, model.sample_rate, duration, model.codec_model_samples_per_frame)
        adjusted_duration = audio.size(1) / model.sample_rate
        logging.info(f"  Adjusted duration: {adjusted_duration:.2f}s (target: {duration}s, codec-aligned)")

        context_audios.append(audio)
        context_audio_lens_list.append(torch.tensor([audio.size(1)], dtype=torch.long))

    logging.info(f"\nBatch size: {batch_size}")
    logging.info(f"Context texts: {context_texts}")
    logging.info(f"Texts to synthesize: {texts}")
    logging.info(f"Phoneme texts: {phoneme_texts}")
    logging.info(f"Use GT phonemes: {args.use_gt_phonemes}")

    # Use single-sample or batched inference
    if batch_size == 1:
        logging.info("\n=== Running single-sample streaming inference ===")
        output, timing_info, context_audio_decoded, context_audio_decoded_lens = run_streaming_inference(
            model=model,
            context_audio=context_audios[0],
            context_audio_lens=context_audio_lens_list[0],
            context_text=context_texts[0],
            text=texts[0],
            phoneme_text=phoneme_texts[0],
            use_gt_phonemes=args.use_gt_phonemes,
            inference_mode=args.inference_mode,
            use_cfg=args.use_cfg,
            cfg_scale=args.cfg_scale,
            use_local_transformer=args.use_local_transformer,
            temperature=args.temperature,
            topk=args.topk,
            max_steps=args.max_steps,
            verbose=args.verbose,
            force_dropout_text=args.force_dropout_text,
        )

        # Save output
        output_dir = os.path.dirname(args.output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        audio_np = output.audio[0, : output.audio_len[0].item()].cpu().numpy()
        sf.write(args.output_path, audio_np, model.output_sample_rate)
        logging.info(f"Output saved to: {args.output_path}")

        # Save decoded context audio for sanity check
        output_base, output_ext = os.path.splitext(args.output_path)
        context_output_path = f"{output_base}_context_decoded{output_ext}"
        context_audio_np = context_audio_decoded[0, : context_audio_decoded_lens[0].item()].cpu().numpy()
        sf.write(context_output_path, context_audio_np, model.output_sample_rate)

        logging.info(f"Context audio (decoded from codes) saved to: {context_output_path}")
        logging.info(f"Context audio duration: {context_audio_decoded_lens[0].item() / model.output_sample_rate:.2f}s")
        logging.info(f"Audio duration: {output.audio_len[0].item() / model.output_sample_rate:.2f}s")
        logging.info(f"Generated codes shape: {output.audio_codes.shape}")
        if output.phoneme_text:
            logging.info(f"Predicted phoneme text: {output.phoneme_text[0]}")

        # Print timing summary
        logging.info("\n=== Timing Summary ===")
        logging.info(f"Init time: {timing_info['init_time']:.3f}s")
        logging.info(f"Generation time: {timing_info['generation_time']:.3f}s")
        logging.info(f"Total time: {timing_info['total_time']:.3f}s")
        logging.info(f"Text tokens processed: {timing_info['num_text_tokens']}")
        logging.info(f"  - Prompt phase tokens: {timing_info['prompt_phase_tokens']}")
        logging.info(f"  - Phoneme-only phase tokens: {timing_info['phoneme_only_phase_tokens']}")
        logging.info(f"Audio frames generated: {timing_info['num_audio_frames']}")
        logging.info(f"Phoneme frames generated: {timing_info['num_phoneme_frames']}")
        logging.info(f"Continuation steps: {timing_info['continuation_steps']}")

        # Calculate RTF
        audio_duration = output.audio_len[0].item() / model.output_sample_rate
        rtf = audio_duration / timing_info['total_time']
        logging.info(f"Real-time factor (RTF): {rtf:.2f}x")

    else:
        logging.info(f"\n=== Running batched streaming inference (batch_size={batch_size}) ===")
        output, timing_info = run_batched_streaming_inference(
            model=model,
            context_audios=context_audios,
            context_audio_lens_list=context_audio_lens_list,
            context_texts=context_texts,
            texts=texts,
            phoneme_texts=phoneme_texts,
            use_gt_phonemes=args.use_gt_phonemes,
            inference_mode=args.inference_mode,
            use_cfg=args.use_cfg,
            cfg_scale=args.cfg_scale,
            use_local_transformer=args.use_local_transformer,
            temperature=args.temperature,
            topk=args.topk,
            max_steps=args.max_steps,
            verbose=args.verbose,
            force_dropout_text=args.force_dropout_text,
        )

        # Save outputs for each batch item
        output_dir = os.path.dirname(args.output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        output_base, output_ext = os.path.splitext(args.output_path)

        for i in range(batch_size):
            output_path_i = f"{output_base}_{i}{output_ext}"
            audio_np = output.audio[i, : output.audio_len[i].item()].cpu().numpy()
            sf.write(output_path_i, audio_np, model.output_sample_rate)
            audio_duration_i = output.audio_len[i].item() / model.output_sample_rate
            logging.info(f"Output {i+1}/{batch_size} saved to: {output_path_i} (duration: {audio_duration_i:.2f}s)")
            if output.phoneme_text and i < len(output.phoneme_text):
                logging.info(f"  Predicted phoneme text: {output.phoneme_text[i]}")

        logging.info(f"\nGenerated codes shape: {output.audio_codes.shape}")

        # Print timing summary
        logging.info("\n=== Timing Summary ===")
        logging.info(f"Init time: {timing_info['init_time']:.3f}s")
        logging.info(f"Generation time: {timing_info['generation_time']:.3f}s")
        logging.info(f"Total time: {timing_info['total_time']:.3f}s")
        logging.info(f"Text tokens per item: {timing_info['num_text_tokens']}")
        logging.info(f"Audio frames generated: {timing_info['num_audio_frames']}")
        logging.info(f"Total steps: {timing_info['total_steps']}")

        # Calculate average RTF
        total_audio_duration = sum(output.audio_len[i].item() for i in range(batch_size)) / model.output_sample_rate
        avg_rtf = total_audio_duration / timing_info['total_time']
        logging.info(f"Average real-time factor (RTF): {avg_rtf:.2f}x")
        logging.info(f"Total audio duration (all items): {total_audio_duration:.2f}s")


if __name__ == "__main__":
    main()
