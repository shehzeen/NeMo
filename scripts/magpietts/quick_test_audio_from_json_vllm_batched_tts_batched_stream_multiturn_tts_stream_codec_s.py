# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.

"""Real-Time Multi-Turn Streaming Inference: Nemotron Nano VL (Async STT) -> EasyMagpieTTS (Artifact Fixed) -> Parakeet ASR."""

import argparse
import os
import sys
import time
import types
import warnings
import json
import asyncio
import uuid
import traceback
from copy import deepcopy

WORKSPACE_DIR = "/lustre/fsw/portfolios/convai/users/ecasanova/Nemotron-omni/cache"
os.makedirs(WORKSPACE_DIR, exist_ok=True)

os.environ['VLLM_CACHE_ROOT'] = WORKSPACE_DIR
os.environ['VLLM_CONFIG_ROOT'] = WORKSPACE_DIR
os.environ['HF_HOME'] = WORKSPACE_DIR
os.environ['TRITON_CACHE_DIR'] = WORKSPACE_DIR
os.environ['XDG_CACHE_HOME'] = WORKSPACE_DIR
os.environ['XDG_CONFIG_HOME'] = WORKSPACE_DIR
os.environ['HF_HOME'] = WORKSPACE_DIR
os.environ['VLLM_NO_USAGE_STATS'] = '1'
os.environ['OMP_NUM_THREADS'] = '2'

# Stub out megatron's unified_memory module
_um = types.ModuleType("megatron.core.inference.unified_memory")
_um.has_unified_memory = False
_um.create_unified_mempool = None
sys.modules["megatron.core.inference.unified_memory"] = _um

import numpy as np
import soundfile as sf
import torch
from nemo.collections.audio.parts.utils.transforms import resample
from omegaconf import OmegaConf, open_dict

# --- vLLM Async Imports ---
from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine

from transformers import AutoProcessor

# --- TTS Streaming Imports ---
from nemo.collections.tts.models import AudioCodecModel
from nemo.collections.tts.modules.audio_codec_modules import VectorQuantizerIndexConverter
from nemo.collections.tts.modules.magpietts_modules import CodecHelper
from nemo.collections.tts.models.easy_magpietts_inference import EasyMagpieTTSInferenceModel
from nemo.collections.asr.models import EncDecRNNTBPEModel
from nemo.utils import logging

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

warnings.filterwarnings("ignore")

class EndToEndSpeechPipeline:
    def __init__(
        self,
        tts_model_path: str,
        codec_model_path: str,
        s2t_model_path: str,
        asr_model_name: str = "nvidia/parakeet-tdt-1.1b",
        phoneme_tokenizer_path: str = None,
        hparams_file: str = None,
        checkpoint_file: str = None,
        device: str = "cuda:0",
        num_turns: int = 2
    ):
        self.device = device
        self.num_turns = num_turns
        self.codec_model_path = codec_model_path
        
        print("=" * 50)
        print("1. Loading Models...")
        print("=" * 50)
        t0 = time.perf_counter()

        self.processor = AutoProcessor.from_pretrained(s2t_model_path, trust_remote_code=True)

        self.s2t_llm = self._load_async_s2t_model(s2t_model_path)
        self.tts_model = self._load_tts_model(
            tts_model_path, codec_model_path, phoneme_tokenizer_path,
            hparams_file, checkpoint_file
        )
        
        self.tts_main_tokenizer_name = list(self.tts_model.cfg.text_tokenizers.keys())[0]
        self.decode_codec_helper = self._build_decode_codec_helper()
        
        # From reference: decode in chunks to avoid artifacts and improve RTF
        self.decode_every_frames = 4

        print(f"\n[PROFILE] Setup completed in: {time.perf_counter() - t0:.1f}s")

    def _apply_inference_overrides(self, model_cfg, codec_model_path, phoneme_tokenizer_path):
        with open_dict(model_cfg):
            model_cfg.target = 'nemo.collections.tts.models.easy_magpietts_inference.EasyMagpieTTSInferenceModel'
            model_cfg.codecmodel_path = codec_model_path
            model_cfg.train_ds = None
            model_cfg.validation_ds = None
            model_cfg.run_val_inference = False
            model_cfg.use_utmos = False
            model_cfg.use_meta_init_for_decoder = True
            if getattr(model_cfg, "phoneme_tokenizer", None) is not None:
                model_cfg.phoneme_tokenizer.tokenizer_path = phoneme_tokenizer_path
        return model_cfg

    def _load_tts_model(self, model_path, codec_model_path, phoneme_tokenizer_path, hparams_file, checkpoint_file):
        if hparams_file is not None and checkpoint_file is not None:
            model_cfg = OmegaConf.load(hparams_file)
            if "cfg" in model_cfg:
                model_cfg = model_cfg.cfg
            model_cfg = self._apply_inference_overrides(model_cfg, codec_model_path, phoneme_tokenizer_path)

            model = EasyMagpieTTSInferenceModel(cfg=model_cfg)
            ckpt = torch.load(checkpoint_file, map_location='cpu')
            model.load_state_dict(ckpt['state_dict'])
        else:
            model_cfg = EasyMagpieTTSInferenceModel.restore_from(model_path, return_config=True)
            model_cfg = self._apply_inference_overrides(model_cfg, codec_model_path, phoneme_tokenizer_path)

            model = EasyMagpieTTSInferenceModel.restore_from(
                model_path,
                override_config_path=model_cfg,
                map_location=torch.device('cpu'),
            )

        model.use_kv_cache_for_inference = True
        model.eval().to(self.device).float()
        return model

    def _build_decode_codec_helper(self) -> CodecHelper:
        codec_model = AudioCodecModel.restore_from(
            self.codec_model_path,
            strict=False,
            map_location=torch.device("cpu"),
        )
        if hasattr(codec_model, "discriminator"):
            del codec_model.discriminator
        codec_model.freeze()
        codec_model = codec_model.to(self.device).eval()

        codec_converter = None
        if self.tts_model._codec_converter is not None:
            vq_new = deepcopy(self.tts_model._codec_converter.vector_quantizer_new).to(self.device).eval()
            codec_converter = VectorQuantizerIndexConverter(
                vector_quantizer_original=codec_model.vector_quantizer,
                vector_quantizer_new=vq_new,
            ).to(self.device)
            codec_converter.eval()

        return CodecHelper(codec_model=codec_model, codec_converter=codec_converter)

    def _load_async_s2t_model(self, model_path: str):
        print(f"Loading ASYNC STT model from {model_path} via vLLM...")
        max_turns = max(10, self.num_turns)
        engine_args = AsyncEngineArgs(
            model=model_path,
            trust_remote_code=True,
            max_model_len=8192, 
            gpu_memory_utilization=0.85, 
            max_num_seqs=1, 
            enforce_eager=False, 
            enable_prefix_caching=True, 
            limit_mm_per_prompt={"video": 1, "image": 1, "audio": max_turns}, 
            swap_space=8,
            mamba_ssm_cache_dtype="float32"
        )
        llm = AsyncLLMEngine.from_engine_args(engine_args)
        print("Async vLLM STT Model loaded successfully!")
        return llm

    def _build_context_inputs(self, context_text: str, context_audio_path: str = None):
        device = next(self.tts_model.parameters()).device
        model_dtype = next(self.tts_model.parameters()).dtype

        text_ids = self.tts_model.tokenizer.encode(context_text, tokenizer_name=self.tts_model.text_conditioning_tokenizer_name)
        context_text_tokens = torch.tensor([text_ids], dtype=torch.long, device=device)
        context_text_lens = torch.tensor([len(text_ids)], dtype=torch.long, device=device)

        if context_audio_path:
            context_audio = self.tts_model._load_audio_for_inference(context_audio_path, self.tts_model.sample_rate)
            context_audio = self.tts_model._adjust_audio_to_duration_for_inference(
                context_audio,
                self.tts_model.sample_rate,
                5.0,
                self.tts_model.codec_model_samples_per_frame,
            )
            context_audio = context_audio.to(device=device, dtype=model_dtype)
            context_audio_lens = torch.tensor([context_audio.size(1)], dtype=torch.long, device=device)
            with torch.inference_mode():
                context_audio_codes, context_audio_codes_lens = self.tts_model._codec_helper.audio_to_codes(context_audio, context_audio_lens)
        else:
            context_audio_codes = torch.zeros(1, self.tts_model.data_num_audio_codebooks, 0, dtype=torch.long, device=device)
            context_audio_codes_lens = torch.zeros(1, dtype=torch.long, device=device)

        return context_audio_codes, context_audio_codes_lens, context_text_tokens, context_text_lens

    def _create_base_streaming_state(self, context_text, context_audio_path, use_cfg, cfg_scale, temperature, topk):
        codes, codes_lens, text_tokens, text_lens = self._build_context_inputs(context_text, context_audio_path)
        with torch.inference_mode():
            state = self.tts_model.streaming_init(
                context_audio_codes=codes,
                context_audio_codes_lens=codes_lens,
                context_text_tokens=text_tokens,
                context_text_tokens_lens=text_lens,
                use_cfg=use_cfg,
                cfg_scale=cfg_scale,
                use_local_transformer=True,
                temperature=temperature,
                topk=topk,
                phoneme_input_type="pred",
                phoneme_sampling_method="argmax",
                use_inference_mode=True,
            )
        return state

    def _decode_new_audio_chunk(self, all_audio_codes: torch.Tensor, last_emitted_sample_idx: int) -> tuple[np.ndarray, int]:
        pred_codes_lens = torch.tensor([all_audio_codes.size(-1)], dtype=torch.long, device=all_audio_codes.device)
        pred_codes, pred_codes_lens = self.tts_model._prepare_codes_for_decode(all_audio_codes, pred_codes_lens)
        audio, audio_len, _ = self.decode_codec_helper.codes_to_audio(pred_codes, pred_codes_lens)
        full_wav = audio[0, : audio_len[0]].detach().float().cpu().numpy()
        if full_wav.shape[0] <= last_emitted_sample_idx:
            return np.zeros((0,), dtype=np.float32), last_emitted_sample_idx
        chunk = full_wav[last_emitted_sample_idx:]
        return chunk, full_wav.shape[0]

    async def infer_stream(
        self,
        input_audio_path: str,
        output_audio_path: str,
        session_messages: list,
        session_audio_arrays: list,
        turn_index: int,
        s2t_prompt: str = "Transcribe the audio.",
        max_s2t_tokens: int = 1024,
        context_audio_path: str = None,
        context_text: str = None,
        language: str = "en",
        use_cfg: bool = True,
        cfg_scale: float = 2.5,
        temperature: float = 0.7,
        topk: int = 80,
        max_tts_steps: int = 500
    ):
        print("\n" + "=" * 50)
        print(f"Processing Turn {turn_index}: {os.path.basename(input_audio_path)}")
        print("=" * 50)
        
        if context_audio_path:
            use_language_tag = bool(getattr(self.tts_model, "add_language_to_context_text", False))
            resolved_context_text = f"[{language.upper()}]" if use_language_tag else "[NO TEXT CONTEXT]"
        else:
            resolved_context_text = (context_text or "[NO TEXT CONTEXT]").strip()

        tts_state = self._create_base_streaming_state(
            resolved_context_text, context_audio_path, use_cfg, cfg_scale, temperature, topk
        )
        
        text_buffer = ""
        sent_tts_token_ids = []
        accumulated_audio_codes = None
        
        # Audio Artifact Filtering Trackers
        generated_audio_frames = 0
        last_decode_frame_mark = 0
        last_emitted_sample_idx = 0
        decoded_chunks = []
        
        interleaved_tts_time = 0.0
        flush_tts_time = 0.0
        tts_steps = 0
        
        sampling_params = SamplingParams(max_tokens=max_s2t_tokens, temperature=0.6, top_p=0.95, skip_special_tokens=False)

        audio_array, sr = sf.read(input_audio_path, dtype="float32")
        if len(audio_array.shape) > 1: audio_array = audio_array.mean(axis=1)
        if sr != 16000:
            audio_tensor = torch.from_numpy(audio_array).unsqueeze(0)
            audio_tensor = resample(audio_tensor, orig_freq=sr, new_freq=16000)
            audio_array = audio_tensor[0].numpy()
            sr = 16000

        if not session_messages:
            session_messages.append({"role": "system", "content": [{"type": "text", "text": "/no_think"}]})
        session_messages.append({"role": "user", "content": [{"type": "audio"}, {"type": "text", "text": s2t_prompt}]})
        session_audio_arrays.append(audio_array)

        prompt_text = self.processor.apply_chat_template(session_messages, add_generation_prompt=True, tokenize=False)
        prompt_token_ids = self.processor.tokenizer.encode(prompt_text)
        if isinstance(prompt_token_ids, list) and len(prompt_token_ids) > 0 and isinstance(prompt_token_ids[0], list):
            prompt_token_ids = prompt_token_ids[0]

        inputs = {"prompt": prompt_text, "prompt_token_ids": prompt_token_ids, "multi_modal_data": {"audio": [(arr, sr) for arr in session_audio_arrays]}}
        request_id = str(uuid.uuid4())
        
        t_stt = time.perf_counter()
        results_generator = self.s2t_llm.generate(inputs, sampling_params, request_id)

        transcript = ""
        first_token_time = None
        ttft = 0.0
        ttfa = 0.0
        total_tokens = 0
        stt_blocking_time = 0.0
        t_await_start = time.perf_counter()
        
        special_token_ids = set(self.processor.tokenizer.all_special_ids)
        raw_tokens_processed = 0 
        
        async for request_output in results_generator:
            t_await_end = time.perf_counter()
            if first_token_time is None:
                first_token_time = t_await_end
                ttft = first_token_time - t_stt
                print(f"[PROFILE] Time To First Token (TTFT): {ttft:.3f}s")
                print("STT Stream: ", end="", flush=True)
            else:
                stt_blocking_time += (t_await_end - t_await_start)

            text = request_output.outputs[0].text
            new_text = text[len(transcript):]
            sys.stdout.write(new_text)
            sys.stdout.flush()
            transcript = text
            text_buffer += new_text
            total_tokens = len(request_output.outputs[0].token_ids)
            
            # --- HIGHLY OPTIMIZED INTERLEAVED TTS LOOP ---
            t_tts_overhead_start = time.perf_counter()
            raw_llm_ids = list(request_output.outputs[0].token_ids)
            new_raw_ids = raw_llm_ids[raw_tokens_processed:]
            raw_tokens_processed = len(raw_llm_ids)
            
            # Artifact Fix Part 1: Strip structural tokens that trigger audio noise
            new_tts_ids = [tid for tid in new_raw_ids if tid not in special_token_ids]
            
            if not new_tts_ids:
                interleaved_tts_time += (time.perf_counter() - t_tts_overhead_start)
                t_await_start = time.perf_counter()
                continue
            
            for tid in new_tts_ids:
                sent_tts_token_ids.append(tid)
                tts_text_tokens = torch.tensor([tid], dtype=torch.long, device=self.device)
                with torch.inference_mode():
                    tts_state, audio_codes, _ = self.tts_model.streaming_step(state=tts_state, text_tokens=tts_text_tokens, use_inference_mode=True)
                tts_steps += 1
                
                if audio_codes is not None:
                    if accumulated_audio_codes is None:
                        accumulated_audio_codes = audio_codes.detach()
                    else:
                        accumulated_audio_codes = torch.cat([accumulated_audio_codes, audio_codes.detach()], dim=-1)
                    
                    generated_audio_frames = int(accumulated_audio_codes.size(-1))
                    
                    # Artifact Fix Part 2: Buffer decoding (Wait 6 frames to let internal state settle)
                    if generated_audio_frames - last_decode_frame_mark >= self.decode_every_frames:
                        with torch.inference_mode():
                            chunk_f32, new_sample_idx = self._decode_new_audio_chunk(accumulated_audio_codes, last_emitted_sample_idx)
                        last_decode_frame_mark = generated_audio_frames
                        last_emitted_sample_idx = new_sample_idx
                        if chunk_f32.size > 0:
                            decoded_chunks.append(chunk_f32)
                            if ttfa == 0.0:
                                ttfa = time.perf_counter() - t_stt
                                print(f"\n[PROFILE] True Time To First PCM Audio (TTFA): {ttfa:.3f}s")
            
            interleaved_tts_time += (time.perf_counter() - t_tts_overhead_start)
            t_await_start = time.perf_counter()

        ttat = time.perf_counter() - t_stt
        print()
        
        stt_duration = ttat
        tokens_after_first = max(0, total_tokens - 1)
        tpot_system = (max(0.0, stt_duration - ttft) / tokens_after_first) if tokens_after_first > 0 else 0.0
        tpot_stt_await = (stt_blocking_time / tokens_after_first) if tokens_after_first > 0 else 0.0
        
        print(f"[PROFILE] Total Interleaved Duration: {ttat:.3f}s")
        print(f"[PROFILE] STT Tokens: {total_tokens} | System TPOT: {tpot_system:.3f}s | STT Await TPOT: {tpot_stt_await:.3f}s")
        
        transcript = transcript.replace("<|endoftext|>", "").replace("<|im_end|>", "").strip()
        session_messages.append({"role": "assistant", "content": [{"type": "text", "text": transcript}]})

        # --- FLUSH PHASE ---
        print("--- Flushing TTS Stream ---")
        t_tts_flush_start = time.perf_counter()
        eos_tokens = torch.tensor([self.tts_model.eos_id], dtype=torch.long, device=self.device)
        with torch.inference_mode():
            tts_state, audio_codes, _ = self.tts_model.streaming_step(state=tts_state, text_tokens=eos_tokens, use_inference_mode=True)
        tts_steps += 1
        
        if audio_codes is not None:
            if accumulated_audio_codes is None: accumulated_audio_codes = audio_codes.detach()
            else: accumulated_audio_codes = torch.cat([accumulated_audio_codes, audio_codes.detach()], dim=-1)

        steps = 0
        while not bool(tts_state.finished.all()) and steps < max_tts_steps:
            with torch.inference_mode():
                tts_state, audio_codes, _ = self.tts_model.streaming_step(state=tts_state, text_tokens=None, use_inference_mode=True)
            tts_steps += 1
            steps += 1
            if audio_codes is not None:
                eos_in_audio_codes = audio_codes == self.tts_model.audio_eos_id # (B, C, S)
                if eos_in_audio_codes.any():
                    eos_in_any_codebook = eos_in_audio_codes.any(dim=1) # (B, S)
                    eos_frame_idx = eos_in_any_codebook.int().argmax(dim=1).item()
                    audio_codes = audio_codes[:, :, :eos_frame_idx]
                    print(f"EOS detected in audio codes at frame {eos_frame_idx}")
                accumulated_audio_codes = torch.cat([accumulated_audio_codes, audio_codes.detach()], dim=-1)
                generated_audio_frames = int(accumulated_audio_codes.size(-1))
                if generated_audio_frames - last_decode_frame_mark >= self.decode_every_frames:
                    chunk_f32, last_emitted_sample_idx = self._decode_new_audio_chunk(accumulated_audio_codes, last_emitted_sample_idx)
                    last_decode_frame_mark = generated_audio_frames
                    if chunk_f32.size > 0: decoded_chunks.append(chunk_f32)

        # Final cleanup pass
        if accumulated_audio_codes is not None and last_decode_frame_mark < int(accumulated_audio_codes.size(-1)):
            chunk_f32, _ = self._decode_new_audio_chunk(accumulated_audio_codes, last_emitted_sample_idx)
            if chunk_f32.size > 0: decoded_chunks.append(chunk_f32)

        flush_duration = time.perf_counter() - t_tts_flush_start
        if decoded_chunks:
            full_wav = np.concatenate(decoded_chunks, axis=0)
            sf.write(output_audio_path, full_wav, self.tts_model.output_sample_rate)
            audio_duration = full_wav.shape[0] / self.tts_model.output_sample_rate
            print(f"TTS Stream decoded {audio_duration:.2f}s audio (Flush RTF: {flush_duration/audio_duration:.3f})")
        
        return transcript, stt_duration, time.perf_counter() - t_stt, 0.0, ttft, ttfa, tpot_system, tpot_stt_await, 0.0, session_messages, session_audio_arrays

# ==========================================
# Main Execution Flow
# ==========================================

async def async_main():
    p = argparse.ArgumentParser(description="Multi-Turn Streaming Pipeline: Audio -> STT Stream -> TTS -> Parakeet ASR")
    p.add_argument("--input_json_path", required=True)
    p.add_argument("--audio_base_dir", required=True)
    p.add_argument("--output_audio_dir", required=True)
    p.add_argument("--output_json_path", required=True)
    p.add_argument("--num_turns", type=int, default=6)
    
    s2t_grp = p.add_argument_group("STT Arguments")
    s2t_grp.add_argument("--s2t_model_path", required=True)
    s2t_grp.add_argument("--device", type=str, default="cuda:0")
    s2t_grp.add_argument("--max_new_tokens", type=int, default=1024)
    s2t_grp.add_argument("--s2t_prompt", type=str, default="Transcribe the audio.")

    tts_grp = p.add_argument_group("TTS Arguments")
    tts_grp.add_argument("--codec_model_path", required=True)
    tts_grp.add_argument("--tts_model_path", default=None)
    tts_grp.add_argument("--hparams_file", default=None)
    tts_grp.add_argument("--checkpoint_file", default=None)
    tts_grp.add_argument("--phoneme_tokenizer_path", default=None)
    tts_grp.add_argument("--language", default="en", choices=["en", "zh", "es", "fr", "de", "it", "hi", "vi"])
    tts_grp.add_argument("--temperature", type=float, default=0.7)
    tts_grp.add_argument("--topk", type=int, default=80)
    tts_grp.add_argument("--use_cfg", action="store_true", default=True)
    tts_grp.add_argument("--cfg_scale", type=float, default=2.5)
    tts_grp.add_argument("--max_steps", type=int, default=500)

    ctx = tts_grp.add_mutually_exclusive_group()
    ctx.add_argument("--context_audio_path", default=None)
    ctx.add_argument("--context_text", default=None)

    args = p.parse_args()
    os.makedirs(args.output_audio_dir, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.output_json_path)), exist_ok=True)

    phoneme_tokenizer_path = args.phoneme_tokenizer_path or os.path.join(os.path.dirname(os.path.abspath(__file__)), "bpe_ipa_tokenizer_2048_en_de_es_fr_hi_it_vi_zh.json")

    pipeline = EndToEndSpeechPipeline(
        tts_model_path=args.tts_model_path, codec_model_path=args.codec_model_path, s2t_model_path=args.s2t_model_path,
        phoneme_tokenizer_path=phoneme_tokenizer_path, hparams_file=args.hparams_file, checkpoint_file=args.checkpoint_file,
        device=args.device, num_turns=args.num_turns
    )

    with open(args.input_json_path, 'r', encoding='utf-8') as f:
        json_lines = [line.strip() for line in f if line.strip()]
    if os.path.exists(args.output_json_path): os.remove(args.output_json_path)

    tracker_ttft_sum = 0.0
    tracker_count = 0

    for i in range(0, len(json_lines), args.num_turns):
        session_group = json_lines[i:i+args.num_turns]
        session_messages, session_audio_arrays = [], []
        
        for turn_idx, line in enumerate(session_group):
            data = json.loads(line)
            audio_rel_path = data.get("audio_path")
            if not audio_rel_path: continue
                
            input_audio_path = os.path.join(args.audio_base_dir, audio_rel_path)
            base_filename = os.path.splitext(os.path.basename(audio_rel_path))[0]
            output_audio_path = os.path.join(args.output_audio_dir, base_filename + ".wav")

            try:
                transcript, stt_time, total_time, tts_rtf, ttft, ttfa, tpot_system, tpot_stt_await, tpot_tts, session_messages, session_audio_arrays = await pipeline.infer_stream(
                    input_audio_path=input_audio_path, output_audio_path=output_audio_path,
                    session_messages=session_messages, session_audio_arrays=session_audio_arrays,
                    turn_index=turn_idx + 1, s2t_prompt=args.s2t_prompt, max_s2t_tokens=args.max_new_tokens,
                    context_audio_path=args.context_audio_path, context_text=args.context_text,
                    language=args.language, use_cfg=args.use_cfg, cfg_scale=args.cfg_scale,
                    temperature=args.temperature, topk=args.topk, max_tts_steps=args.max_steps
                )
                
                tracker_ttft_sum += ttft
                tracker_count += 1
                
                output_data = {
                    "turn": turn_idx + 1, "ref_user": data.get("problem", ""), "pred_agent": transcript,
                    "stt_time_seconds": round(stt_time, 3), "ttft_seconds": round(ttft, 3), "ttfa_seconds": round(ttfa, 3),
                    "tpot_system_seconds": round(tpot_system, 3), "tpot_stt_await_seconds": round(tpot_stt_await, 3),
                    "output_audio_path": output_audio_path, "data_id": data.get("key", base_filename),
                }
                with open(args.output_json_path, 'a', encoding='utf-8') as f_out:
                    f_out.write(json.dumps(output_data) + "\n")

            except Exception:
                traceback.print_exc()
            finally:
                torch.cuda.empty_cache()

    if tracker_count > 0:
        print(f"\n[SUMMARY] Total Turns: {tracker_count} | Avg TTFT: {tracker_ttft_sum/tracker_count:.3f}s")

    # PHASE 2: ASR Transcription
    print("\nPHASE 2: ASR Transcription...")
    del pipeline
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    try:
        from vllm.distributed.parallel_state import destroy_model_parallel, destroy_distributed_environment
        destroy_model_parallel()
        destroy_distributed_environment()
    except Exception: pass

    asr_model = EncDecRNNTBPEModel.from_pretrained("nvidia/parakeet-tdt-1.1b").to(args.device).eval()
    with open(args.output_json_path, 'r', encoding='utf-8') as f:
        completed_lines = [json.loads(line) for line in f if line.strip()]

    updated_lines = []
    for item in completed_lines:
        out_audio = item.get("output_audio_path")
        if out_audio and os.path.exists(out_audio):
            waveform, sr = sf.read(out_audio)
            if sr != 16000: waveform = resample(torch.from_numpy(waveform).float().unsqueeze(0), orig_freq=sr, new_freq=16000)[0].numpy()
            target_asr_texts = asr_model.transcribe([waveform], batch_size=1, verbose=False)
            item["speech_pred_transcribed"] = target_asr_texts[0].text
        updated_lines.append(item)

    with open(args.output_json_path, 'w', encoding='utf-8') as f_out:
        for item in updated_lines: f_out.write(json.dumps(item) + "\n")
    print("PIPELINE COMPLETE")

if __name__ == "__main__":
    asyncio.run(async_main())
