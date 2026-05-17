"""Minimal single-utterance inference script for EasyMagpieTTS."""
import argparse
import os
import sys
import time
import types

os.environ['OMP_NUM_THREADS'] = '2'

# Stub out megatron's unified_memory module to skip a 9.5s CUDA JIT compile
# that always fails in this environment and is unused for TTS inference.
_um = types.ModuleType("megatron.core.inference.unified_memory")
_um.has_unified_memory = False
_um.create_unified_mempool = None
sys.modules["megatron.core.inference.unified_memory"] = _um

import soundfile as sf
import torch
from omegaconf import OmegaConf, open_dict

from nemo.collections.tts.models.easy_magpietts_inference import EasyMagpieTTSInferenceModel
from nemo.utils import logging


def _apply_inference_overrides(model_cfg, codec_model_path, phoneme_tokenizer_path):
    """Apply config overrides that disable training-only features and enable fast init."""
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


def load_model(model_path, codec_model_path, phoneme_tokenizer_path,
               hparams_file=None, checkpoint_file=None):
    """Load an EasyMagpieTTS model configured for inference.

    Supports two loading modes:
      1. .nemo archive:  pass model_path pointing to a .nemo file.
      2. Checkpoint mode: pass hparams_file (.yaml) and checkpoint_file (.ckpt).
    """
    if hparams_file is not None and checkpoint_file is not None:
        logging.info(f"Loading from checkpoint: {checkpoint_file}")
        model_cfg = OmegaConf.load(hparams_file)
        if "cfg" in model_cfg:
            model_cfg = model_cfg.cfg
        model_cfg = _apply_inference_overrides(model_cfg, codec_model_path, phoneme_tokenizer_path)

        model = EasyMagpieTTSInferenceModel(cfg=model_cfg)
        ckpt = torch.load(checkpoint_file, map_location='cpu')
        model.load_state_dict(ckpt['state_dict'])
    else:
        logging.info(f"Loading from .nemo archive: {model_path}")
        model_cfg = EasyMagpieTTSInferenceModel.restore_from(model_path, return_config=True)
        model_cfg = _apply_inference_overrides(model_cfg, codec_model_path, phoneme_tokenizer_path)

        model = EasyMagpieTTSInferenceModel.restore_from(
            model_path,
            override_config_path=model_cfg,
            map_location=torch.device('cpu'),
        )

    model.use_kv_cache_for_inference = True
    model.eval().cuda().float()
    return model


def main():
    p = argparse.ArgumentParser(description="EasyMagpieTTS single inference")
    p.add_argument("--transcript", required=True, help="Text to synthesize")
    p.add_argument("--codec_model_path", required=True, help="Path to .nemo codec model")
    p.add_argument("--output_path", required=True, help="Output .wav file path")

    model_src = p.add_argument_group("model source (provide one)")
    model_src.add_argument("--model_path", default=None, help="Path to .nemo TTS checkpoint")
    model_src.add_argument("--hparams_file", default=None, help="Path to hparams.yaml (use with --checkpoint_file)")
    model_src.add_argument("--checkpoint_file", default=None, help="Path to .ckpt file (use with --hparams_file)")

    ctx = p.add_mutually_exclusive_group(required=True)
    ctx.add_argument("--context_audio_path", help="Path to context audio file")
    ctx.add_argument("--context_text", help="Context text string")

    p.add_argument("--phoneme_tokenizer_path", default=None,
                   help="Path to phoneme tokenizer JSON (auto-detected next to this script if omitted)")
    p.add_argument("--language", default="en", choices=["en", "zh", "es", "fr", "de", "it", "hi", "vi"])
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--topk", type=int, default=80)
    p.add_argument("--use_cfg", action="store_true", default=True)
    p.add_argument("--no_cfg", dest="use_cfg", action="store_false")
    p.add_argument("--cfg_scale", type=float, default=2.5)
    p.add_argument("--max_steps", type=int, default=300)
    args = p.parse_args()

    has_ckpt = args.hparams_file is not None and args.checkpoint_file is not None
    has_nemo = args.model_path is not None
    if not has_ckpt and not has_nemo:
        p.error("Provide either --model_path (.nemo) or both --hparams_file and --checkpoint_file")

    phoneme_tokenizer_path = args.phoneme_tokenizer_path
    if phoneme_tokenizer_path is None:
        phoneme_tokenizer_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "bpe_ipa_tokenizer_2048_en_de_es_fr_hi_it_vi_zh.json",
        )

    t0 = time.perf_counter()
    model = load_model(
        args.model_path, args.codec_model_path, phoneme_tokenizer_path,
        hparams_file=args.hparams_file, checkpoint_file=args.checkpoint_file,
    )
    print(f"[PROFILE] Setup time: {time.perf_counter() - t0:.1f}s")

    # --- resolve context ---
    if args.context_audio_path:
        use_language_tag = bool(getattr(model, "add_language_to_context_text", False))
        context_text = f"[{args.language.upper()}]" if use_language_tag else "[NO TEXT CONTEXT]"
        context_audio_path = args.context_audio_path
    else:
        context_text = args.context_text
        context_audio_path = None

    transcript = args.transcript.strip()
    if not transcript.endswith((".", "?", "!")):
        transcript += "."

    # --- infer ---
    audio, audio_len = model.do_tts(
        transcript=transcript,
        context_audio_file_path=context_audio_path,
        context_text=context_text,
        use_cfg=args.use_cfg,
        cfg_scale=args.cfg_scale,
        use_local_transformer=True,
        temperature=args.temperature,
        topk=args.topk,
        max_steps=args.max_steps,
    )

    audio_np = audio[0, : audio_len[0]].cpu().numpy()
    sf.write(args.output_path, audio_np, model.output_sample_rate)
    print(f"Saved {args.output_path}  ({audio_np.shape[0]} samples, {model.output_sample_rate} Hz)")
    print(f"[PROFILE] Total time: {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()