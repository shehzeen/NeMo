"""
Inspect the cross-lingual context dataset by decoding target and context
audio codes back to waveforms and saving them alongside the original
recording audio for comparison.

Usage (inside docker):
    python scripts/magpietts/inspect_crosslingual_dataset.py \
        --shar-dir /data/crosslingual_context_dataset/lhotse_shar \
        --codec-model-path /model_artifacts/25fps_spectral_codec_with_bandwidth_extension.nemo \
        --codec-name 25fpsSpectralCodecBWE \
        --output-dir /data/crosslingual_context_dataset/inspect \
        --num-samples 10
"""

import argparse
import logging
import os

import numpy as np
import soundfile as sf
import torch
from lhotse import CutSet

from nemo.collections.tts.models import AudioCodecModel


def main():
    parser = argparse.ArgumentParser(description="Inspect cross-lingual dataset: decode codes and save audio.")
    parser.add_argument("--shar-dir", required=True, help="Path to lhotse_shar directory.")
    parser.add_argument("--codec-model-path", required=True, help="Path to .nemo codec model.")
    parser.add_argument("--codec-name", default="25fpsSpectralCodecBWE", help="Codec subdirectory name.")
    parser.add_argument("--output-dir", required=True, help="Directory to save inspection outputs.")
    parser.add_argument("--num-samples", type=int, default=10, help="Number of samples to inspect.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load codec model
    logging.info(f"Loading codec model from {args.codec_model_path}")
    codec_model = AudioCodecModel.restore_from(args.codec_model_path, map_location="cpu", strict=False)
    codec_model = codec_model.to(device)
    codec_model.eval()
    codec_sr = codec_model.sample_rate
    logging.info(f"Codec output sample rate: {codec_sr}")

    # Build shar fields for first shard
    cuts_dir = os.path.join(args.shar_dir, "cuts")
    target_audio_dir = os.path.join(args.shar_dir, "target_audio")
    context_audio_dir = os.path.join(args.shar_dir, "context_audio")
    target_codes_dir = os.path.join(args.shar_dir, args.codec_name, "target_codes")
    context_codes_dir = os.path.join(args.shar_dir, args.codec_name, "context_codes")

    # Use first shard only
    fields = {
        "cuts": [os.path.join(cuts_dir, "cuts.000000.jsonl.gz")],
        "recording": [os.path.join(target_audio_dir, "recording.000000.tar")],
        "context_recording": [os.path.join(context_audio_dir, "recording.000000.tar")],
        "target_codes": [os.path.join(target_codes_dir, "codes.000000.tar")],
        "context_codes": [os.path.join(context_codes_dir, "codes.000000.tar")],
    }

    for k, v in fields.items():
        if not os.path.isfile(v[0]):
            logging.error(f"Missing file for '{k}': {v[0]}")
            return

    logging.info("Loading CutSet from shar...")
    cutset = CutSet.from_shar(fields=fields)

    count = 0
    for cut in cutset:
        if count >= args.num_samples:
            break

        sup = cut.supervisions[0] if cut.supervisions else None
        lang = sup.language if sup else "unk"
        speaker = sup.speaker if sup else "unk"
        ctx_lang = sup.custom.get("context_language", "unk") if sup and hasattr(sup, "custom") else "unk"
        ssim = sup.custom.get("context_speaker_similarity", "N/A") if sup and hasattr(sup, "custom") else "N/A"

        sample_dir = os.path.join(args.output_dir, f"sample_{count:03d}_{lang}")
        os.makedirs(sample_dir, exist_ok=True)

        logging.info(f"--- Sample {count} ---")
        logging.info(f"  Cut ID: {cut.id}")
        logging.info(f"  Target lang: {lang}, Context lang: {ctx_lang}, SSIM: {ssim}")
        logging.info(f"  Speaker: {speaker}")
        if sup:
            logging.info(f"  Text: {sup.text[:80]}...")

        # 1. Save original target recording audio
        target_audio_np = cut.recording.resample(codec_sr).load_audio().squeeze(0)
        sf.write(os.path.join(sample_dir, "target_recording.wav"), target_audio_np, codec_sr)
        logging.info(f"  Saved target_recording.wav ({len(target_audio_np)/codec_sr:.2f}s)")

        # 2. Save original context recording audio
        if cut.has_custom("context_recording"):
            ctx_audio_np = cut.context_recording.resample(codec_sr).load_audio().squeeze(0)
            sf.write(os.path.join(sample_dir, "context_recording.wav"), ctx_audio_np, codec_sr)
            logging.info(f"  Saved context_recording.wav ({len(ctx_audio_np)/codec_sr:.2f}s)")

        # 3. Decode target codes -> audio
        if cut.has_custom("target_codes"):
            target_codes_np = cut.target_codes.load().astype(np.int32)  # (C, T)
            target_codes_t = torch.from_numpy(target_codes_np).unsqueeze(0).to(device)  # (1, C, T)
            target_codes_len = torch.tensor([target_codes_t.shape[2]], device=device)
            with torch.inference_mode():
                decoded_target, decoded_target_len = codec_model.decode(
                    tokens=target_codes_t, tokens_len=target_codes_len
                )
            decoded_target_np = decoded_target[0, : decoded_target_len[0]].cpu().float().numpy()
            sf.write(
                os.path.join(sample_dir, "target_decoded_from_codes.wav"),
                decoded_target_np,
                codec_model.output_sample_rate,
            )
            logging.info(
                f"  Saved target_decoded_from_codes.wav ({len(decoded_target_np)/codec_model.output_sample_rate:.2f}s), codes shape: {target_codes_np.shape}"
            )
        else:
            logging.warning(f"  No target_codes found for cut {cut.id}")

        # 4. Decode context codes -> audio
        if cut.has_custom("context_codes"):
            ctx_codes_np = cut.context_codes.load().astype(np.int32)  # (C, T)
            ctx_codes_t = torch.from_numpy(ctx_codes_np).unsqueeze(0).to(device)  # (1, C, T)
            ctx_codes_len = torch.tensor([ctx_codes_t.shape[2]], device=device)
            with torch.inference_mode():
                decoded_ctx, decoded_ctx_len = codec_model.decode(tokens=ctx_codes_t, tokens_len=ctx_codes_len)
            decoded_ctx_np = decoded_ctx[0, : decoded_ctx_len[0]].cpu().float().numpy()
            sf.write(
                os.path.join(sample_dir, "context_decoded_from_codes.wav"),
                decoded_ctx_np,
                codec_model.output_sample_rate,
            )
            logging.info(
                f"  Saved context_decoded_from_codes.wav ({len(decoded_ctx_np)/codec_model.output_sample_rate:.2f}s), codes shape: {ctx_codes_np.shape}"
            )
        else:
            logging.warning(f"  No context_codes found for cut {cut.id}")

        # 5. Write metadata
        with open(os.path.join(sample_dir, "info.txt"), "w") as f:
            f.write(f"cut_id: {cut.id}\n")
            f.write(f"target_language: {lang}\n")
            f.write(f"context_language: {ctx_lang}\n")
            f.write(f"speaker: {speaker}\n")
            f.write(f"context_speaker_similarity: {ssim}\n")
            f.write(f"text: {sup.text if sup else ''}\n")
            f.write(f"duration: {cut.duration}\n")

        count += 1

    logging.info(f"Done. Saved {count} samples to {args.output_dir}")


if __name__ == "__main__":
    main()
