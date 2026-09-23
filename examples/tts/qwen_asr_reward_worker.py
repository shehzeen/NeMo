#!/usr/bin/env python3
"""Persistent JSON-lines Qwen3-ASR worker for GRPO reward computation."""

import argparse
import contextlib
import json
import sys

import torch
from qwen_asr import Qwen3ASRModel

LANGUAGE_MAP = {
    "ar": "Arabic",
    "de": "German",
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "hi": "Hindi",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "pt": "Portuguese",
    "vi": "Vietnamese",
    "zh": "Chinese",
}


def emit(payload):
    sys.__stdout__.write(json.dumps(payload, ensure_ascii=False) + "\n")
    sys.__stdout__.flush()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    args = parser.parse_args()

    with contextlib.redirect_stdout(sys.stderr):
        model = Qwen3ASRModel.from_pretrained(
            args.model,
            dtype=torch.bfloat16,
            device_map=args.device,
            max_inference_batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
        )
    emit({"status": "ready", "model": args.model, "device": args.device})

    for line in sys.stdin:
        try:
            request = json.loads(line)
            if request.get("command") == "shutdown":
                emit({"status": "stopped"})
                return
            if request.get("command") != "transcribe":
                raise ValueError(f"Unknown command: {request.get('command')!r}")
            audio_paths = request["audio_paths"]
            languages = [LANGUAGE_MAP.get(language, language) for language in request["languages"]]
            if len(audio_paths) != len(languages):
                raise ValueError(
                    f"audio_paths and languages must have equal lengths, got {len(audio_paths)} and {len(languages)}"
                )
            with contextlib.redirect_stdout(sys.stderr):
                results = model.transcribe(audio=audio_paths, language=languages)
            emit(
                {
                    "status": "ok",
                    "transcripts": [result.text for result in results],
                    "detected_languages": [result.language for result in results],
                }
            )
        except Exception as error:
            emit({"status": "error", "error": repr(error)})


if __name__ == "__main__":
    main()
