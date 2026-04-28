#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

DEFAULT_CONFIG_PATH = Path(__file__).parent / "ipa_manifest_config.json"

# Keep this mapping aligned with ipa_script.py
ESPEAK_VOICE_BY_LANG: Dict[str, str] = {
    "de": "de",
    "en": "en",
    "es": "es",
    "fr": "fr",
    "hi": "hi",
    "it": "it",
    "vi": "vi",
    "zh": "zh",
    "ru": "ru",
    "ja": "ja",
    "ko": "ko",
    "ar": "ar",
    "he": "he",
    "nl": "nl",
    "pl": "pl",
    "pt": "pt",
    "ar-AE": "ar",
    "ar-MSA": "ar",
    "ar-SA": "ar",
    "ar-SY": "ar",
    "ko-KR": "ko",
}

IPA_FLAG = "--ipa"
COMMON_FLAGS = ["-q"]
MAX_ESPEAK_RETRIES = 2
ESPEAK_RETRY_SLEEP_SECONDS = 0.1
_WS_RE = re.compile(r"\s+")


def load_config(path: Path) -> Dict[str, List[str]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a JSON object, got {type(data)}")
    return data


def find_espeak_binaries() -> List[str]:
    binaries: List[str] = []
    for exe in ("espeak-ng", "espeak"):
        if shutil.which(exe):
            binaries.append(exe)
    if binaries:
        print(f"[INFO] Found espeak binaries on PATH: {', '.join(binaries)}")
        return binaries
    raise RuntimeError("Neither 'espeak-ng' nor 'espeak' found on PATH.")


class EspeakRunner:
    def __init__(self, exe: str, voice: str, fallback_exe: Optional[str] = None) -> None:
        self.exe = exe
        self.voice = voice
        self.fallback_exe = fallback_exe
        self._cache: Dict[Tuple[str, str], str] = {}

    def text_to_ipa(self, text: str) -> str:
        cached = self._cache.get((self.voice, text))
        if cached is not None:
            return cached

        errors: List[str] = []
        executables = [self.exe]
        if self.fallback_exe and self.fallback_exe != self.exe:
            executables.append(self.fallback_exe)

        for exe in executables:
            cmd = [exe, "-v", self.voice, IPA_FLAG] + COMMON_FLAGS
            for attempt in range(1, MAX_ESPEAK_RETRIES + 2):
                try:
                    proc = subprocess.run(
                        cmd,
                        input=text.encode("utf-8"),
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        check=False,
                    )
                except Exception as e:
                    errors.append(f"cmd={' '.join(cmd)} attempt={attempt} error={e}")
                    break

                if proc.returncode == 0:
                    out = proc.stdout.decode("utf-8", errors="replace").strip()
                    out = _WS_RE.sub(" ", out).strip()
                    self._cache[(self.voice, text)] = out
                    return out

                stderr = proc.stderr.decode("utf-8", errors="replace")
                errors.append(f"cmd={' '.join(cmd)} attempt={attempt} rc={proc.returncode} stderr={stderr}")
                if proc.returncode < 0 and attempt <= MAX_ESPEAK_RETRIES:
                    time.sleep(ESPEAK_RETRY_SLEEP_SECONDS)
                    continue
                break

        raise RuntimeError("espeak command failed:\n" + "\n".join(errors))


def choose_text(item: dict, voice: str) -> Optional[str]:
    if voice == "vi" and isinstance(item.get("original_text"), str) and item["original_text"].strip():
        text = item["original_text"].strip()
        item["text"] = text
        item["normalized_text"] = text
        return text

    for key in ("normalized_text", "text"):
        val = item.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return None


def _is_json_array_manifest(path: Path) -> bool:
    with path.open("r", encoding="utf-8") as f:
        while True:
            ch = f.read(1)
            if ch == "":
                return False
            if not ch.isspace():
                return ch == "["


def process_jsonl_manifest(path: Path, runner: EspeakRunner) -> int:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    n = 0
    with path.open("r", encoding="utf-8") as fin, tmp_path.open("w", encoding="utf-8") as fout:
        for raw in fin:
            line = raw.strip()
            if not line:
                continue
            item = json.loads(line)
            text = choose_text(item, runner.voice)
            if text:
                print("Old IPA: ", item.get("ipa"))
                item["ipa"] = runner.text_to_ipa(text)
                print(f"New ipa: {item['ipa']}")
            fout.write(json.dumps(item, ensure_ascii=False))
            fout.write("\n")
            n += 1
    tmp_path.replace(path)
    return n


def process_json_array_manifest(path: Path, runner: EspeakRunner) -> int:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected list in JSON array manifest: {path}")

    n = 0
    for item in data:
        if not isinstance(item, dict):
            continue
        text = choose_text(item, runner.voice)
        if text:
            item["ipa"] = runner.text_to_ipa(text)
        n += 1

    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    tmp_path.replace(path)
    return n


def process_manifest(path: Path, runner: EspeakRunner) -> int:
    if _is_json_array_manifest(path):
        return process_json_array_manifest(path, runner)
    return process_jsonl_manifest(path, runner)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rewrite/add top-level IPA field in simple manifests in-place."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_CONFIG_PATH),
        help=f"JSON config mapping language -> list of manifest paths (default: {DEFAULT_CONFIG_PATH})",
    )
    parser.add_argument(
        "--lang",
        type=str,
        default="all",
        help="Language key from config, or 'all' (default: all).",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    config = load_config(config_path)
    langs = list(config.keys()) if args.lang == "all" else [args.lang]

    exes = find_espeak_binaries()
    primary_exe = exes[0]
    fallback_exe = exes[1] if len(exes) > 1 else None

    for lang in langs:
        if lang not in config:
            print(f"[WARN] Language '{lang}' not found in config; skipping.", file=sys.stderr)
            continue
        voice = ESPEAK_VOICE_BY_LANG.get(lang, lang)
        runner = EspeakRunner(exe=primary_exe, fallback_exe=fallback_exe, voice=voice)
        manifests = config[lang]
        print(f"[INFO] Processing lang={lang} (voice={voice}), manifests={len(manifests)}")
        for manifest in manifests:
            path = Path(manifest)
            if not path.exists():
                print(f"[WARN] Missing manifest: {path}", file=sys.stderr)
                continue
            count = process_manifest(path, runner)
            print(f"[OK] Rewrote {path} (records={count})")


if __name__ == "__main__":
    main()
