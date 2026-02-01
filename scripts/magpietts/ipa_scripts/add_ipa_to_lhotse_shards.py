#!/usr/bin/env python3
"""
Add IPA strings (from espeak/espeak-ng) to Lhotse cuts jsonl.gz shards.

For each cuts directory like:
  /Data/.../de/.../cuts
creates:
  /Data/.../de/.../cuts_with_ipa
and writes corresponding cuts.000000.jsonl.gz, etc. with an added IPA field.

IPA is added to each supervision under:
  cut["supervisions"][i]["custom"]["ipa"]

Usage:
  python add_ipa_to_cuts.py --lang de
  python add_ipa_to_cuts.py --lang all  # run all languages

Edit the `CUTS_DIRS_BY_LANG` dict below (or replace with argparse/config as desired).
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import gzip
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

# -------------------------
# USER CONFIG
# -------------------------

# Default config file path (same directory as this script)
DEFAULT_CONFIG_PATH = Path(__file__).parent / "cuts_dirs_config.json"


def load_cuts_dirs_config(config_path: Optional[Path] = None) -> Dict[str, List[str]]:
    """Load CUTS_DIRS_BY_LANG from a JSON config file."""
    if config_path is None:
        config_path = DEFAULT_CONFIG_PATH
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)

# Map your dataset language keys to espeak voice codes (adjust as needed).
# For German, espeak-ng uses "de" typically.
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
}

OUTPUT_SUFFIX = "_with_ipa"  # cuts -> cuts_with_ipa
SHARD_GLOB = "cuts.*.jsonl.gz"

# Parallelism
MAX_WORKERS = max(1, (os.cpu_count() or 4) - 1)
# MAX_WORKERS = 8

# If True, skip writing if output shard exists (basic resume)
SKIP_EXISTING_OUTPUT_SHARDS = False
# -------------------------
# IMPLEMENTATION
# -------------------------

IPA_FLAG = "--ipa"  # espeak-ng uses --ipa, espeak supports --ipa in many builds
# Use --quiet if available; safe to try.
COMMON_FLAGS = ["-q"]

# Some espeak builds output extra spaces/newlines; we normalize.
_WS_RE = re.compile(r"\s+")


def _find_espeak_binary() -> str:
    """Prefer espeak-ng if present, else espeak."""
    for exe in ("espeak-ng", "espeak"):
        if shutil.which(exe):
            return exe
    raise RuntimeError(
        "Neither 'espeak-ng' nor 'espeak' was found on PATH. "
        "Install espeak-ng (recommended) or espeak."
    )


@dataclass(frozen=True)
class EspeakRunner:
    exe: str
    voice: str

    def text_to_ipa(self, text: str) -> str:
        """
        Convert text -> IPA using espeak/espeak-ng.
        """
        # Note: We pass text via stdin to avoid shell escaping issues.
        cmd = [self.exe, "-v", self.voice, IPA_FLAG] + COMMON_FLAGS
        try:
            proc = subprocess.run(
                cmd,
                input=text.encode("utf-8"),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to run {cmd}: {e}") from e

        if proc.returncode != 0:
            raise RuntimeError(
                f"espeak command failed (rc={proc.returncode})\n"
                f"cmd: {' '.join(cmd)}\n"
                f"stderr: {proc.stderr.decode('utf-8', errors='replace')}"
            )

        out = proc.stdout.decode("utf-8", errors="replace").strip()
        # Normalize whitespace to single spaces
        out = _WS_RE.sub(" ", out).strip()
        return out


def iter_shards(cuts_dir: Path) -> List[Path]:
    return sorted(cuts_dir.glob(SHARD_GLOB))


def derive_output_dir(cuts_dir: Path) -> Path:
    # If dir name ends with "cuts", produce "cuts_with_ipa".
    # Otherwise append suffix to the directory name.
    name = cuts_dir.name
    if name == "cuts":
        out_name = f"cuts{OUTPUT_SUFFIX}"
    else:
        out_name = f"{name}{OUTPUT_SUFFIX}"
    return cuts_dir.parent / out_name


def load_json_line(line: str) -> dict:
    return json.loads(line)


def dump_json_line(obj: dict) -> str:
    # compact, consistent output
    return json.dumps(obj, ensure_ascii=False)


class IPACache:
    """
    Process-local cache. Speeds up repeated identical texts.
    """

    def __init__(self) -> None:
        self._cache: Dict[Tuple[str, str], str] = {}

    def get(self, voice: str, text: str) -> Optional[str]:
        return self._cache.get((voice, text))

    def set(self, voice: str, text: str, ipa: str) -> None:
        self._cache[(voice, text)] = ipa


def add_ipa_to_cut(
    cut: dict,
    espeak: EspeakRunner,
    cache: IPACache,
) -> dict:
    """
    Adds IPA to each supervision custom field: custom["ipa"].
    Uses supervision["custom"]["normalized_text"] if available, otherwise supervision["text"] as source text.
    For Vietnamese (vi), uses original_text and updates text/normalized_text fields.
    """
    sups = cut.get("supervisions") or []
    is_vietnamese = espeak.voice == "vi"
    for sup in sups:
        custom = sup.get("custom")
        if custom is None:
            custom = {}
            sup["custom"] = custom

        # For Vietnamese, use original_text and fix the text fields
        if is_vietnamese and custom.get("original_text"):
            text = custom["original_text"]
            sup["text"] = text
            custom["normalized_text"] = text
        else:
            text = custom.get("normalized_text") or sup.get("text")
        
        if not text:
            continue

        # If already has IPA, keep it
        if "ipa" in custom and isinstance(custom["ipa"], str) and custom["ipa"].strip():
            continue

        cached = cache.get(espeak.voice, text)
        if cached is None:
            cached = espeak.text_to_ipa(text)
            cache.set(espeak.voice, text, cached)

        custom["ipa"] = cached

    return cut


def process_shard(
    shard_path: Path,
    out_shard_path: Path,
    espeak: EspeakRunner,
) -> Tuple[Path, int]:
    """
    Read shard jsonl.gz, add IPA, write out shard jsonl.gz
    Returns: (out_shard_path, num_lines)
    """
    cache = IPACache()
    n = 0

    with gzip.open(shard_path, "rt", encoding="utf-8") as fin, gzip.open(
        out_shard_path, "wt", encoding="utf-8"
    ) as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            cut = load_json_line(line)
            cut = add_ipa_to_cut(cut, espeak=espeak, cache=cache)
            fout.write(dump_json_line(cut))
            fout.write("\n")
            n += 1

    return out_shard_path, n


def process_cuts_dir(lang: str, cuts_dir: Path) -> None:
    voice = ESPEAK_VOICE_BY_LANG.get(lang, lang)
    exe = _find_espeak_binary()
    espeak = EspeakRunner(exe=exe, voice=voice)

    out_dir = derive_output_dir(cuts_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    shards = iter_shards(cuts_dir)
    if not shards:
        print(f"[WARN] No shards matched {SHARD_GLOB} in {cuts_dir}", file=sys.stderr)
        return

    print(f"[INFO] {lang}: {cuts_dir} -> {out_dir}  (shards={len(shards)})")

    jobs: List[Tuple[Path, Path]] = []
    for shard in shards:
        out_shard = out_dir / shard.name
        if SKIP_EXISTING_OUTPUT_SHARDS and out_shard.exists():
            continue
        jobs.append((shard, out_shard))

    if not jobs:
        print(f"[INFO] {lang}: nothing to do in {cuts_dir} (all outputs exist).")
        return

    # Parallelize per shard
    with cf.ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = []
        for shard, out_shard in jobs:
            futures.append(ex.submit(_process_shard_worker, shard, out_shard, espeak.exe, espeak.voice))

        for fut in cf.as_completed(futures):
            out_shard_path, n = fut.result()
            print(f"[OK] wrote {out_shard_path}  (lines={n})")


def _process_shard_worker(shard: Path, out_shard: Path, exe: str, voice: str) -> Tuple[Path, int]:
    # Re-create runner in worker process
    espeak = EspeakRunner(exe=exe, voice=voice)
    return process_shard(shard, out_shard, espeak)


def get_available_languages(cuts_dirs: Dict[str, List[str]]) -> List[str]:
    """Return list of all available language codes."""
    return list(cuts_dirs.keys())


def process_language(lang: str, cuts_dirs: Dict[str, List[str]]) -> bool:
    """
    Process all directories for a given language.
    Returns True if successful, False if there was an issue.
    """
    if lang not in cuts_dirs:
        print(f"[ERROR] Unknown language: {lang}", file=sys.stderr)
        print(f"[ERROR] Available languages: {get_available_languages(cuts_dirs)}", file=sys.stderr)
        return False

    dirs = cuts_dirs[lang]
    for d in dirs:
        cuts_dir = Path(d)
        if not cuts_dir.exists():
            print(f"[WARN] missing dir: {cuts_dir}", file=sys.stderr)
            continue
        process_cuts_dir(lang, cuts_dir)
    
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add IPA strings to Lhotse cuts jsonl.gz shards."
    )
    parser.add_argument(
        "--lang",
        type=str,
        required=True,
        help="Language code to process (e.g., 'de', 'en', 'fr') or 'all' for all languages."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help=f"Path to JSON config file with cuts directories. Default: {DEFAULT_CONFIG_PATH}"
    )
    args = parser.parse_args()

    # Load config
    config_path = Path(args.config) if args.config else None
    cuts_dirs = load_cuts_dirs_config(config_path)
    print(f"[INFO] Loaded config with languages: {get_available_languages(cuts_dirs)}")

    if args.lang == "all":
        # Process all languages
        for lang in cuts_dirs.keys():
            print(f"\n{'='*60}")
            print(f"[INFO] Processing language: {lang}")
            print(f"{'='*60}")
            process_language(lang, cuts_dirs)
    else:
        success = process_language(args.lang, cuts_dirs)
        if not success:
            sys.exit(1)


if __name__ == "__main__":
    main()
