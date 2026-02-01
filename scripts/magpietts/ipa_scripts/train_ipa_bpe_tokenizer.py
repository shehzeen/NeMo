#!/usr/bin/env python3
"""
Train a byte-level BPE tokenizer on IPA strings from Lhotse cuts_with_ipa shards.

This script:
1. Reads IPA strings from cuts_with_ipa directories (output of add_ipa_to_lhotse_shards.py)
2. Optionally balances data across languages (samples equal amounts from each)
3. Trains a HuggingFace ByteLevelBPETokenizer on all extracted IPA strings
4. Saves vocab.json and merges.txt to the specified output directory

Features:
- Language balancing: uses the same number of samples from each language
- Configurable max samples per language

Usage:
    python train_ipa_bpe_tokenizer.py --output_dir /path/to/output --vocab_size 1024
    python train_ipa_bpe_tokenizer.py --output_dir /path/to/output --train_langs en,de --vocab_size 2048
    python train_ipa_bpe_tokenizer.py --output_dir /path/to/output --train_langs all --max_samples_per_lang 50000

The trained tokenizer can be loaded using the IPABPETokenizer class in:
    nemo/collections/common/tokenizers/text_to_speech/tts_tokenizers.py
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, Generator, List, Optional, Tuple

from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer

# -------------------------
# USER CONFIG - Same structure as add_ipa_to_lhotse_shards.py
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


OUTPUT_SUFFIX = "_with_ipa"  # cuts -> cuts_with_ipa
SHARD_GLOB = "cuts.*.jsonl.gz"


def get_ipa_dir(cuts_dir: Path) -> Path:
    """Convert a cuts directory path to its corresponding cuts_with_ipa path."""
    name = cuts_dir.name
    if name == "cuts":
        out_name = f"cuts{OUTPUT_SUFFIX}"
    elif name.endswith("_textContext"):
        # Handle cuts_textContext -> cuts_textContext_with_ipa
        out_name = f"{name}{OUTPUT_SUFFIX}"
    else:
        out_name = f"{name}{OUTPUT_SUFFIX}"
    return cuts_dir.parent / out_name


def iter_shards(ipa_dir: Path) -> List[Path]:
    """Get all shard files in a directory."""
    return sorted(ipa_dir.glob(SHARD_GLOB))


def extract_ipa_from_shard(shard_path: Path) -> Generator[str, None, None]:
    """
    Extract all IPA strings from a single shard file.

    Yields:
        IPA strings from cut["supervisions"][i]["custom"]["ipa"]
    """
    with gzip.open(shard_path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                cut = json.loads(line)
                supervisions = cut.get("supervisions", [])
                for sup in supervisions:
                    custom = sup.get("custom", {})
                    ipa = custom.get("ipa")
                    if ipa and isinstance(ipa, str) and ipa.strip():
                        yield ipa.strip()
            except json.JSONDecodeError:
                continue


def extract_ipa_from_dir(ipa_dir: Path) -> Generator[str, None, None]:
    """Extract all IPA strings from all shards in a directory."""
    shards = iter_shards(ipa_dir)
    for shard in shards:
        yield from extract_ipa_from_shard(shard)


def get_available_languages(cuts_dirs: Dict[str, List[str]]) -> List[str]:
    """Return list of all available language codes."""
    return list(cuts_dirs.keys())


def collect_ipa_strings(
    cuts_dirs: Dict[str, List[str]],
    lang: Optional[str] = None,
) -> Generator[str, None, None]:
    """
    Collect all IPA strings from the specified language(s).

    Args:
        cuts_dirs: Dictionary mapping language codes to lists of cuts directories
        lang: Language code or None for all languages.

    Yields:
        IPA strings
    """
    if lang is None or lang == "all":
        langs_to_process = list(cuts_dirs.keys())
    else:
        if lang not in cuts_dirs:
            raise ValueError(f"Unknown language: {lang}. Available: {get_available_languages(cuts_dirs)}")
        langs_to_process = [lang]

    for lang_code in langs_to_process:
        print(f"[INFO] Processing language: {lang_code}")
        for cuts_dir_str in cuts_dirs[lang_code]:
            cuts_dir = Path(cuts_dir_str)
            ipa_dir = get_ipa_dir(cuts_dir)

            if not ipa_dir.exists():
                print(f"[WARN] IPA directory does not exist: {ipa_dir}", file=sys.stderr)
                continue

            print(f"[INFO] Reading from: {ipa_dir}")
            count = 0
            for ipa in extract_ipa_from_dir(ipa_dir):
                yield ipa
                count += 1
            print(f"[INFO] Extracted {count} IPA strings from {ipa_dir}")


def iter_ipa_strings_for_lang(
    lang: str,
    cuts_dirs: Dict[str, List[str]],
) -> Generator[str, None, None]:
    """Iterate over all IPA strings for a single language (memory-efficient)."""
    if lang not in cuts_dirs:
        return

    for cuts_dir_str in cuts_dirs[lang]:
        cuts_dir = Path(cuts_dir_str)
        ipa_dir = get_ipa_dir(cuts_dir)

        if not ipa_dir.exists():
            continue

        for ipa in extract_ipa_from_dir(ipa_dir):
            yield ipa


def count_ipa_strings_for_lang(lang: str, cuts_dirs: Dict[str, List[str]], max_count: int = 100000) -> int:
    """Count IPA strings for a language without loading into memory."""
    count = 0
    for _ in iter_ipa_strings_for_lang(lang, cuts_dirs):
        count += 1
        if count >= max_count:
            break
    return count


def simple_sample_ipa_strings(
    lang: str,
    cuts_dirs: Dict[str, List[str]],
    k: int,
    max_collect: int = 100000,
    seed: int = 42,
) -> List[str]:
    """
    Simple sampling: collect up to max_collect IPA strings, then randomly sample k.

    This avoids reading through all data like reservoir sampling does.

    Args:
        lang: Language code
        cuts_dirs: Dictionary mapping language codes to lists of cuts directories
        k: Number of samples to select
        max_collect: Maximum number of strings to collect before sampling
        seed: Random seed for reproducibility

    Returns:
        List of up to k sampled IPA strings
    """
    rng = random.Random(seed)
    collected: List[str] = []

    for ipa in iter_ipa_strings_for_lang(lang, cuts_dirs):
        collected.append(ipa)
        if len(collected) >= max_collect:
            break

    # If we have fewer than k, return all
    if len(collected) <= k:
        return collected

    # Otherwise, randomly sample k
    return rng.sample(collected, k)


def parse_langs_arg(arg: str, available_langs: List[str]) -> List[str]:
    """Parse a language argument (comma-separated or 'all')."""
    if arg == "all":
        return available_langs
    langs = [l.strip() for l in arg.split(",") if l.strip()]
    for lang in langs:
        if lang not in available_langs:
            raise ValueError(f"Unknown language: {lang}. Available: {available_langs}")
    return langs


def create_balanced_corpus(
    train_langs: List[str],
    cuts_dirs: Dict[str, List[str]],
    output_file: str,
    max_samples_per_lang: Optional[int] = None,
    max_count_per_lang: int = 100000,
    seed: int = 42,
) -> Tuple[str, Dict[str, int]]:
    """
    Create a balanced IPA corpus file with equal samples from each language.

    Uses a memory-efficient two-pass approach:
    1. First pass: Count sentences per language (up to max_count_per_lang)
    2. Second pass: Use simple sampling to select samples

    Args:
        train_langs: List of language codes to include
        cuts_dirs: Dictionary mapping language codes to lists of cuts directories
        output_file: Path to write the balanced corpus
        max_samples_per_lang: Optional cap on samples per language
        max_count_per_lang: Max count per language when counting IPA strings
        seed: Random seed for reproducibility

    Returns:
        Tuple of (corpus_file_path, dict of lang -> actual_count)
    """
    # First pass: Count sentences per language
    print("[INFO] Pass 1: Counting IPA strings per language...")
    lang_counts: Dict[str, int] = {}

    for lang in train_langs:
        if lang not in cuts_dirs:
            print(f"[WARN] Language {lang} not in config, skipping")
            continue
        print(f"[INFO] Counting {lang}...", end=" ", flush=True)
        count = count_ipa_strings_for_lang(lang, cuts_dirs, max_count_per_lang)
        lang_counts[lang] = count
        print(f"{count} IPA strings")

    if not lang_counts:
        raise ValueError("No IPA strings found for any language")

    # Find minimum count across languages
    min_count = min(lang_counts.values())
    print(f"[INFO] Minimum count across languages: {min_count}")

    # Apply max_samples_per_lang cap if specified
    samples_per_lang = min_count
    if max_samples_per_lang is not None and max_samples_per_lang < min_count:
        samples_per_lang = max_samples_per_lang
        print(f"[INFO] Using max_samples_per_lang cap: {samples_per_lang}")

    # Second pass: Sample from each language using simple sampling
    print(f"[INFO] Pass 2: Sampling {samples_per_lang} strings per language...")
    actual_counts: Dict[str, int] = {}
    total_written = 0

    with open(output_file, "w", encoding="utf-8") as f:
        for lang in lang_counts.keys():
            print(f"[INFO] Sampling from {lang}...", end=" ", flush=True)
            # Use different seed per language for variety, but reproducible
            lang_seed = seed + hash(lang) % 10000
            sampled = simple_sample_ipa_strings(lang, cuts_dirs, samples_per_lang, max_count_per_lang, lang_seed)

            for ipa in sampled:
                f.write(ipa + "\n")
                total_written += 1

            actual_counts[lang] = len(sampled)
            print(f"sampled {len(sampled)} strings")

    print(f"[INFO] Total IPA strings written to corpus: {total_written}")
    print(f"[INFO] Balanced corpus saved to: {output_file}")

    return output_file, actual_counts


def train_bpe_tokenizer(
    corpus_file: str,
    vocab_size: int = 1024,
    min_frequency: int = 2,
    output_dir: str = "./ipa_bpe_tokenizer",
) -> Tokenizer:
    """
    Train a byte-level BPE tokenizer on IPA strings from a corpus file.

    Args:
        corpus_file: Path to the IPA corpus file (one IPA string per line)
        vocab_size: Target vocabulary size
        min_frequency: Minimum frequency for a token to be included
        output_dir: Directory to save the tokenizer files

    Returns:
        Trained Tokenizer object
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Check if tokenizer already exists
    tokenizer_path = os.path.join(output_dir, "tokenizer.json")
    if os.path.exists(tokenizer_path):
        print(f"[INFO] Loading existing tokenizer from {tokenizer_path}")
        return Tokenizer.from_file(tokenizer_path)

    # Count lines in corpus
    with open(corpus_file, "r", encoding="utf-8") as f:
        total_count = sum(1 for _ in f)
    print(f"[INFO] Corpus contains {total_count} IPA strings")

    if total_count == 0:
        raise ValueError("Corpus file is empty. Make sure the cuts_with_ipa directories exist.")

    # Initialize a byte-level BPE tokenizer
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))

    # Use byte-level pre-tokenization (like GPT-2)
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)

    # Add byte-level decoder to properly convert back to original text
    tokenizer.decoder = ByteLevelDecoder()

    # Define special tokens
    special_tokens = ["<pad>", "<blank>", "<unk>"]

    # Create trainer
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=special_tokens,
        show_progress=True,
    )

    # Train the tokenizer
    print(f"[INFO] Training BPE tokenizer with vocab_size={vocab_size}, min_frequency={min_frequency}")
    tokenizer.train(files=[corpus_file], trainer=trainer)

    # Save the tokenizer
    vocab_path = os.path.join(output_dir, "vocab.json")
    merges_path = os.path.join(output_dir, "merges.txt")

    # Save using the tokenizer's model save method
    tokenizer.model.save(output_dir)

    # Also save the full tokenizer for easy loading
    tokenizer.save(tokenizer_path)

    print(f"[INFO] Tokenizer saved to: {output_dir}")
    print(f"[INFO]   - vocab.json: {vocab_path}")
    print(f"[INFO]   - merges.txt: {merges_path}")
    print(f"[INFO]   - tokenizer.json: {tokenizer_path}")
    print(f"[INFO] Vocabulary size: {tokenizer.get_vocab_size()}")

    return tokenizer


def main():
    parser = argparse.ArgumentParser(
        description="Train a byte-level BPE tokenizer on IPA strings from Lhotse cuts_with_ipa shards."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save the trained tokenizer files (vocab.json, merges.txt, tokenizer.json)",
    )
    parser.add_argument(
        "--vocab_size",
        type=int,
        default=1024,
        help="Vocabulary size for the BPE tokenizer (default: 1024)",
    )
    parser.add_argument(
        "--min_frequency",
        type=int,
        default=2,
        help="Minimum frequency for a token to be included in vocabulary (default: 2)",
    )
    parser.add_argument(
        "--train_langs",
        type=str,
        default="all",
        help="Comma-separated language codes for training, or 'all' (default: all)",
    )
    parser.add_argument(
        "--max_samples_per_lang",
        type=int,
        default=None,
        help="Optional cap on samples per language (default: use min count across langs for balance)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling (default: 42)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help=f"Path to JSON config file with cuts directories. Default: {DEFAULT_CONFIG_PATH}",
    )
    parser.add_argument(
        "--max_count_per_lang",
        type=int,
        default=100000,
        help="Max count per language when counting IPA strings (default: 100000)",
    )
    args = parser.parse_args()

    # Load config
    config_path = Path(args.config) if args.config else None
    cuts_dirs = load_cuts_dirs_config(config_path)
    available_langs = get_available_languages(cuts_dirs)

    # Parse train_langs
    try:
        train_langs = parse_langs_arg(args.train_langs, available_langs)
    except ValueError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    print(f"[INFO] Training IPA BPE tokenizer")
    print(f"[INFO]   Output directory: {args.output_dir}")
    print(f"[INFO]   Vocabulary size: {args.vocab_size}")
    print(f"[INFO]   Min frequency: {args.min_frequency}")
    print(f"[INFO]   Training languages: {train_langs}")
    print(f"[INFO]   Max samples per lang: {args.max_samples_per_lang or 'auto (min across langs)'}")
    print(f"[INFO]   Max count per lang: {args.max_count_per_lang}")
    print(f"[INFO]   Available languages: {available_langs}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Step 1: Create balanced corpus
    print("\n" + "=" * 60)
    print("STEP 1: Creating balanced IPA corpus")
    print("=" * 60)

    corpus_file = os.path.join(args.output_dir, "ipa_corpus_balanced.txt")

    if os.path.exists(corpus_file):
        print(f"[INFO] Using existing corpus file: {corpus_file}")
        with open(corpus_file, "r", encoding="utf-8") as f:
            line_count = sum(1 for _ in f)
        print(f"[INFO] Corpus contains {line_count} IPA strings")
    else:
        corpus_file, lang_counts = create_balanced_corpus(
            train_langs=train_langs,
            cuts_dirs=cuts_dirs,
            output_file=corpus_file,
            max_samples_per_lang=args.max_samples_per_lang,
            max_count_per_lang=args.max_count_per_lang,
            seed=args.seed,
        )

    # Step 2: Train tokenizer
    print("\n" + "=" * 60)
    print("STEP 2: Training BPE tokenizer")
    print("=" * 60)

    tokenizer = train_bpe_tokenizer(
        corpus_file=corpus_file,
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        output_dir=args.output_dir,
    )

    # Test the tokenizer
    print("\n[INFO] Testing tokenizer with sample IPA strings:")
    test_strings = [
        "həˈloʊ wɜːld",  # hello world
        "ˈaɪ pʰiː eɪ",  # IPA
        "ˈtɛstɪŋ wʌn tuː θriː",  # testing one two three
    ]
    for test_str in test_strings:
        encoded = tokenizer.encode(test_str)
        decoded = tokenizer.decode(encoded.ids)
        print(f"  Input:   '{test_str}'")
        print(f"  Tokens:  {encoded.tokens}")
        print(f"  IDs:     {encoded.ids}")
        print(f"  Decoded: '{decoded}'")
        print()

    print("[INFO] Done!")


if __name__ == "__main__":
    main()
