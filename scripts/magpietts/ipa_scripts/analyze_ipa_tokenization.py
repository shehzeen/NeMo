#!/usr/bin/env python3
"""
Analyze and compare tokenization (tokens per second of audio) between:
1. Qwen/Qwen2.5-1.5B-Instruct tokenizer on raw text
2. NVIDIA Nemotron Nano 30B tokenizer on raw text
3. IPABPETokenizer on phonemized IPA text at different vocab sizes

This script:
1. Creates a balanced IPA corpus (equal samples per language) from train_langs
2. Trains IPA BPE tokenizers at vocab sizes 512, 1024, 2048, 4096
3. For each test language, samples text pairs from cuts_with_ipa directories
4. Computes tokens per second (tokens / audio duration) for each tokenizer
5. Outputs comparison statistics showing tokens/second for each tokenizer

Features:
- Reads data once and reuses across all vocab sizes (efficient)
- Balances training data across languages (uses min count across all train langs)
- Supports separate train and test language sets
- Computes tokens per second using audio duration from cuts

Usage:
    # Train and test on all languages
    python analyze_ipa_tokenization.py --output_dir /path/to/output

    # Train on en,de,fr but test on all languages
    python analyze_ipa_tokenization.py --output_dir /path/to/output --train_langs en,de,fr --test_langs all

    # Train on all, test on specific languages
    python analyze_ipa_tokenization.py --output_dir /path/to/output --train_langs all --test_langs en,zh

    # Cap training samples per language
    python analyze_ipa_tokenization.py --output_dir /path/to/output --max_samples_per_lang 50000
"""

import argparse
import gzip
import json
import os
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Generator, List, Optional, Tuple

import numpy as np
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer
from transformers import AutoTokenizer

# -------------------------
# CONFIGURATION
# -------------------------

VOCAB_SIZES = [512, 1024, 2048, 4096]

# Default config file path (same directory as this script)
DEFAULT_CONFIG_PATH = Path(__file__).parent / "cuts_all_no_ja.json"


def load_cuts_dirs_config(config_path: Optional[Path] = None) -> Dict[str, List[str]]:
    """Load CUTS_DIRS_BY_LANG from a JSON config file."""
    if config_path is None:
        config_path = DEFAULT_CONFIG_PATH
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)

OUTPUT_SUFFIX = "_with_ipaNG"
SHARD_GLOB = "cuts.*.jsonl.gz"


@dataclass
class TextPair:
    """A pair of raw text and its IPA phonemization with audio duration."""
    raw_text: str
    ipa_text: str
    lang: str
    duration: float  # audio duration in seconds


@dataclass
class TokenizationStats:
    """Statistics for tokenization comparison (tokens per second)."""
    lang: str
    num_samples: int
    total_duration: float  # sum of all durations in seconds
    qwen_tokens_per_second: float
    nemotron_tokens_per_second: float
    ipa_tokens_per_second: Dict[int, float]  # vocab_size -> tokens/sec


def get_ipa_dir(cuts_dir: Path) -> Path:
    """Convert a cuts directory path to its corresponding cuts_with_ipa path."""
    name = cuts_dir.name
    if name == "cuts":
        out_name = f"cuts{OUTPUT_SUFFIX}"
    else:
        out_name = f"{name}{OUTPUT_SUFFIX}"
    return cuts_dir.parent / out_name


def iter_shards(ipa_dir: Path) -> List[Path]:
    """Get all shard files in a directory."""
    return sorted(ipa_dir.glob(SHARD_GLOB))


def extract_text_pairs_from_shard(shard_path: Path, lang: str) -> Generator[TextPair, None, None]:
    """
    Extract text pairs (raw text + IPA) from a single shard file.
    
    Yields:
        TextPair objects with raw_text, ipa_text, and duration
    """
    with gzip.open(shard_path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                cut = json.loads(line)
                # Get duration from the top-level cut object
                duration = cut.get("duration", 0.0)
                supervisions = cut.get("supervisions", [])
                for sup in supervisions:
                    custom = sup.get("custom", {})
                    ipa = custom.get("ipa")
                    # Get raw text - prefer normalized_text, fallback to text
                    raw_text = custom.get("normalized_text") or sup.get("text")
                    
                    if ipa and raw_text and isinstance(ipa, str) and isinstance(raw_text, str):
                        ipa = ipa.strip()
                        raw_text = raw_text.strip()
                        if ipa and raw_text and duration > 0:
                            yield TextPair(raw_text=raw_text, ipa_text=ipa, lang=lang, duration=duration)
            except json.JSONDecodeError:
                continue


def sample_text_pairs(
    lang: str,
    cuts_dirs: Dict[str, List[str]],
    num_samples: int = 1000,
    seed: int = 42,
) -> List[TextPair]:
    """
    Sample text pairs from a language's cuts_with_ipa directories.
    
    Args:
        lang: Language code
        cuts_dirs: Dictionary mapping language codes to lists of cuts directories
        num_samples: Number of samples to collect
        seed: Random seed for reproducibility
    
    Returns:
        List of TextPair objects
    """
    random.seed(seed)
    
    if lang not in cuts_dirs:
        raise ValueError(f"Unknown language: {lang}")
    
    # Collect all text pairs from all directories
    all_pairs = []
    for cuts_dir_str in cuts_dirs[lang]:
        cuts_dir = Path(cuts_dir_str)
        ipa_dir = get_ipa_dir(cuts_dir)
        
        if not ipa_dir.exists():
            print(f"[WARN] IPA directory does not exist: {ipa_dir}", file=sys.stderr)
            continue
        
        shards = iter_shards(ipa_dir)
        for shard in shards:
            for pair in extract_text_pairs_from_shard(shard, lang):
                all_pairs.append(pair)
                # Early exit if we have way more than needed
                if len(all_pairs) >= num_samples * 10:
                    break
            if len(all_pairs) >= num_samples * 10:
                break
        if len(all_pairs) >= num_samples * 10:
            break
    
    # Sample
    if len(all_pairs) <= num_samples:
        print(f"[INFO] {lang}: Only found {len(all_pairs)} pairs, using all")
        return all_pairs
    
    return random.sample(all_pairs, num_samples)


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
        
        shards = iter_shards(ipa_dir)
        for shard in shards:
            with gzip.open(shard, "rt", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        cut = json.loads(line)
                        for sup in cut.get("supervisions", []):
                            ipa = sup.get("custom", {}).get("ipa")
                            if ipa and isinstance(ipa, str) and ipa.strip():
                                yield ipa.strip()
                    except json.JSONDecodeError:
                        continue


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
    if max_samples_per_lang is not None:
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


def train_ipa_bpe_tokenizer(
    output_dir: str,
    vocab_size: int,
    corpus_file: str,
    min_frequency: int = 2,
) -> Tokenizer:
    """
    Train a byte-level BPE tokenizer on IPA strings from a pre-built corpus file.
    
    Args:
        output_dir: Directory to save tokenizer files
        vocab_size: Target vocabulary size
        corpus_file: Path to the IPA corpus file (one IPA string per line)
        min_frequency: Minimum frequency for a token to be included
    
    Returns:
        Trained Tokenizer object
    """
    tokenizer_dir = os.path.join(output_dir, f"ipa_bpe_v{vocab_size}")
    os.makedirs(tokenizer_dir, exist_ok=True)
    
    tokenizer_file = os.path.join(tokenizer_dir, "tokenizer.json")
    
    # Check if already trained
    if os.path.exists(tokenizer_file):
        print(f"[INFO] Loading existing tokenizer from {tokenizer_file}")
        return Tokenizer.from_file(tokenizer_file)
    
    # Initialize tokenizer
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    
    special_tokens = ["<pad>", "<blank>", "<unk>"]
    
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=special_tokens,
        show_progress=True,
    )
    
    print(f"[INFO] Training BPE tokenizer with vocab_size={vocab_size}...")
    tokenizer.train(files=[corpus_file], trainer=trainer)
    
    # Save
    tokenizer.save(tokenizer_file)
    tokenizer.model.save(tokenizer_dir)
    
    print(f"[INFO] Saved tokenizer to {tokenizer_dir}")
    
    return tokenizer


def compute_stats(
    text_pairs: List[TextPair],
    qwen_tokenizer: AutoTokenizer,
    nemotron_tokenizer: AutoTokenizer,
    ipa_tokenizers: Dict[int, Tokenizer],
    lang: str,
) -> TokenizationStats:
    """
    Compute tokenization statistics (tokens per second) for a set of text pairs.
    """
    qwen_counts = []
    nemotron_counts = []
    ipa_counts = {vs: [] for vs in ipa_tokenizers.keys()}
    
    for pair in text_pairs:
        # Qwen tokenizer on raw text
        qwen_tokens = qwen_tokenizer.encode(pair.raw_text)
        qwen_counts.append(len(qwen_tokens))
        
        # Nemotron tokenizer on raw text
        nemotron_tokens = nemotron_tokenizer.encode(pair.raw_text)
        nemotron_counts.append(len(nemotron_tokens))
        
        # IPA tokenizers on IPA text
        for vocab_size, tokenizer in ipa_tokenizers.items():
            ipa_tokens = tokenizer.encode(pair.ipa_text)
            ipa_counts[vocab_size].append(len(ipa_tokens.ids))
    
    # Calculate total duration and token counts
    total_duration = sum(pair.duration for pair in text_pairs)
    qwen_total = sum(qwen_counts)
    nemotron_total = sum(nemotron_counts)
    
    # Compute tokens per second
    qwen_tps = qwen_total / total_duration if total_duration > 0 else 0.0
    nemotron_tps = nemotron_total / total_duration if total_duration > 0 else 0.0
    
    ipa_tps = {}
    for vocab_size in ipa_tokenizers.keys():
        ipa_total = sum(ipa_counts[vocab_size])
        ipa_tps[vocab_size] = ipa_total / total_duration if total_duration > 0 else 0.0
    
    return TokenizationStats(
        lang=lang,
        num_samples=len(text_pairs),
        total_duration=total_duration,
        qwen_tokens_per_second=qwen_tps,
        nemotron_tokens_per_second=nemotron_tps,
        ipa_tokens_per_second=ipa_tps,
    )


def print_stats_table(all_stats: List[TokenizationStats], vocab_sizes: List[int]):
    """Print a formatted table of tokens per second statistics."""
    print("\n" + "=" * 120)
    print("TOKENS PER SECOND: Qwen2.5-1.5B-Instruct & Nemotron Nano 30B (raw text) vs IPA BPE (phonemized)")
    print("=" * 120)
    
    # Header
    header = f"{'Lang':<6} {'Samples':>8} {'Duration(s)':>12} {'Qwen tok/s':>12} {'Nemo tok/s':>12}"
    for vs in vocab_sizes:
        header += f" {'IPA-' + str(vs):>10}"
    print(header)
    print("-" * 120)
    
    # Data rows
    for stats in all_stats:
        row = f"{stats.lang:<6} {stats.num_samples:>8} {stats.total_duration:>12.2f} {stats.qwen_tokens_per_second:>12.2f} {stats.nemotron_tokens_per_second:>12.2f}"
        for vs in vocab_sizes:
            row += f" {stats.ipa_tokens_per_second[vs]:>10.2f}"
        print(row)
    
    # Aggregated stats
    print("-" * 120)
    total_samples = sum(s.num_samples for s in all_stats)
    total_duration = sum(s.total_duration for s in all_stats)
    
    # Compute overall tokens per second (weighted by duration)
    total_qwen_tokens = sum(s.qwen_tokens_per_second * s.total_duration for s in all_stats)
    total_nemotron_tokens = sum(s.nemotron_tokens_per_second * s.total_duration for s in all_stats)
    overall_qwen_tps = total_qwen_tokens / total_duration if total_duration > 0 else 0
    overall_nemotron_tps = total_nemotron_tokens / total_duration if total_duration > 0 else 0
    
    agg_row = f"{'TOTAL':<6} {total_samples:>8} {total_duration:>12.2f} {overall_qwen_tps:>12.2f} {overall_nemotron_tps:>12.2f}"
    for vs in vocab_sizes:
        total_ipa_tokens = sum(s.ipa_tokens_per_second[vs] * s.total_duration for s in all_stats)
        overall_ipa_tps = total_ipa_tokens / total_duration if total_duration > 0 else 0
        agg_row += f" {overall_ipa_tps:>10.2f}"
    print(agg_row)
    print("=" * 120)
    
    # Summary
    print("\nSUMMARY:")
    print(f"  - Total samples analyzed: {total_samples}")
    print(f"  - Total audio duration: {total_duration:.2f} seconds ({total_duration/3600:.2f} hours)")
    print(f"  - Qwen tokens/second: {overall_qwen_tps:.2f}")
    print(f"  - Nemotron tokens/second: {overall_nemotron_tps:.2f}")
    for vs in vocab_sizes:
        total_ipa_tokens = sum(s.ipa_tokens_per_second[vs] * s.total_duration for s in all_stats)
        overall_ipa_tps = total_ipa_tokens / total_duration if total_duration > 0 else 0
        print(f"  - IPA-{vs} tokens/second: {overall_ipa_tps:.2f}")
    print()


def save_results_json(
    all_stats: List[TokenizationStats],
    output_path: str,
    train_langs: Optional[List[str]] = None,
    test_langs: Optional[List[str]] = None,
):
    """Save results to JSON file with metadata."""
    output = {
        "metadata": {
            "train_langs": train_langs or [],
            "test_langs": test_langs or [],
        },
        "results": [],
    }
    
    for stats in all_stats:
        output["results"].append({
            "lang": stats.lang,
            "num_samples": stats.num_samples,
            "total_duration_seconds": stats.total_duration,
            "qwen_tokens_per_second": stats.qwen_tokens_per_second,
            "nemotron_tokens_per_second": stats.nemotron_tokens_per_second,
            "ipa_tokens_per_second": {
                str(vs): stats.ipa_tokens_per_second[vs]
                for vs in stats.ipa_tokens_per_second.keys()
            }
        })
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"[INFO] Saved results to {output_path}")


def parse_lang_arg(arg: str, available_langs: List[str]) -> List[str]:
    """Parse a language argument (comma-separated or 'all')."""
    if arg == "all":
        return available_langs
    langs = [l.strip() for l in arg.split(",") if l.strip()]
    # Validate languages
    for lang in langs:
        if lang not in available_langs:
            raise ValueError(f"Unknown language: {lang}. Available: {available_langs}")
    return langs


def main():
    parser = argparse.ArgumentParser(
        description="Compare tokenization between Qwen and IPA BPE tokenizers."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save tokenizers and results",
    )
    parser.add_argument(
        "--samples_per_lang",
        type=int,
        default=1000,
        help="Number of samples per language for testing (default: 1000)",
    )
    parser.add_argument(
        "--train_langs",
        type=str,
        default="all",
        help="Comma-separated languages for training tokenizer, or 'all' (default: all)",
    )
    parser.add_argument(
        "--test_langs",
        type=str,
        default="all",
        help="Comma-separated languages for testing/analysis, or 'all' (default: all)",
    )
    parser.add_argument(
        "--max_samples_per_lang",
        type=int,
        default=None,
        help="Optional cap on training samples per language (default: use min count across langs)",
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
        help=f"Path to JSON config file with cuts directories. Default: {DEFAULT_CONFIG_PATH}"
    )
    parser.add_argument(
        "--max_count_per_lang",
        type=int,
        default=100000,
        help="Max count per language when counting IPA strings (default: 100000)",
    )
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load config
    config_path = Path(args.config) if args.config else None
    cuts_dirs = load_cuts_dirs_config(config_path)
    available_langs = list(cuts_dirs.keys())
    print(f"[INFO] Loaded config with languages: {available_langs}")
    
    # Parse train and test languages
    try:
        train_langs = parse_lang_arg(args.train_langs, available_langs)
        test_langs = parse_lang_arg(args.test_langs, available_langs)
    except ValueError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)
    
    print(f"[INFO] Training languages: {train_langs}")
    print(f"[INFO] Testing languages: {test_langs}")
    print(f"[INFO] Samples per language for testing: {args.samples_per_lang}")
    print(f"[INFO] Max samples per language for training: {args.max_samples_per_lang or 'auto (min across langs)'}")
    print(f"[INFO] Vocab sizes: {VOCAB_SIZES}")
    
    # Step 1: Create balanced IPA corpus once
    print("\n" + "=" * 60)
    print("STEP 1: Creating balanced IPA corpus")
    print("=" * 60)
    
    corpus_file = os.path.join(args.output_dir, "ipa_corpus_balanced.txt")
    
    # Check if corpus already exists
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
    
    # Step 2: Train IPA BPE tokenizers at different vocab sizes (reusing corpus)
    print("\n" + "=" * 60)
    print("STEP 2: Training IPA BPE tokenizers")
    print("=" * 60)
    
    ipa_tokenizers = {}
    for vocab_size in VOCAB_SIZES:
        print(f"\n[INFO] Training tokenizer with vocab_size={vocab_size}")
        ipa_tokenizers[vocab_size] = train_ipa_bpe_tokenizer(
            output_dir=args.output_dir,
            vocab_size=vocab_size,
            corpus_file=corpus_file,
            min_frequency=2,
        )
    
    # Step 3: Load Qwen and Nemotron tokenizers
    print("\n" + "=" * 60)
    print("STEP 3: Loading Qwen and Nemotron tokenizers")
    print("=" * 60)
    
    print("[INFO] Loading Qwen/Qwen2.5-1.5B-Instruct tokenizer...")
    qwen_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
    print(f"[INFO] Qwen tokenizer vocab size: {qwen_tokenizer.vocab_size}")
    
    print("[INFO] Loading nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 tokenizer...")
    
    nemotron_tokenizer =  AutoTokenizer.from_pretrained("nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16", trust_remote_code=True)
    
    print(f"[INFO] Nemotron tokenizer vocab size: {nemotron_tokenizer.vocab_size}")
    
    # Step 4: Sample text pairs and compute statistics (on test languages)
    print("\n" + "=" * 60)
    print("STEP 4: Sampling and analyzing (test languages)")
    print("=" * 60)
    
    all_stats = []
    for lang in test_langs:
        print(f"\n[INFO] Processing language: {lang}")
        
        # Sample text pairs
        text_pairs = sample_text_pairs(lang, cuts_dirs, args.samples_per_lang, args.seed)
        
        if not text_pairs:
            print(f"[WARN] No text pairs found for {lang}, skipping")
            continue
        
        print(f"[INFO] Sampled {len(text_pairs)} text pairs for {lang}")
        
        # Compute stats
        stats = compute_stats(text_pairs, qwen_tokenizer, nemotron_tokenizer, ipa_tokenizers, lang)
        all_stats.append(stats)
        
        # Print intermediate results
        print(f"[INFO] {lang}: duration={stats.total_duration:.2f}s, Qwen={stats.qwen_tokens_per_second:.2f} tok/s, Nemotron={stats.nemotron_tokens_per_second:.2f} tok/s")
        for vs in VOCAB_SIZES:
            print(f"       IPA-{vs}={stats.ipa_tokens_per_second[vs]:.2f} tok/s")
    
    # Step 5: Print and save results
    print("\n" + "=" * 60)
    print("STEP 5: Results")
    print("=" * 60)
    
    print_stats_table(all_stats, VOCAB_SIZES)
    
    # Save to JSON with metadata
    results_path = os.path.join(args.output_dir, "tokenization_comparison.json")
    save_results_json(all_stats, results_path, train_langs, test_langs)
    
    print("[INFO] Done!")


if __name__ == "__main__":
    main()
