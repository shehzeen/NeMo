#!/usr/bin/env python3
"""
Analyze and compare tokenization between:
1. Qwen/Qwen2.5-1.5B-Instruct tokenizer on raw text
2. IPABPETokenizer on phonemized IPA text at different vocab sizes

This script:
1. Trains IPA BPE tokenizers at vocab sizes 512, 1024, and 2048
2. For each language, samples 1000 sentences from cuts_with_ipa directories
3. Computes token counts using both Qwen tokenizer (on raw text) and IPA BPE tokenizer (on IPA)
4. Outputs comparison statistics

Usage:
    python analyze_ipa_tokenization.py --output_dir /path/to/output
    python analyze_ipa_tokenization.py --output_dir /path/to/output --lang en
    python analyze_ipa_tokenization.py --output_dir /path/to/output --samples_per_lang 500
"""

from __future__ import annotations

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

VOCAB_SIZES = [512, 1024, 2048]

CUTS_DIRS_BY_LANG: Dict[str, List[str]] = {
    "de": ["/Data/tts_lhotse_datasets/speech_data/de/cmltts_de_train/cuts"],
    "es": [
        "/Data/tts_lhotse_datasets/speech_data/es/cmltts_es_train/cuts",
        "/Data/tts_lhotse_datasets/speech_data/es/riva_ES_RubbyCarlos/cuts",
        "/Data/tts_lhotse_datasets/speech_data/es/riva_ES_RubbyCarlos/cuts_textContext",
    ],
    "fr": [
        "/Data/tts_lhotse_datasets/speech_data/fr/cmltts_fr_train/cuts",
        "/Data/tts_lhotse_datasets/speech_data/fr/riva_FR_VirginieSamy/cuts",
        "/Data/tts_lhotse_datasets/speech_data/fr/riva_FR_VirginieSamy/cuts_textContext",
    ],
    # "hi": [
    #     "/Data/tts_lhotse_datasets/speech_data/hi/nvyt_hi/filter_1/cuts",
    #     "/Data/tts_lhotse_datasets/speech_data/hi/nvyt_hi/filter_2/cuts",
    #     "/Data/tts_lhotse_datasets/speech_data/hi/nvyt_hi_2/filter_1/cuts",
    #     "/Data/tts_lhotse_datasets/speech_data/hi/nvyt_hi_2/filter_2/cuts",
    # ],
    "it": ["/Data/tts_lhotse_datasets/speech_data/it/cmltts_it_train/cuts"],
    "vi": [
        "/Data/tts_lhotse_datasets/speech_data/vi/Infore1_2_lsvsc/cuts",
        "/Data/tts_lhotse_datasets/speech_data/vi/Long_ContextAudio/cuts",
        "/Data/tts_lhotse_datasets/speech_data/vi/Long_ContextAudio/cuts_textContext",
        "/Data/tts_lhotse_datasets/speech_data/vi/NorthFemale/cuts",
        "/Data/tts_lhotse_datasets/speech_data/vi/NorthFemale/cuts_textContext",
        "/Data/tts_lhotse_datasets/speech_data/vi/nvyt_vi/nvyt_yt12k/cuts",
        "/Data/tts_lhotse_datasets/speech_data/vi/nvyt_vi/nvyt_yt2025/cuts",
    ],
    # "zh": [
    #     "/Data/tts_lhotse_datasets/speech_data/zh/riva_ZH_SiweiHouZhen/cuts",
    #     "/Data/tts_lhotse_datasets/speech_data/zh/riva_ZH_SiweiHouZhen/cuts_textContext",
    #     "/Data/tts_lhotse_datasets/speech_data/zh/nvyt_zh/filter_1/cuts",
    #     "/Data/tts_lhotse_datasets/speech_data/zh/nvyt_zh/filter_2/cuts",
    # ],
    # "en": [
    #     "/Data/tts_lhotse_datasets/speech_data/en/nvyt2505/lhotse_shar_shuffle_shardSize256/cuts",
    #     "/Data/tts_lhotse_datasets/speech_data/en/hifitts/lhotse_shar_shuffle_shardSize256/cuts",
    #     "/Data/tts_lhotse_datasets/speech_data/en/hifitts2/lhotse_shar_shuffle_shardSize256/cuts",
    #     "/Data/tts_lhotse_datasets/speech_data/en/jhsdGtc20Amp20Keynote/lhotse_shar_shuffle_shardSize256/cuts",
    #     "/Data/tts_lhotse_datasets/speech_data/en/libritts/lhotse_shar_shuffle_shardSize256/cuts",
    #     "/Data/tts_lhotse_datasets/speech_data/en/rivaLindyRodney/lhotse_shar_shuffle_shardSize256/cuts",
    #     "/Data/tts_lhotse_datasets/speech_data/en/rivaLindyRodney/lhotse_shar_shuffle_shardSize256/cuts_textContext",
    #     "/Data/tts_lhotse_datasets/speech_data/en/rivaEmmaMeganSeanTom/lhotse_shar_shuffle_shardSize256/cuts",
    #     "/Data/tts_lhotse_datasets/speech_data/en/rivaEmmaMeganSeanTom/lhotse_shar_shuffle_shardSize256/cuts_textContext",
    #     "/Data/tts_lhotse_datasets/speech_data/en/jhsdGtc20Amp20Keynote/lhotse_shar_shuffle_shardSize256/cuts_textContext",
    # ],
}

OUTPUT_SUFFIX = "_with_ipa"
SHARD_GLOB = "cuts.*.jsonl.gz"


@dataclass
class TextPair:
    """A pair of raw text and its IPA phonemization."""
    raw_text: str
    ipa_text: str
    lang: str


@dataclass
class TokenizationStats:
    """Statistics for tokenization comparison."""
    lang: str
    num_samples: int
    qwen_tokens_mean: float
    qwen_tokens_std: float
    qwen_tokens_total: int
    ipa_tokens_mean: Dict[int, float]  # vocab_size -> mean
    ipa_tokens_std: Dict[int, float]   # vocab_size -> std
    ipa_tokens_total: Dict[int, int]   # vocab_size -> total
    compression_ratio: Dict[int, float]  # vocab_size -> ipa/qwen ratio


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
        TextPair objects with raw_text and ipa_text
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
                    # Get raw text - prefer normalized_text, fallback to text
                    raw_text = custom.get("normalized_text") or sup.get("text")
                    
                    if ipa and raw_text and isinstance(ipa, str) and isinstance(raw_text, str):
                        ipa = ipa.strip()
                        raw_text = raw_text.strip()
                        if ipa and raw_text:
                            yield TextPair(raw_text=raw_text, ipa_text=ipa, lang=lang)
            except json.JSONDecodeError:
                continue


def sample_text_pairs(lang: str, num_samples: int = 1000, seed: int = 42) -> List[TextPair]:
    """
    Sample text pairs from a language's cuts_with_ipa directories.
    
    Args:
        lang: Language code
        num_samples: Number of samples to collect
        seed: Random seed for reproducibility
    
    Returns:
        List of TextPair objects
    """
    random.seed(seed)
    
    if lang not in CUTS_DIRS_BY_LANG:
        raise ValueError(f"Unknown language: {lang}")
    
    # Collect all text pairs from all directories
    all_pairs = []
    for cuts_dir_str in CUTS_DIRS_BY_LANG[lang]:
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


def collect_all_ipa_strings(langs: List[str]) -> Generator[str, None, None]:
    """Collect all IPA strings from specified languages for tokenizer training."""
    for lang in langs:
        if lang not in CUTS_DIRS_BY_LANG:
            continue
        print(f"[INFO] Collecting IPA strings from {lang}...")
        for cuts_dir_str in CUTS_DIRS_BY_LANG[lang]:
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


def train_ipa_bpe_tokenizer(
    output_dir: str,
    vocab_size: int,
    langs: List[str],
    min_frequency: int = 2,
) -> Tokenizer:
    """
    Train a byte-level BPE tokenizer on IPA strings.
    """
    tokenizer_dir = os.path.join(output_dir, f"ipa_bpe_v{vocab_size}")
    os.makedirs(tokenizer_dir, exist_ok=True)
    
    tokenizer_file = os.path.join(tokenizer_dir, "tokenizer.json")
    
    # Check if already trained
    if os.path.exists(tokenizer_file):
        print(f"[INFO] Loading existing tokenizer from {tokenizer_file}")
        return Tokenizer.from_file(tokenizer_file)
    
    # Write IPA strings to temp file
    temp_file = os.path.join(tokenizer_dir, "ipa_corpus.txt")
    print(f"[INFO] Writing IPA corpus for vocab_size={vocab_size}...")
    
    total_count = 0
    with open(temp_file, "w", encoding="utf-8") as f:
        for ipa in collect_all_ipa_strings(langs):
            f.write(ipa + "\n")
            total_count += 1
            if total_count % 100000 == 0:
                print(f"[INFO] Written {total_count} IPA strings...")
    
    print(f"[INFO] Total IPA strings: {total_count}")
    
    if total_count == 0:
        raise ValueError("No IPA strings found.")
    
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
    tokenizer.train(files=[temp_file], trainer=trainer)
    
    # Save
    tokenizer.save(tokenizer_file)
    tokenizer.model.save(tokenizer_dir)
    
    print(f"[INFO] Saved tokenizer to {tokenizer_dir}")
    
    # Cleanup temp file
    if os.path.exists(temp_file):
        os.remove(temp_file)
    
    return tokenizer


def compute_stats(
    text_pairs: List[TextPair],
    qwen_tokenizer: AutoTokenizer,
    ipa_tokenizers: Dict[int, Tokenizer],
    lang: str,
) -> TokenizationStats:
    """
    Compute tokenization statistics for a set of text pairs.
    """
    qwen_counts = []
    ipa_counts = {vs: [] for vs in ipa_tokenizers.keys()}
    
    for pair in text_pairs:
        # Qwen tokenizer on raw text
        qwen_tokens = qwen_tokenizer.encode(pair.raw_text)
        qwen_counts.append(len(qwen_tokens))
        
        # IPA tokenizers on IPA text
        for vocab_size, tokenizer in ipa_tokenizers.items():
            ipa_tokens = tokenizer.encode(pair.ipa_text)
            ipa_counts[vocab_size].append(len(ipa_tokens.ids))
    
    qwen_counts = np.array(qwen_counts)
    
    stats = TokenizationStats(
        lang=lang,
        num_samples=len(text_pairs),
        qwen_tokens_mean=float(np.mean(qwen_counts)),
        qwen_tokens_std=float(np.std(qwen_counts)),
        qwen_tokens_total=int(np.sum(qwen_counts)),
        ipa_tokens_mean={},
        ipa_tokens_std={},
        ipa_tokens_total={},
        compression_ratio={},
    )
    
    for vocab_size in ipa_tokenizers.keys():
        ipa_arr = np.array(ipa_counts[vocab_size])
        stats.ipa_tokens_mean[vocab_size] = float(np.mean(ipa_arr))
        stats.ipa_tokens_std[vocab_size] = float(np.std(ipa_arr))
        stats.ipa_tokens_total[vocab_size] = int(np.sum(ipa_arr))
        stats.compression_ratio[vocab_size] = stats.ipa_tokens_total[vocab_size] / stats.qwen_tokens_total
    
    return stats


def print_stats_table(all_stats: List[TokenizationStats], vocab_sizes: List[int]):
    """Print a formatted table of statistics."""
    print("\n" + "=" * 120)
    print("TOKENIZATION COMPARISON: Qwen2.5-1.5B-Instruct (raw text) vs IPA BPE (phonemized)")
    print("=" * 120)
    
    # Header
    header = f"{'Lang':<6} {'Samples':>8} {'Qwen Mean':>12} {'Qwen Std':>10}"
    for vs in vocab_sizes:
        header += f" {'IPA-' + str(vs) + ' Mean':>14} {'IPA-' + str(vs) + ' Std':>12} {'Ratio':>8}"
    print(header)
    print("-" * 120)
    
    # Data rows
    for stats in all_stats:
        row = f"{stats.lang:<6} {stats.num_samples:>8} {stats.qwen_tokens_mean:>12.2f} {stats.qwen_tokens_std:>10.2f}"
        for vs in vocab_sizes:
            row += f" {stats.ipa_tokens_mean[vs]:>14.2f} {stats.ipa_tokens_std[vs]:>12.2f} {stats.compression_ratio[vs]:>8.2f}"
        print(row)
    
    # Aggregated stats
    print("-" * 120)
    total_samples = sum(s.num_samples for s in all_stats)
    total_qwen = sum(s.qwen_tokens_total for s in all_stats)
    
    agg_row = f"{'TOTAL':<6} {total_samples:>8} {'-':>12} {'-':>10}"
    for vs in vocab_sizes:
        total_ipa = sum(s.ipa_tokens_total[vs] for s in all_stats)
        ratio = total_ipa / total_qwen if total_qwen > 0 else 0
        agg_row += f" {'-':>14} {'-':>12} {ratio:>8.2f}"
    print(agg_row)
    print("=" * 120)
    
    # Summary
    print("\nSUMMARY:")
    print(f"  - Total samples analyzed: {total_samples}")
    print(f"  - Total Qwen tokens: {total_qwen}")
    for vs in vocab_sizes:
        total_ipa = sum(s.ipa_tokens_total[vs] for s in all_stats)
        ratio = total_ipa / total_qwen if total_qwen > 0 else 0
        print(f"  - Total IPA tokens (vocab={vs}): {total_ipa} (ratio: {ratio:.2f}x)")
    print()


def save_results_json(all_stats: List[TokenizationStats], output_path: str):
    """Save results to JSON file."""
    results = []
    for stats in all_stats:
        results.append({
            "lang": stats.lang,
            "num_samples": stats.num_samples,
            "qwen": {
                "mean": stats.qwen_tokens_mean,
                "std": stats.qwen_tokens_std,
                "total": stats.qwen_tokens_total,
            },
            "ipa_bpe": {
                str(vs): {
                    "mean": stats.ipa_tokens_mean[vs],
                    "std": stats.ipa_tokens_std[vs],
                    "total": stats.ipa_tokens_total[vs],
                    "ratio_vs_qwen": stats.compression_ratio[vs],
                }
                for vs in stats.ipa_tokens_mean.keys()
            }
        })
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"[INFO] Saved results to {output_path}")


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
        help="Number of samples per language (default: 1000)",
    )
    parser.add_argument(
        "--lang",
        type=str,
        default="all",
        help="Language to analyze, or 'all' for all languages",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling (default: 42)",
    )
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Determine languages to process
    if args.lang == "all":
        langs = list(CUTS_DIRS_BY_LANG.keys())
    else:
        if args.lang not in CUTS_DIRS_BY_LANG:
            print(f"[ERROR] Unknown language: {args.lang}")
            print(f"[ERROR] Available: {list(CUTS_DIRS_BY_LANG.keys())}")
            sys.exit(1)
        langs = [args.lang]
    
    print(f"[INFO] Languages to analyze: {langs}")
    print(f"[INFO] Samples per language: {args.samples_per_lang}")
    print(f"[INFO] Vocab sizes: {VOCAB_SIZES}")
    
    # Step 1: Train IPA BPE tokenizers at different vocab sizes
    print("\n" + "=" * 60)
    print("STEP 1: Training IPA BPE tokenizers")
    print("=" * 60)
    
    ipa_tokenizers = {}
    for vocab_size in VOCAB_SIZES:
        print(f"\n[INFO] Training tokenizer with vocab_size={vocab_size}")
        ipa_tokenizers[vocab_size] = train_ipa_bpe_tokenizer(
            output_dir=args.output_dir,
            vocab_size=vocab_size,
            langs=langs,
            min_frequency=2,
        )
    
    # Step 2: Load Qwen tokenizer
    print("\n" + "=" * 60)
    print("STEP 2: Loading Qwen tokenizer")
    print("=" * 60)
    
    print("[INFO] Loading Qwen/Qwen2.5-1.5B-Instruct tokenizer...")
    qwen_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
    print(f"[INFO] Qwen tokenizer vocab size: {qwen_tokenizer.vocab_size}")
    
    # Step 3: Sample text pairs and compute statistics
    print("\n" + "=" * 60)
    print("STEP 3: Sampling and analyzing")
    print("=" * 60)
    
    all_stats = []
    for lang in langs:
        print(f"\n[INFO] Processing language: {lang}")
        
        # Sample text pairs
        text_pairs = sample_text_pairs(lang, args.samples_per_lang, args.seed)
        
        if not text_pairs:
            print(f"[WARN] No text pairs found for {lang}, skipping")
            continue
        
        print(f"[INFO] Sampled {len(text_pairs)} text pairs for {lang}")
        
        # Compute stats
        stats = compute_stats(text_pairs, qwen_tokenizer, ipa_tokenizers, lang)
        all_stats.append(stats)
        
        # Print intermediate results
        print(f"[INFO] {lang}: Qwen mean={stats.qwen_tokens_mean:.2f}, ", end="")
        for vs in VOCAB_SIZES:
            print(f"IPA-{vs} mean={stats.ipa_tokens_mean[vs]:.2f} (ratio={stats.compression_ratio[vs]:.2f}), ", end="")
        print()
    
    # Step 4: Print and save results
    print("\n" + "=" * 60)
    print("STEP 4: Results")
    print("=" * 60)
    
    print_stats_table(all_stats, VOCAB_SIZES)
    
    # Save to JSON
    results_path = os.path.join(args.output_dir, "tokenization_comparison.json")
    save_results_json(all_stats, results_path)
    
    print("[INFO] Done!")


if __name__ == "__main__":
    main()
