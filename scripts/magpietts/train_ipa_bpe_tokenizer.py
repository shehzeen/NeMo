#!/usr/bin/env python3
"""
Train a byte-level BPE tokenizer on IPA strings from Lhotse cuts_with_ipa shards.

This script:
1. Reads IPA strings from cuts_with_ipa directories (output of add_ipa_to_lhotse_shards.py)
2. Trains a HuggingFace ByteLevelBPETokenizer on all extracted IPA strings
3. Saves vocab.json and merges.txt to the specified output directory

Usage:
    python train_ipa_bpe_tokenizer.py --output_dir /path/to/output --vocab_size 1024
    python train_ipa_bpe_tokenizer.py --output_dir /path/to/output --lang en --vocab_size 2048
    python train_ipa_bpe_tokenizer.py --output_dir /path/to/output --lang all  # all languages

The trained tokenizer can be loaded using the IPABPETokenizer class in:
    nemo/collections/common/tokenizers/text_to_speech/tts_tokenizers.py
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, Generator, List, Optional

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer

# -------------------------
# USER CONFIG - Same structure as add_ipa_to_lhotse_shards.py
# -------------------------

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
    "hi": [
        "/Data/tts_lhotse_datasets/speech_data/hi/nvyt_hi/filter_1/cuts",
        "/Data/tts_lhotse_datasets/speech_data/hi/nvyt_hi/filter_2/cuts",
        "/Data/tts_lhotse_datasets/speech_data/hi/nvyt_hi_2/filter_1/cuts",
        "/Data/tts_lhotse_datasets/speech_data/hi/nvyt_hi_2/filter_2/cuts",
    ],
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
    "zh": [
        "/Data/tts_lhotse_datasets/speech_data/zh/riva_ZH_SiweiHouZhen/cuts",
        "/Data/tts_lhotse_datasets/speech_data/zh/riva_ZH_SiweiHouZhen/cuts_textContext",
        "/Data/tts_lhotse_datasets/speech_data/zh/nvyt_zh/filter_1/cuts",
        "/Data/tts_lhotse_datasets/speech_data/zh/nvyt_zh/filter_2/cuts",
    ],
    "en": [
        "/Data/tts_lhotse_datasets/speech_data/en/nvyt2505/lhotse_shar_shuffle_shardSize256/cuts",
        "/Data/tts_lhotse_datasets/speech_data/en/hifitts/lhotse_shar_shuffle_shardSize256/cuts",
        "/Data/tts_lhotse_datasets/speech_data/en/hifitts2/lhotse_shar_shuffle_shardSize256/cuts",
        "/Data/tts_lhotse_datasets/speech_data/en/jhsdGtc20Amp20Keynote/lhotse_shar_shuffle_shardSize256/cuts",
        "/Data/tts_lhotse_datasets/speech_data/en/libritts/lhotse_shar_shuffle_shardSize256/cuts",
        "/Data/tts_lhotse_datasets/speech_data/en/rivaLindyRodney/lhotse_shar_shuffle_shardSize256/cuts",
        "/Data/tts_lhotse_datasets/speech_data/en/rivaLindyRodney/lhotse_shar_shuffle_shardSize256/cuts_textContext",
        "/Data/tts_lhotse_datasets/speech_data/en/rivaEmmaMeganSeanTom/lhotse_shar_shuffle_shardSize256/cuts",
        "/Data/tts_lhotse_datasets/speech_data/en/rivaEmmaMeganSeanTom/lhotse_shar_shuffle_shardSize256/cuts_textContext",
        "/Data/tts_lhotse_datasets/speech_data/en/jhsdGtc20Amp20Keynote/lhotse_shar_shuffle_shardSize256/cuts_textContext",
    ],
}

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


def get_available_languages() -> List[str]:
    """Return list of all available language codes."""
    return list(CUTS_DIRS_BY_LANG.keys())


def collect_ipa_strings(lang: Optional[str] = None) -> Generator[str, None, None]:
    """
    Collect all IPA strings from the specified language(s).
    
    Args:
        lang: Language code or None for all languages.
    
    Yields:
        IPA strings
    """
    if lang is None or lang == "all":
        langs_to_process = list(CUTS_DIRS_BY_LANG.keys())
    else:
        if lang not in CUTS_DIRS_BY_LANG:
            raise ValueError(f"Unknown language: {lang}. Available: {get_available_languages()}")
        langs_to_process = [lang]
    
    for lang_code in langs_to_process:
        print(f"[INFO] Processing language: {lang_code}")
        for cuts_dir_str in CUTS_DIRS_BY_LANG[lang_code]:
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


def train_bpe_tokenizer(
    ipa_strings: Generator[str, None, None],
    vocab_size: int = 1024,
    min_frequency: int = 2,
    output_dir: str = "./ipa_bpe_tokenizer",
) -> Tokenizer:
    """
    Train a byte-level BPE tokenizer on IPA strings.
    
    Args:
        ipa_strings: Generator yielding IPA strings
        vocab_size: Target vocabulary size
        min_frequency: Minimum frequency for a token to be included
        output_dir: Directory to save the tokenizer files
    
    Returns:
        Trained Tokenizer object
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Write IPA strings to a temporary file for training
    # (HuggingFace tokenizers can train from files more efficiently)
    temp_file = os.path.join(output_dir, "ipa_corpus.txt")
    print(f"[INFO] Writing IPA corpus to temporary file: {temp_file}")
    
    total_count = 0
    with open(temp_file, "w", encoding="utf-8") as f:
        for ipa in ipa_strings:
            f.write(ipa + "\n")
            total_count += 1
            if total_count % 100000 == 0:
                print(f"[INFO] Written {total_count} IPA strings...")
    
    print(f"[INFO] Total IPA strings collected: {total_count}")
    
    if total_count == 0:
        raise ValueError("No IPA strings found. Make sure the cuts_with_ipa directories exist.")
    
    # Initialize a byte-level BPE tokenizer
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    
    # Use byte-level pre-tokenization (like GPT-2)
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    
    # Define special tokens
    # Note: Byte-level BPE can represent any UTF-8 text, so no OOV token is needed
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
    tokenizer.train(files=[temp_file], trainer=trainer)
    
    # Save the tokenizer
    vocab_path = os.path.join(output_dir, "vocab.json")
    merges_path = os.path.join(output_dir, "merges.txt")
    
    # Save using the tokenizer's model save method
    tokenizer.model.save(output_dir)
    
    # Also save the full tokenizer for easy loading
    tokenizer_path = os.path.join(output_dir, "tokenizer.json")
    tokenizer.save(tokenizer_path)
    
    print(f"[INFO] Tokenizer saved to: {output_dir}")
    print(f"[INFO]   - vocab.json: {vocab_path}")
    print(f"[INFO]   - merges.txt: {merges_path}")
    print(f"[INFO]   - tokenizer.json: {tokenizer_path}")
    print(f"[INFO] Vocabulary size: {tokenizer.get_vocab_size()}")
    
    # Clean up temp file
    if os.path.exists(temp_file):
        os.remove(temp_file)
        print(f"[INFO] Removed temporary corpus file")
    
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
        "--lang",
        type=str,
        default="all",
        help=f"Language code to process, or 'all' for all languages. Available: {get_available_languages()}",
    )
    args = parser.parse_args()
    
    print(f"[INFO] Training IPA BPE tokenizer")
    print(f"[INFO]   Output directory: {args.output_dir}")
    print(f"[INFO]   Vocabulary size: {args.vocab_size}")
    print(f"[INFO]   Min frequency: {args.min_frequency}")
    print(f"[INFO]   Language(s): {args.lang}")
    
    # Collect IPA strings
    ipa_generator = collect_ipa_strings(args.lang)
    
    # Train tokenizer
    tokenizer = train_bpe_tokenizer(
        ipa_strings=ipa_generator,
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        output_dir=args.output_dir,
    )
    
    # Test the tokenizer
    print("\n[INFO] Testing tokenizer with sample IPA strings:")
    test_strings = [
        "həˈloʊ wɜːld",  # hello world
        "ˈaɪ pʰiː eɪ",   # IPA
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
