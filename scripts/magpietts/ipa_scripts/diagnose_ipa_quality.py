#!/usr/bin/env python3
"""
Diagnose IPA quality by inspecting samples from each language.
Helps identify potential issues with phonemization.
"""

import argparse
import gzip
import json
import random
from pathlib import Path
from typing import Dict, List

# Default config file path
DEFAULT_CONFIG_PATH = Path(__file__).parent / "cuts_dirs_config.json"
OUTPUT_SUFFIX = "_with_ipa"
SHARD_GLOB = "cuts.*.jsonl.gz"


def load_config(config_path: Path) -> Dict[str, List[str]]:
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_ipa_dir(cuts_dir: Path) -> Path:
    name = cuts_dir.name
    if name == "cuts":
        out_name = f"cuts{OUTPUT_SUFFIX}"
    else:
        out_name = f"{name}{OUTPUT_SUFFIX}"
    return cuts_dir.parent / out_name


def sample_cuts(lang: str, cuts_dirs: Dict[str, List[str]], num_samples: int = 20, seed: int = 42) -> List[dict]:
    """Sample raw cut data for inspection."""
    random.seed(seed)
    all_cuts = []
    
    for cuts_dir_str in cuts_dirs.get(lang, []):
        cuts_dir = Path(cuts_dir_str)
        ipa_dir = get_ipa_dir(cuts_dir)
        
        if not ipa_dir.exists():
            print(f"[WARN] IPA directory does not exist: {ipa_dir}")
            continue
        
        shards = sorted(ipa_dir.glob(SHARD_GLOB))
        for shard in shards:
            with gzip.open(shard, "rt", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        cut = json.loads(line)
                        duration = cut.get("duration", 0.0)
                        supervisions = cut.get("supervisions", [])
                        for sup in supervisions:
                            custom = sup.get("custom", {})
                            ipa = custom.get("ipa", "")
                            raw_text = custom.get("normalized_text") or sup.get("text", "")
                            
                            if ipa and raw_text and duration > 0:
                                all_cuts.append({
                                    "id": cut.get("id", "unknown"),
                                    "duration": duration,
                                    "raw_text": raw_text,
                                    "ipa": ipa,
                                    "text_len": len(raw_text),
                                    "ipa_len": len(ipa),
                                    "ipa_char_per_sec": len(ipa) / duration,
                                    "text_char_per_sec": len(raw_text) / duration,
                                    "ipa_to_text_ratio": len(ipa) / len(raw_text) if len(raw_text) > 0 else 0,
                                })
                    except json.JSONDecodeError:
                        continue
            
            if len(all_cuts) >= num_samples * 10:
                break
        if len(all_cuts) >= num_samples * 10:
            break
    
    if len(all_cuts) <= num_samples:
        return all_cuts
    return random.sample(all_cuts, num_samples)


def analyze_language(lang: str, cuts_dirs: Dict[str, List[str]], num_samples: int = 50):
    """Analyze IPA quality for a language."""
    print(f"\n{'='*80}")
    print(f"LANGUAGE: {lang}")
    print(f"{'='*80}")
    
    samples = sample_cuts(lang, cuts_dirs, num_samples)
    
    if not samples:
        print(f"[ERROR] No samples found for {lang}")
        return
    
    # Compute statistics
    durations = [s["duration"] for s in samples]
    ipa_lens = [s["ipa_len"] for s in samples]
    text_lens = [s["text_len"] for s in samples]
    ipa_per_sec = [s["ipa_char_per_sec"] for s in samples]
    text_per_sec = [s["text_char_per_sec"] for s in samples]
    ipa_to_text = [s["ipa_to_text_ratio"] for s in samples]
    
    print(f"\nSTATISTICS (n={len(samples)}):")
    print(f"  Duration:       min={min(durations):.2f}s, max={max(durations):.2f}s, avg={sum(durations)/len(durations):.2f}s")
    print(f"  Text length:    min={min(text_lens)}, max={max(text_lens)}, avg={sum(text_lens)/len(text_lens):.1f} chars")
    print(f"  IPA length:     min={min(ipa_lens)}, max={max(ipa_lens)}, avg={sum(ipa_lens)/len(ipa_lens):.1f} chars")
    print(f"  Text chars/sec: min={min(text_per_sec):.2f}, max={max(text_per_sec):.2f}, avg={sum(text_per_sec)/len(text_per_sec):.2f}")
    print(f"  IPA chars/sec:  min={min(ipa_per_sec):.2f}, max={max(ipa_per_sec):.2f}, avg={sum(ipa_per_sec)/len(ipa_per_sec):.2f}")
    print(f"  IPA/Text ratio: min={min(ipa_to_text):.2f}, max={max(ipa_to_text):.2f}, avg={sum(ipa_to_text)/len(ipa_to_text):.2f}")
    
    # Show sample pairs
    print(f"\nSAMPLE TEXT-IPA PAIRS:")
    print("-" * 80)
    
    # Sort by IPA chars/sec to show potential outliers
    samples_sorted = sorted(samples, key=lambda x: x["ipa_char_per_sec"], reverse=True)
    
    for i, s in enumerate(samples_sorted[:10]):
        print(f"\n[{i+1}] Duration: {s['duration']:.2f}s | IPA chars/sec: {s['ipa_char_per_sec']:.2f} | IPA/Text ratio: {s['ipa_to_text_ratio']:.2f}")
        print(f"    TEXT ({s['text_len']} chars): {s['raw_text'][:100]}{'...' if len(s['raw_text']) > 100 else ''}")
        print(f"    IPA  ({s['ipa_len']} chars): {s['ipa'][:100]}{'...' if len(s['ipa']) > 100 else ''}")
    
    # Check for potential issues
    print(f"\nPOTENTIAL ISSUES:")
    
    # Check for very high IPA/text ratio (might indicate repeated phonemes or errors)
    high_ratio = [s for s in samples if s["ipa_to_text_ratio"] > 2.0]
    if high_ratio:
        print(f"  - {len(high_ratio)} samples have IPA/Text ratio > 2.0 (unusual)")
    
    # Check for very low IPA/text ratio (might indicate truncated IPA)
    low_ratio = [s for s in samples if s["ipa_to_text_ratio"] < 0.5]
    if low_ratio:
        print(f"  - {len(low_ratio)} samples have IPA/Text ratio < 0.5 (might be truncated)")
    
    # Check for very short durations
    short_dur = [s for s in samples if s["duration"] < 1.0]
    if short_dur:
        print(f"  - {len(short_dur)} samples have duration < 1 second")
    
    # Check for suspiciously high IPA chars/sec
    high_ipa_rate = [s for s in samples if s["ipa_char_per_sec"] > 30]
    if high_ipa_rate:
        print(f"  - {len(high_ipa_rate)} samples have > 30 IPA chars/sec (suspicious)")
    
    if not (high_ratio or low_ratio or short_dur or high_ipa_rate):
        print("  - No obvious issues detected")


def main():
    parser = argparse.ArgumentParser(description="Diagnose IPA quality across languages")
    parser.add_argument("--langs", type=str, default="vi", help="Comma-separated languages to analyze (default: vi)")
    parser.add_argument("--num_samples", type=int, default=50, help="Number of samples per language")
    parser.add_argument("--config", type=str, default=None, help="Path to config file")
    args = parser.parse_args()
    
    config_path = Path(args.config) if args.config else DEFAULT_CONFIG_PATH
    cuts_dirs = load_config(config_path)
    
    langs = [l.strip() for l in args.langs.split(",")]
    
    for lang in langs:
        if lang not in cuts_dirs:
            print(f"[ERROR] Language '{lang}' not in config")
            continue
        analyze_language(lang, cuts_dirs, args.num_samples)
    
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS:")
    print("=" * 80)
    print("""
1. Compare IPA/Text ratios across languages - they should be roughly similar
2. Manually inspect a few IPA transcriptions for correctness
3. Check if the phonemizer handles the language correctly (especially tones)
4. Verify audio durations match actual audio file lengths
5. Look for patterns in outliers (same speaker, same dataset, etc.)
""")


if __name__ == "__main__":
    main()
