#!/usr/bin/env python3
"""
Find IPA outliers - samples with unusually high phonemes per second.
"""

import argparse
import gzip
import json
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


def find_outliers(lang: str, cuts_dirs: Dict[str, List[str]], min_ipa_per_sec: float = 30.0):
    """Find all samples with IPA chars/sec above threshold."""
    outliers = []
    total_processed = 0
    
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
                        cut_id = cut.get("id", "unknown")
                        supervisions = cut.get("supervisions", [])
                        
                        for sup in supervisions:
                            custom = sup.get("custom", {})
                            ipa = custom.get("ipa", "")
                            raw_text = custom.get("normalized_text") or sup.get("text", "")
                            
                            if ipa and raw_text and duration > 0:
                                total_processed += 1
                                ipa_per_sec = len(ipa) / duration
                                
                                if ipa_per_sec >= min_ipa_per_sec:
                                    outliers.append({
                                        "id": cut_id,
                                        "duration": duration,
                                        "raw_text": raw_text,
                                        "ipa": ipa,
                                        "text_len": len(raw_text),
                                        "ipa_len": len(ipa),
                                        "ipa_per_sec": ipa_per_sec,
                                    })
                    except json.JSONDecodeError:
                        continue
    
    return outliers, total_processed


def main():
    parser = argparse.ArgumentParser(description="Find IPA outliers with high phonemes/second")
    parser.add_argument("--lang", type=str, default="vi", help="Language to analyze (default: vi)")
    parser.add_argument("--min_ipa_per_sec", type=float, default=30.0, help="Minimum IPA chars/sec threshold (default: 30)")
    parser.add_argument("--config", type=str, default=None, help="Path to config file")
    parser.add_argument("--max_display", type=int, default=100, help="Max outliers to display (default: 100)")
    args = parser.parse_args()
    
    config_path = Path(args.config) if args.config else DEFAULT_CONFIG_PATH
    cuts_dirs = load_config(config_path)
    
    if args.lang not in cuts_dirs:
        print(f"[ERROR] Language '{args.lang}' not in config")
        return
    
    print(f"Searching for {args.lang} samples with IPA chars/sec >= {args.min_ipa_per_sec}...")
    print("This may take a while as it scans all data...")
    print()
    
    outliers, total_processed = find_outliers(args.lang, cuts_dirs, args.min_ipa_per_sec)
    
    # Sort by IPA per sec descending
    outliers.sort(key=lambda x: x["ipa_per_sec"], reverse=True)
    
    print("=" * 100)
    print(f"OUTLIERS: {args.lang} samples with IPA chars/sec >= {args.min_ipa_per_sec}")
    print("=" * 100)
    print(f"Total samples processed: {total_processed}")
    print(f"Outliers found: {len(outliers)} ({100*len(outliers)/total_processed:.2f}%)")
    print()
    
    if not outliers:
        print("No outliers found!")
        return
    
    # Display outliers
    display_count = min(len(outliers), args.max_display)
    print(f"Showing top {display_count} outliers (sorted by IPA chars/sec):")
    print("-" * 100)
    
    for i, o in enumerate(outliers[:display_count]):
        print(f"\n[{i+1}] ID: {o['id']}")
        print(f"    Duration: {o['duration']:.2f}s | IPA chars/sec: {o['ipa_per_sec']:.2f}")
        print(f"    Text length: {o['text_len']} chars | IPA length: {o['ipa_len']} chars")
        print(f"    TEXT: {o['raw_text']}")
        print(f"    IPA:  {o['ipa']}")
    
    # Summary statistics for outliers
    if outliers:
        print("\n" + "=" * 100)
        print("OUTLIER STATISTICS:")
        print("=" * 100)
        durations = [o["duration"] for o in outliers]
        ipa_lens = [o["ipa_len"] for o in outliers]
        text_lens = [o["text_len"] for o in outliers]
        ipa_per_sec = [o["ipa_per_sec"] for o in outliers]
        
        print(f"  Duration:     min={min(durations):.2f}s, max={max(durations):.2f}s, avg={sum(durations)/len(durations):.2f}s")
        print(f"  Text length:  min={min(text_lens)}, max={max(text_lens)}, avg={sum(text_lens)/len(text_lens):.1f}")
        print(f"  IPA length:   min={min(ipa_lens)}, max={max(ipa_lens)}, avg={sum(ipa_lens)/len(ipa_lens):.1f}")
        print(f"  IPA/sec:      min={min(ipa_per_sec):.2f}, max={max(ipa_per_sec):.2f}, avg={sum(ipa_per_sec)/len(ipa_per_sec):.2f}")


if __name__ == "__main__":
    main()
