#!/usr/bin/env python3
"""
Analyze corruption pattern within a specific directory.
Shows corruption status per shard to identify patterns.
"""

import argparse
import gzip
import json
import re
from pathlib import Path

# IPA-specific characters that should NOT appear in normal text
IPA_MARKERS = set([
    'ː', 'ˈ', 'ˌ', 'ɔ', 'ɲ', 'ɗ', 'ɛ', 'ə', 'ɜ', 'ʊ', 'ɪ', 
    'ʃ', 'ʒ', 'θ', 'ð', 'ɹ', 'ʔ', 'ɡ', 'ɑ', 'æ', 'ʌ', 'ɒ',
    'ʈ', 'ɖ', 'ʂ', 'ʐ', 'ɕ', 'ʑ', 'ɻ', 'ɭ', 'ɳ',
])

TONE_PATTERN = re.compile(r'[aeiouəɛɔʊɪæɑ][ː]?[1-7]')


def text_contains_ipa(text: str) -> bool:
    for char in text:
        if char in IPA_MARKERS:
            return True
    if TONE_PATTERN.search(text):
        return True
    return False


def analyze_directory(ipa_dir: Path):
    """Analyze corruption per shard in a directory."""
    
    shards = sorted(ipa_dir.glob("cuts.*.jsonl.gz"))
    
    if not shards:
        print(f"No shards found in {ipa_dir}")
        return
    
    print(f"Analyzing {len(shards)} shards in: {ipa_dir}")
    print("=" * 100)
    print(f"{'Shard':<40} {'Total':>8} {'Corrupted':>10} {'Pct':>8} {'Status':<15}")
    print("-" * 100)
    
    all_results = []
    cumulative_total = 0
    cumulative_corrupted = 0
    
    for shard in shards:
        shard_name = shard.name
        total = 0
        corrupted = 0
        first_corrupted_idx = None
        first_clean_idx = None
        examples = []
        
        with gzip.open(shard, "rt", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    cut = json.loads(line)
                    supervisions = cut.get("supervisions", [])
                    
                    for sup in supervisions:
                        custom = sup.get("custom", {})
                        raw_text = custom.get("normalized_text") or sup.get("text", "")
                        
                        if raw_text:
                            total += 1
                            is_corrupted = text_contains_ipa(raw_text)
                            
                            if is_corrupted:
                                corrupted += 1
                                if first_corrupted_idx is None:
                                    first_corrupted_idx = idx
                                if len(examples) < 2:
                                    examples.append({
                                        "idx": idx,
                                        "id": cut.get("id", "?"),
                                        "text": raw_text[:100]
                                    })
                            else:
                                if first_clean_idx is None:
                                    first_clean_idx = idx
                except json.JSONDecodeError:
                    continue
        
        pct = 100 * corrupted / total if total > 0 else 0
        
        if corrupted == 0:
            status = "✓ CLEAN"
        elif corrupted == total:
            status = "✗ ALL CORRUPTED"
        else:
            status = "⚠ MIXED"
        
        print(f"{shard_name:<40} {total:>8} {corrupted:>10} {pct:>7.1f}% {status:<15}")
        
        cumulative_total += total
        cumulative_corrupted += corrupted
        
        all_results.append({
            "shard": shard_name,
            "total": total,
            "corrupted": corrupted,
            "pct": pct,
            "first_corrupted_idx": first_corrupted_idx,
            "first_clean_idx": first_clean_idx,
            "examples": examples,
        })
    
    print("-" * 100)
    overall_pct = 100 * cumulative_corrupted / cumulative_total if cumulative_total > 0 else 0
    print(f"{'TOTAL':<40} {cumulative_total:>8} {cumulative_corrupted:>10} {overall_pct:>7.1f}%")
    
    # Analyze pattern
    print("\n" + "=" * 100)
    print("CORRUPTION PATTERN ANALYSIS")
    print("=" * 100)
    
    clean_shards = [r for r in all_results if r["corrupted"] == 0]
    fully_corrupted = [r for r in all_results if r["corrupted"] == r["total"] and r["total"] > 0]
    mixed_shards = [r for r in all_results if 0 < r["corrupted"] < r["total"]]
    
    print(f"\nClean shards: {len(clean_shards)}")
    print(f"Fully corrupted shards: {len(fully_corrupted)}")
    print(f"Mixed shards: {len(mixed_shards)}")
    
    # Check if there's a clear boundary
    first_corrupted_shard = None
    last_clean_shard = None
    
    for i, r in enumerate(all_results):
        if r["corrupted"] > 0 and first_corrupted_shard is None:
            first_corrupted_shard = i
        if r["corrupted"] == 0:
            last_clean_shard = i
    
    if first_corrupted_shard is not None:
        print(f"\nFirst shard with corruption: {all_results[first_corrupted_shard]['shard']} (index {first_corrupted_shard})")
    
    if last_clean_shard is not None:
        print(f"Last fully clean shard: {all_results[last_clean_shard]['shard']} (index {last_clean_shard})")
    
    # Check if pattern is "first N clean, rest corrupted"
    if clean_shards and fully_corrupted:
        clean_indices = [all_results.index(r) for r in clean_shards]
        corrupted_indices = [all_results.index(r) for r in fully_corrupted]
        
        if max(clean_indices) < min(corrupted_indices):
            print(f"\n📊 PATTERN DETECTED: First {len(clean_shards)} shards are clean, then corruption starts")
            print(f"   Clean shards: {[r['shard'] for r in clean_shards]}")
        elif min(clean_indices) > max(corrupted_indices):
            print(f"\n📊 PATTERN DETECTED: First {len(fully_corrupted)} shards are corrupted, then clean")
    
    # Show examples from mixed/corrupted shards
    print("\n" + "=" * 100)
    print("SAMPLE CORRUPTED TEXTS")
    print("=" * 100)
    
    shown = 0
    for r in all_results:
        if r["examples"] and shown < 10:
            for ex in r["examples"]:
                print(f"\nShard: {r['shard']}, Line index: {ex['idx']}")
                print(f"ID: {ex['id']}")
                print(f"TEXT: {ex['text']}...")
                shown += 1
                if shown >= 10:
                    break


def main():
    parser = argparse.ArgumentParser(description="Analyze corruption pattern in a directory")
    parser.add_argument("--dir", type=str, required=True, help="Path to the cuts_with_ipa directory")
    args = parser.parse_args()
    
    ipa_dir = Path(args.dir)
    if not ipa_dir.exists():
        print(f"Directory does not exist: {ipa_dir}")
        return
    
    analyze_directory(ipa_dir)


if __name__ == "__main__":
    main()
