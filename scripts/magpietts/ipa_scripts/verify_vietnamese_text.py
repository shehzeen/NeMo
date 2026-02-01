#!/usr/bin/env python3
"""
Verify that Vietnamese cuts contain regular text (not IPA) in the text field.
"""

import gzip
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

# Vietnamese IPA cuts directories (with _with_ipa suffix)
VI_CUTS_DIRS = [
    "/Data/tts_lhotse_datasets/speech_data/vi/Infore1_2_lsvsc/cuts_with_ipa",
    "/Data/tts_lhotse_datasets/speech_data/vi/Long_ContextAudio/cuts_with_ipa",
    "/Data/tts_lhotse_datasets/speech_data/vi/Long_ContextAudio/cuts_textContext_with_ipa",
    "/Data/tts_lhotse_datasets/speech_data/vi/NorthFemale/cuts_with_ipa",
    "/Data/tts_lhotse_datasets/speech_data/vi/nvyt_vi/nvyt_yt2025/cuts_with_ipa",
]

SHARD_GLOB = "cuts.*.jsonl.gz"

# Common IPA symbols that wouldn't appear in regular Vietnamese text
IPA_MARKERS = set([
    'ɑ', 'ɐ', 'ɒ', 'æ', 'ɓ', 'ʙ', 'β', 'ɔ', 'ɕ', 'ç', 'ɗ', 'ɖ', 'ð',
    'ʤ', 'ə', 'ɘ', 'ɚ', 'ɛ', 'ɜ', 'ɝ', 'ɞ', 'ɤ', 'ɠ', 'ɢ', 'ʛ', 'ɦ',
    'ɧ', 'ħ', 'ɥ', 'ʜ', 'ɨ', 'ɪ', 'ʝ', 'ɟ', 'ʄ', 'ɡ', 'ɬ', 'ɮ', 'ʟ',
    'ɭ', 'ɱ', 'ɯ', 'ɰ', 'ŋ', 'ɳ', 'ɲ', 'ɴ', 'ø', 'ɵ', 'ɸ', 'θ', 'œ',
    'ɶ', 'ʘ', 'ɹ', 'ɺ', 'ɾ', 'ɻ', 'ʀ', 'ʁ', 'ɽ', 'ʂ', 'ʃ', 'ʈ', 'ʧ',
    'ʉ', 'ʊ', 'ʋ', 'ⱱ', 'ʌ', 'ɣ', 'ɤ', 'ʍ', 'χ', 'ʎ', 'ʏ', 'ʑ', 'ʐ',
    'ʒ', 'ʔ', 'ʡ', 'ʕ', 'ʢ', 'ǀ', 'ǁ', 'ǂ', 'ǃ', 'ˈ', 'ˌ', 'ː', 'ˑ',
    'ʼ', 'ʴ', 'ʰ', 'ʱ', 'ʲ', 'ʷ', 'ˠ', 'ˤ', '˞', 'ⁿ', 'ˡ', '˥', '˦',
    '˧', '˨', '˩', '̈', '̃', '̊', '̚'
])


def contains_ipa_markers(text: str) -> Tuple[bool, List[str]]:
    """Check if text contains IPA markers."""
    found = []
    for char in text:
        if char in IPA_MARKERS:
            found.append(char)
    return len(found) > 0, found


def analyze_cuts_dir(cuts_dir: Path) -> Dict:
    """Analyze cuts in a directory."""
    result = {
        "dir": str(cuts_dir),
        "exists": cuts_dir.exists(),
        "shards_found": 0,
        "samples_checked": 0,
        "regular_text_count": 0,
        "ipa_text_count": 0,
        "empty_text_count": 0,
        "sample_texts": [],
        "ipa_flagged": [],
    }
    
    if not cuts_dir.exists():
        return result
    
    shards = sorted(cuts_dir.glob(SHARD_GLOB))
    result["shards_found"] = len(shards)
    
    if not shards:
        return result
    
    # Process ALL shards
    for shard_idx, shard in enumerate(shards):
        try:
            with gzip.open(shard, "rt", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        cut = json.loads(line)
                        supervisions = cut.get("supervisions", [])
                        
                        for sup in supervisions:
                            text = sup.get("text", "")
                            custom = sup.get("custom", {})
                            ipa = custom.get("ipa", "")
                            normalized_text = custom.get("normalized_text", "")
                            
                            result["samples_checked"] += 1
                            
                            if not text:
                                result["empty_text_count"] += 1
                                continue
                            
                            has_ipa, ipa_chars = contains_ipa_markers(text)
                            
                            if has_ipa:
                                result["ipa_text_count"] += 1
                                if len(result["ipa_flagged"]) < 5:
                                    result["ipa_flagged"].append({
                                        "id": cut.get("id", "unknown"),
                                        "text": text,
                                        "ipa_chars_found": ipa_chars[:10],
                                        "ipa_field": ipa[:100] if ipa else "[no ipa field]",
                                    })
                            else:
                                result["regular_text_count"] += 1
                                if len(result["sample_texts"]) < 5:
                                    result["sample_texts"].append({
                                        "id": cut.get("id", "unknown"),
                                        "text": text,
                                        "normalized_text": normalized_text[:100] if normalized_text else "[none]",
                                        "ipa_field": ipa[:100] if ipa else "[no ipa field]",
                                    })
                            
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            result["error"] = f"Error in shard {shard_idx}: {str(e)}"
        
        # Print progress every 10 shards
        if (shard_idx + 1) % 10 == 0:
            print(f"    Processed {shard_idx + 1}/{len(shards)} shards, {result['samples_checked']} samples so far...")
    
    return result


def main():
    print("=" * 100)
    print("Vietnamese IPA Cuts - Text Field Verification")
    print("Checking if 'text' field contains regular text (not IPA) in IPA cuts directories")
    print("=" * 100)
    
    all_results = []
    
    for cuts_dir_str in VI_CUTS_DIRS:
        cuts_dir = Path(cuts_dir_str)
        print(f"\nChecking: {cuts_dir}")
        result = analyze_cuts_dir(cuts_dir)
        all_results.append(result)
        
        if not result["exists"]:
            print(f"  [WARNING] Directory does not exist!")
            continue
        
        if result["shards_found"] == 0:
            print(f"  [WARNING] No shards found!")
            continue
        
        print(f"  Shards found: {result['shards_found']}")
        print(f"  Samples checked: {result['samples_checked']}")
        print(f"  Regular text: {result['regular_text_count']}")
        print(f"  IPA text (flagged): {result['ipa_text_count']}")
        print(f"  Empty text: {result['empty_text_count']}")
        
        if result["sample_texts"]:
            print("\n  Sample regular texts:")
            for i, sample in enumerate(result["sample_texts"][:3]):
                print(f"    [{i+1}] ID: {sample['id']}")
                print(f"        text: {sample['text'][:80]}...")
                print(f"        ipa:  {sample['ipa_field'][:80]}...")
        
        if result["ipa_flagged"]:
            print("\n  [ALERT] IPA-like characters found in 'text' field:")
            for i, sample in enumerate(result["ipa_flagged"][:3]):
                print(f"    [{i+1}] ID: {sample['id']}")
                print(f"        text: {sample['text'][:80]}")
                print(f"        IPA chars found: {sample['ipa_chars_found']}")
    
    # Summary
    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    
    total_regular = sum(r["regular_text_count"] for r in all_results)
    total_ipa = sum(r["ipa_text_count"] for r in all_results)
    total_empty = sum(r["empty_text_count"] for r in all_results)
    total_checked = sum(r["samples_checked"] for r in all_results)
    
    print(f"Total samples checked: {total_checked}")
    print(f"Regular text fields: {total_regular} ({100*total_regular/total_checked:.1f}%)" if total_checked > 0 else "No samples")
    print(f"IPA-like text fields: {total_ipa} ({100*total_ipa/total_checked:.1f}%)" if total_checked > 0 else "No samples")
    print(f"Empty text fields: {total_empty} ({100*total_empty/total_checked:.1f}%)" if total_checked > 0 else "No samples")
    
    if total_ipa == 0:
        print("\n✅ VERIFIED: All Vietnamese 'text' fields contain regular text (no IPA detected)")
    else:
        print(f"\n⚠️ WARNING: {total_ipa} samples have IPA-like characters in the 'text' field!")


if __name__ == "__main__":
    main()
