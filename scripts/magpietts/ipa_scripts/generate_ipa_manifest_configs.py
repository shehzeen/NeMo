#!/usr/bin/env python3
"""
Generate IPA-enhanced manifest config files for TTS training.

This script reads existing train_ais_25fpsSpectralCodec_{lang}.yaml files and creates
new train_25fpsSpectralCodec_{lang}_with_ipa.yaml files with updated paths pointing
to the IPA-enhanced cuts directories.

Transformations:
- All paths: s3://data/TTS/lhotse_datasets_2505/... -> /data/speech_data/...
- cuts directory: cuts/ -> cuts_with_ipa/
- cuts_textContext/ -> cuts_textContext_with_ipa/

Special handling:
- Hindi: Replace old cuts_ipa/cuts/ paths with new cuts_with_ipa/ paths
- Vietnamese: Skip Long_IPA dataset entries

Usage:
    python generate_ipa_manifest_configs.py
"""

import re
from pathlib import Path
from typing import Dict, List, Any
import yaml


# Configuration
MANIFESTS_DIR = Path("/Data/tts_lhotse_datasets/magpie_pretraining_data/manifests")
LANGUAGES = ["en", "de", "es", "fr", "hi", "it", "vi", "zh"]

# Path transformation patterns
S3_PREFIX = "s3://data/TTS/lhotse_datasets_2505/"
LOCAL_PREFIX = "/data/speech_data/"


def transform_cuts_path(path: str, lang: str) -> str:
    """
    Transform a cuts path from S3 format to local IPA format.
    
    Examples:
        s3://data/TTS/lhotse_datasets_2505/en/nvyt2505/.../cuts/cuts.{...}.jsonl.gz
        -> /data/speech_data/en/nvyt2505/.../cuts_with_ipa/cuts.{...}.jsonl.gz
        
        s3://data/TTS/lhotse_datasets_2505/en/rivaLindyRodney/.../cuts_textContext/cuts.{...}.jsonl.gz
        -> /data/speech_data/en/rivaLindyRodney/.../cuts_textContext_with_ipa/cuts.{...}.jsonl.gz
    """
    # Replace S3 prefix with local prefix
    if path.startswith(S3_PREFIX):
        path = LOCAL_PREFIX + path[len(S3_PREFIX):]
    
    # Handle Hindi special case: cuts_ipa/cuts/ -> cuts_with_ipa/
    if lang == "hi" and "/cuts_ipa/cuts/" in path:
        path = path.replace("/cuts_ipa/cuts/", "/cuts_with_ipa/")
    # Regular cuts directory transformation
    elif "/cuts/cuts." in path:
        path = path.replace("/cuts/cuts.", "/cuts_with_ipa/cuts.")
    # Text context cuts transformation
    elif "/cuts_textContext/cuts." in path:
        path = path.replace("/cuts_textContext/cuts.", "/cuts_textContext_with_ipa/cuts.")
    
    return path


def should_skip_entry(entry: Dict[str, Any], lang: str) -> bool:
    """
    Determine if an entry should be skipped.
    
    For Vietnamese, skip Long_IPA dataset entries.
    """
    if lang == "vi":
        shar_path = entry.get("shar_path", {})
        cuts_path = shar_path.get("cuts", "")
        if "/Long_IPA/" in cuts_path:
            return True
    return False


def transform_s3_path(path: str) -> str:
    """
    Transform any S3 path to local path by replacing the S3 prefix.
    
    Example:
        s3://data/TTS/lhotse_datasets_2505/en/nvyt2505/.../target_audio/...
        -> /data/speech_data/en/nvyt2505/.../target_audio/...
    """
    if path.startswith(S3_PREFIX):
        return LOCAL_PREFIX + path[len(S3_PREFIX):]
    return path


def transform_entry(entry: Dict[str, Any], lang: str) -> Dict[str, Any]:
    """
    Transform a single manifest entry to use IPA cuts paths and local paths.
    
    - 'cuts' path: S3 -> local, and cuts/ -> cuts_with_ipa/
    - Other paths (target_audio, context_audio, target_codes, context_codes): S3 -> local
    """
    new_entry = {}
    
    for key, value in entry.items():
        if key == "shar_path":
            new_shar_path = {}
            for shar_key, shar_value in value.items():
                if shar_key == "cuts":
                    # Transform cuts path (S3 -> local + cuts -> cuts_with_ipa)
                    new_shar_path[shar_key] = transform_cuts_path(shar_value, lang)
                else:
                    # Transform other paths (S3 -> local only)
                    new_shar_path[shar_key] = transform_s3_path(shar_value)
            new_entry[key] = new_shar_path
        else:
            new_entry[key] = value
    
    return new_entry


def load_yaml_preserving_order(filepath: Path) -> List[Dict[str, Any]]:
    """Load YAML file as a list of dictionaries."""
    with open(filepath, 'r') as f:
        content = yaml.safe_load(f)
    
    # The YAML file is a list of entries
    if isinstance(content, list):
        return content
    else:
        return [content]


def save_yaml(data: List[Dict[str, Any]], filepath: Path):
    """Save data to YAML file with proper formatting."""
    
    class FlowStyleDumper(yaml.SafeDumper):
        pass
    
    def represent_list(dumper, data):
        # Use flow style for tokenizer_names lists
        if len(data) > 0 and isinstance(data[0], str):
            return dumper.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=False)
        return dumper.represent_sequence('tag:yaml.org,2002:seq', data)
    
    FlowStyleDumper.add_representer(list, represent_list)
    
    with open(filepath, 'w') as f:
        yaml.dump(data, f, Dumper=FlowStyleDumper, default_flow_style=False, 
                  allow_unicode=True, sort_keys=False)


def process_language_manifest(lang: str) -> int:
    """
    Process a single language manifest file.
    
    Returns the number of entries written.
    """
    input_file = MANIFESTS_DIR / f"train_ais_25fpsSpectralCodec_{lang}.yaml"
    output_file = MANIFESTS_DIR / f"train_25fpsSpectralCodec_{lang}_with_ipa.yaml"
    
    if not input_file.exists():
        print(f"[WARN] Input file not found: {input_file}")
        return 0
    
    print(f"[INFO] Processing {lang}: {input_file}")
    
    # Load entries
    entries = load_yaml_preserving_order(input_file)
    
    # Transform entries
    new_entries = []
    skipped = 0
    for entry in entries:
        if should_skip_entry(entry, lang):
            skipped += 1
            continue
        new_entries.append(transform_entry(entry, lang))
    
    # Save output
    save_yaml(new_entries, output_file)
    
    print(f"[OK] Wrote {output_file} ({len(new_entries)} entries, {skipped} skipped)")
    return len(new_entries)


def generate_combined_manifest():
    """
    Generate the combined manifest file that references all language files.
    """
    input_file = MANIFESTS_DIR / "train_ais_25fpsSpectralCodec_en_de_es_fr_hi_it_vi_zh.yaml"
    output_file = MANIFESTS_DIR / "train_25fpsSpectralCodec_en_de_es_fr_hi_it_vi_zh_with_ipa.yaml"
    
    if not input_file.exists():
        print(f"[WARN] Combined input file not found: {input_file}")
        return
    
    print(f"[INFO] Processing combined manifest: {input_file}")
    
    # Load entries
    entries = load_yaml_preserving_order(input_file)
    
    # Transform entries - update input_cfg paths
    new_entries = []
    for entry in entries:
        new_entry = dict(entry)
        if "input_cfg" in new_entry:
            # Transform the input_cfg path
            # /data/magpie_pretraining_data/manifests/train_ais_25fpsSpectralCodec_en.yaml
            # -> /data/magpie_pretraining_data/manifests/train_25fpsSpectralCodec_en_with_ipa.yaml
            old_cfg = new_entry["input_cfg"]
            # Remove "ais_" and add "_with_ipa" before .yaml
            new_cfg = old_cfg.replace("train_ais_25fpsSpectralCodec_", "train_25fpsSpectralCodec_")
            new_cfg = new_cfg.replace(".yaml", "_with_ipa.yaml")
            new_entry["input_cfg"] = new_cfg
        new_entries.append(new_entry)
    
    # Save output
    save_yaml(new_entries, output_file)
    
    print(f"[OK] Wrote {output_file} ({len(new_entries)} entries)")


def main():
    print("=" * 60)
    print("Generating IPA-enhanced manifest config files")
    print("=" * 60)
    
    # Process individual language manifests
    for lang in LANGUAGES:
        process_language_manifest(lang)
        print()
    
    # Generate combined manifest
    generate_combined_manifest()
    
    print()
    print("=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
