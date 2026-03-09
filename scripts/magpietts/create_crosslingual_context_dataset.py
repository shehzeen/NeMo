# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Creates a cross-lingual context dataset for TTS training.

For each target utterance in language A, finds the closest speaker voice from a
different language B (using TitaNet speaker embeddings) and pairs the target with
context audio from that cross-lingual speaker.

The script operates in three stages:
  Stage 1: Build a per-speaker TitaNet embedding index across all languages.
  Stage 2: Compute cross-lingual speaker matches and sample a language-balanced subset.
  Stage 3: Extract audio to disk and write a NeMo-format JSONL manifest.

After running this script, use create_lhotse_shar_from_nemo_manifest.py to convert
the output manifest into lhotse shar format, then optionally run
extend_lhotse_shards_with_audio_codes.py to add codec codes.

Example usage:
    python scripts/magpietts/create_crosslingual_context_dataset.py \
        --master-yaml /data/magpie_pretraining_data/manifests/ipa_manifests/train_25fpsSpectralCodecBWE_en_de_es_fr_hi_it_vi_zh_with_ipa.yaml \
        --output-dir /data/crosslingual_context_dataset \
        --target-hours 50.0 \
        --samples-per-speaker 5 \
        --seed 42 \
        --log-level INFO
"""

import argparse
import glob as glob_module
import gzip
import json
import logging
import os
import pickle
import random
import re
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import numpy as np
import soundfile as sf
import torch
import yaml
from lhotse import CutSet
from tqdm import tqdm

TITANET_MODEL_NAME = "nvidia/speakerverification_en_titanet_large"
TITANET_SAMPLE_RATE = 16000


# ---------------------------------------------------------------------------
# YAML / shar helpers
# ---------------------------------------------------------------------------


def parse_master_yaml(yaml_path: str) -> Dict[str, List[Dict]]:
    """
    Parse the master multilingual YAML and each per-language YAML it references.
    Returns {language: [list of shar_entry dicts with context_audio]}.
    """
    yaml_base_dir = os.path.dirname(yaml_path)
    with open(yaml_path, 'r') as f:
        master_entries = yaml.safe_load(f)

    lang_to_shar_entries: Dict[str, List[Dict]] = defaultdict(list)
    for entry in master_entries:
        lang = entry.get("tags", {}).get("lang")
        child_yaml_path = entry.get("input_cfg")
        if not lang or not child_yaml_path:
            continue
        if not os.path.isabs(child_yaml_path):
            child_yaml_path = os.path.join(yaml_base_dir, child_yaml_path)
        if not os.path.isfile(child_yaml_path):
            logging.warning(f"Per-language YAML not found: {child_yaml_path}")
            continue
        with open(child_yaml_path, 'r') as f:
            child_entries = yaml.safe_load(f)
        for ce in child_entries:
            shar_path = ce.get("shar_path", {})
            if "context_audio" not in shar_path:
                logging.debug(
                    f"Skipping text-context-only entry (no context_audio): {shar_path.get('cuts', 'unknown')}"
                )
                continue
            lang_to_shar_entries[lang].append(ce)

    return dict(lang_to_shar_entries)


def expand_shar_range(pattern: str) -> List[str]:
    """
    Expand a shar path pattern like '.../cuts.{000000..001231}.jsonl.gz'
    into a list of concrete file paths.
    """
    match = re.search(r'\{(\d+)\.\.(\d+)\}', pattern)
    if not match:
        return [pattern]
    start_idx = int(match.group(1))
    end_idx = int(match.group(2))
    width = len(match.group(1))
    prefix = pattern[: match.start()]
    suffix = pattern[match.end() :]
    return [f"{prefix}{i:0{width}d}{suffix}" for i in range(start_idx, end_idx + 1)]


def parse_speaker_field(speaker_str: str) -> Tuple[str, str, str]:
    """Extract (language, dataset, speaker_id) from '| Language:XX Dataset:YYY Speaker:ZZZ |'."""
    lang_m = re.search(r"Language:(\w+)", speaker_str)
    dataset_m = re.search(r"Dataset:([\w\d\W]+?) Speaker:", speaker_str)
    spk_m = re.search(r"Speaker:([\w\d\W]+?) \|", speaker_str)
    lang = lang_m.group(1) if lang_m else "unknown"
    dataset = dataset_m.group(1).strip() if dataset_m else "unknown"
    speaker_id = spk_m.group(1).strip() if spk_m else "unknown"
    return lang, dataset, speaker_id


# ---------------------------------------------------------------------------
# Stage 1: Build speaker embedding index
# ---------------------------------------------------------------------------


def discover_speakers_from_cuts(
    lang_to_shar_entries: Dict[str, List[Dict]],
    max_cuts_per_speaker: int,
    max_shards_per_dataset: int = 0,
) -> Dict[str, Dict]:
    """
    Pass 1 (metadata only): Read cut JSONL files to discover unique speakers
    and collect up to max_cuts_per_speaker cut metadata entries per speaker.

    Args:
        max_shards_per_dataset: If > 0, only scan this many .jsonl.gz shard
            files per shar group (dataset) instead of all shards. This
            dramatically speeds up discovery for large datasets while still
            finding most speakers.

    Returns: {speaker_str: {"language": str, "cut_metas": [list of (shar_entry, shard_idx, cut_json_dict)]}}
    """
    speaker_info: Dict[str, Dict] = {}

    for lang, shar_entries in lang_to_shar_entries.items():
        logging.info(f"[Stage 1] Discovering speakers for language: {lang} ({len(shar_entries)} shar groups)")
        for se in shar_entries:
            cuts_pattern = se["shar_path"]["cuts"]
            cuts_files = expand_shar_range(cuts_pattern)
            if max_shards_per_dataset > 0 and len(cuts_files) > max_shards_per_dataset:
                logging.info(
                    f"  Limiting scan to {max_shards_per_dataset}/{len(cuts_files)} "
                    f"shards for dataset: {cuts_pattern}"
                )
                cuts_files = cuts_files[:max_shards_per_dataset]
            for cuts_file in cuts_files:
                if not os.path.isfile(cuts_file):
                    continue
                shard_idx_match = re.search(r"cuts\.(\d+)\.jsonl\.gz$", cuts_file)
                shard_idx = int(shard_idx_match.group(1)) if shard_idx_match else 0
                try:
                    with gzip.open(cuts_file, 'rt', encoding='utf-8') as f:
                        for line in f:
                            cut_json = json.loads(line)
                            supervisions = cut_json.get("supervisions", [])
                            if not supervisions:
                                continue
                            speaker_str = supervisions[0].get("speaker", "")
                            if not speaker_str:
                                continue
                            if speaker_str not in speaker_info:
                                speaker_info[speaker_str] = {
                                    "language": lang,
                                    "cut_metas": [],
                                }
                            if len(speaker_info[speaker_str]["cut_metas"]) < max_cuts_per_speaker:
                                speaker_info[speaker_str]["cut_metas"].append((se, shard_idx, cut_json))
                except Exception as e:
                    logging.warning(f"Error reading {cuts_file}: {e}")

    logging.info(
        f"[Stage 1] Discovered {len(speaker_info)} unique speakers across {len(lang_to_shar_entries)} languages"
    )
    for lang in sorted(lang_to_shar_entries.keys()):
        n = sum(1 for v in speaker_info.values() if v["language"] == lang)
        logging.info(f"  {lang}: {n} speakers")
    return speaker_info


def compute_speaker_embeddings(
    speaker_info: Dict[str, Dict],
    sv_model: torch.nn.Module,
    device: torch.device,
    batch_size: int = 16,
) -> Dict[str, Dict]:
    """
    Pass 2: For each speaker, load audio from shar tars for the sampled cuts,
    compute TitaNet embeddings, and average them into a single representative vector.

    Returns: {speaker_str: {"language": str, "embedding": np.ndarray}}
    """
    speaker_embeddings: Dict[str, Dict] = {}

    speakers_needing_audio = {}
    for spk, info in speaker_info.items():
        cut_metas = info["cut_metas"]
        if not cut_metas:
            continue
        grouped_by_shar_and_shard: Dict[str, Dict[int, List]] = defaultdict(lambda: defaultdict(list))
        for se, shard_idx, cut_json in cut_metas:
            shar_key = json.dumps(se["shar_path"], sort_keys=True)
            grouped_by_shar_and_shard[shar_key][shard_idx].append((se, cut_json))
        speakers_needing_audio[spk] = {
            "language": info["language"],
            "grouped": grouped_by_shar_and_shard,
        }

    # Collect audio in batches: load from shar, accumulate waveforms per speaker
    speaker_audio_tensors: Dict[str, List[torch.Tensor]] = defaultdict(list)

    logging.info(f"[Stage 1] Loading audio for {len(speakers_needing_audio)} speakers to compute embeddings...")

    # Group all (shar_entry, shard_idx) that we need to load
    shar_shard_to_speakers: Dict[Tuple[str, int], List[Tuple[str, str]]] = defaultdict(list)
    for spk, data in speakers_needing_audio.items():
        for shar_key, shard_map in data["grouped"].items():
            for shard_idx, items in shard_map.items():
                for se, cut_json in items:
                    cut_id = cut_json.get("id", "")
                    shar_shard_to_speakers[(shar_key, shard_idx)].append((spk, cut_id))

    # Process shard by shard to minimize tar file openings
    total_shards = len(shar_shard_to_speakers)
    for (shar_key, shard_idx), spk_cut_pairs in tqdm(
        shar_shard_to_speakers.items(), desc="[Stage 1] Loading audio shards", total=total_shards
    ):
        se_shar_path = json.loads(shar_key)
        cuts_files = expand_shar_range(se_shar_path["cuts"])
        target_audio_files = expand_shar_range(se_shar_path.get("target_audio", ""))

        if shard_idx >= len(cuts_files) or shard_idx >= len(target_audio_files):
            logging.warning(f"Shard index {shard_idx} out of range, skipping")
            continue

        cut_file = cuts_files[shard_idx]
        target_tar = target_audio_files[shard_idx]

        if not os.path.isfile(cut_file) or not os.path.isfile(target_tar):
            logging.warning(f"Missing shard files: cuts={cut_file}, target={target_tar}")
            continue

        needed_cut_ids = {cut_id for (_, cut_id) in spk_cut_pairs}
        cut_id_to_spk = {cut_id: spk for (spk, cut_id) in spk_cut_pairs}

        try:
            fields = {
                "cuts": [cut_file],
                "recording": [target_tar],
            }
            # Also include context_recording if available, to avoid errors
            context_audio_files = expand_shar_range(se_shar_path.get("context_audio", ""))
            if shard_idx < len(context_audio_files) and os.path.isfile(context_audio_files[shard_idx]):
                fields["context_recording"] = [context_audio_files[shard_idx]]

            shard_cutset = CutSet.from_shar(fields=fields)
            for cut in shard_cutset:
                if cut.id in needed_cut_ids:
                    spk = cut_id_to_spk[cut.id]
                    audio_np = cut.recording.resample(TITANET_SAMPLE_RATE).load_audio().squeeze(0)
                    audio_tensor = torch.from_numpy(audio_np).float()
                    speaker_audio_tensors[spk].append(audio_tensor)
                    needed_cut_ids.discard(cut.id)
                    if not needed_cut_ids:
                        break
        except Exception as e:
            logging.warning(f"Error loading shard {cut_file}: {e}")

    # Now compute embeddings in batches
    logging.info(f"[Stage 1] Computing TitaNet embeddings for {len(speaker_audio_tensors)} speakers...")
    all_speakers = list(speaker_audio_tensors.keys())

    for batch_start in tqdm(range(0, len(all_speakers), batch_size), desc="[Stage 1] TitaNet batches"):
        batch_speakers = all_speakers[batch_start : batch_start + batch_size]
        audio_list = []
        audio_lens = []
        spk_indices = []  # maps each audio in batch back to speaker

        for spk in batch_speakers:
            for audio_t in speaker_audio_tensors[spk]:
                audio_list.append(audio_t.to(device))
                audio_lens.append(audio_t.size(0))
                spk_indices.append(spk)

        if not audio_list:
            continue

        batch_lens = torch.tensor(audio_lens, device=device).long()
        max_len = int(batch_lens.max().item())
        padded = torch.zeros(len(audio_list), max_len, device=device, dtype=torch.float32)
        for i, t in enumerate(audio_list):
            padded[i, : t.size(0)] = t

        with torch.inference_mode():
            _, embeddings = sv_model.forward(input_signal=padded, input_signal_length=batch_lens)

        embeddings_np = embeddings.cpu().float().numpy()

        # Average embeddings per speaker
        spk_emb_accum: Dict[str, List[np.ndarray]] = defaultdict(list)
        for i, spk in enumerate(spk_indices):
            spk_emb_accum[spk].append(embeddings_np[i])

        for spk in batch_speakers:
            if spk in spk_emb_accum and spk_emb_accum[spk]:
                avg_emb = np.mean(spk_emb_accum[spk], axis=0)
                avg_emb = avg_emb / (np.linalg.norm(avg_emb) + 1e-8)
                speaker_embeddings[spk] = {
                    "language": speakers_needing_audio[spk]["language"],
                    "embedding": avg_emb,
                }

    logging.info(f"[Stage 1] Computed embeddings for {len(speaker_embeddings)} speakers")
    return speaker_embeddings


def run_stage1(
    lang_to_shar_entries: Dict[str, List[Dict]],
    samples_per_speaker: int,
    device: torch.device,
    index_path: str,
    batch_size: int = 16,
    max_shards_per_dataset: int = 0,
) -> Dict[str, Dict]:
    """Run full Stage 1: discover speakers, load audio, compute embeddings, save index."""
    if os.path.isfile(index_path):
        logging.info(f"[Stage 1] Loading cached speaker index from {index_path}")
        with open(index_path, 'rb') as f:
            return pickle.load(f)

    from nemo.collections.asr.models import EncDecSpeakerLabelModel

    logging.info(f"[Stage 1] Loading TitaNet model: {TITANET_MODEL_NAME}")
    sv_model = EncDecSpeakerLabelModel.from_pretrained(TITANET_MODEL_NAME)
    sv_model = sv_model.to(device)
    sv_model.eval()

    speaker_info = discover_speakers_from_cuts(
        lang_to_shar_entries,
        max_cuts_per_speaker=samples_per_speaker,
        max_shards_per_dataset=max_shards_per_dataset,
    )
    speaker_embeddings = compute_speaker_embeddings(speaker_info, sv_model, device, batch_size=batch_size)

    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    with open(index_path, 'wb') as f:
        pickle.dump(speaker_embeddings, f)
    logging.info(f"[Stage 1] Saved speaker index to {index_path}")

    del sv_model
    torch.cuda.empty_cache()
    return speaker_embeddings


# ---------------------------------------------------------------------------
# Stage 2: Cross-lingual speaker matching + language-balanced sampling
# ---------------------------------------------------------------------------


def build_crosslingual_map(speaker_embeddings: Dict[str, Dict]) -> Dict[str, Tuple[str, float]]:
    """
    For each speaker S in language L, find the closest speaker S' from a different
    language by cosine similarity of their TitaNet embeddings.

    Returns: {speaker_str: (best_match_speaker_str, cosine_similarity)}
    """
    speakers = list(speaker_embeddings.keys())
    n = len(speakers)
    logging.info(f"[Stage 2] Building cross-lingual map for {n} speakers...")

    # Build embedding matrix
    emb_matrix = np.stack([speaker_embeddings[s]["embedding"] for s in speakers])
    langs = [speaker_embeddings[s]["language"] for s in speakers]

    # Cosine similarity matrix (embeddings are already L2-normalized)
    sim_matrix = emb_matrix @ emb_matrix.T

    cross_lingual_map: Dict[str, Tuple[str, float]] = {}
    for i in range(n):
        best_j = -1
        best_sim = -2.0
        for j in range(n):
            if langs[j] == langs[i]:
                continue
            if sim_matrix[i, j] > best_sim:
                best_sim = sim_matrix[i, j]
                best_j = j
        if best_j >= 0:
            cross_lingual_map[speakers[i]] = (speakers[best_j], float(best_sim))
        else:
            logging.warning(f"No cross-lingual match found for speaker: {speakers[i]}")

    logging.info(f"[Stage 2] Built cross-lingual map with {len(cross_lingual_map)} entries")
    avg_sim = np.mean([v[1] for v in cross_lingual_map.values()]) if cross_lingual_map else 0
    logging.info(f"[Stage 2] Average cross-lingual similarity: {avg_sim:.4f}")
    return cross_lingual_map


def sample_balanced_cuts(
    lang_to_shar_entries: Dict[str, List[Dict]],
    cross_lingual_map: Dict[str, Tuple[str, float]],
    target_hours: float,
    seed: int,
    max_shards_per_dataset: int = 0,
) -> Tuple[Dict[str, List[Dict]], Dict[str, List[Dict]]]:
    """
    Sample cuts across languages so each language contributes approximately
    target_hours / num_languages hours of target audio.

    Args:
        max_shards_per_dataset: If > 0, only read this many shard files per
            dataset. Since we only need ~6.25h per language, reading a small
            fraction of shards is sufficient and avoids scanning tens of
            thousands of files for large datasets.

    Returns:
        target_cuts_by_lang: {lang: [list of cut_json dicts with extra metadata]}
        context_pool_by_speaker: {speaker_str: [list of (shar_entry, shard_idx, cut_json)]}
    """
    rng = random.Random(seed)
    num_langs = len(lang_to_shar_entries)
    hours_per_lang = target_hours / num_langs
    secs_per_lang = hours_per_lang * 3600
    # Collect 3x the target to allow shuffling diversity
    collect_secs_per_lang = secs_per_lang * 3

    logging.info(
        f"[Stage 2] Sampling ~{hours_per_lang:.2f}h per language ({num_langs} languages, {target_hours}h total)"
    )

    all_matched_speakers = set(v[0] for v in cross_lingual_map.values())

    target_cuts_by_lang: Dict[str, List[Dict]] = {}
    context_pool_by_speaker: Dict[str, List] = defaultdict(list)

    for lang, shar_entries in lang_to_shar_entries.items():
        logging.info(f"[Stage 2] Reading cuts for language: {lang}")
        lang_cuts = []
        lang_collected_secs = 0.0
        lang_done = False

        for se in shar_entries:
            if lang_done:
                break
            cuts_pattern = se["shar_path"]["cuts"]
            cuts_files = expand_shar_range(cuts_pattern)
            if max_shards_per_dataset > 0 and len(cuts_files) > max_shards_per_dataset:
                cuts_files = cuts_files[:max_shards_per_dataset]
                logging.info(
                    f"  Limiting to {max_shards_per_dataset} shards for dataset: " f"{se['shar_path']['cuts']}"
                )
            for cuts_file in cuts_files:
                if lang_done:
                    break
                if not os.path.isfile(cuts_file):
                    continue
                shard_idx_match = re.search(r"cuts\.(\d+)\.jsonl\.gz$", cuts_file)
                shard_idx = int(shard_idx_match.group(1)) if shard_idx_match else 0
                try:
                    with gzip.open(cuts_file, 'rt', encoding='utf-8') as f:
                        for line in f:
                            cut_json = json.loads(line)
                            speaker_str = cut_json.get("supervisions", [{}])[0].get("speaker", "")
                            if not speaker_str:
                                continue
                            if speaker_str in all_matched_speakers:
                                context_pool_by_speaker[speaker_str].append((se, shard_idx, cut_json))
                            if speaker_str in cross_lingual_map:
                                cut_json["_shar_entry"] = se
                                cut_json["_shard_idx"] = shard_idx
                                cut_json["_speaker_str"] = speaker_str
                                lang_cuts.append(cut_json)
                                lang_collected_secs += cut_json.get("duration", 0)
                                if lang_collected_secs >= collect_secs_per_lang:
                                    lang_done = True
                                    break
                except Exception as e:
                    logging.warning(f"Error reading {cuts_file}: {e}")

        logging.info(f"  {lang}: {len(lang_cuts)} candidate target cuts ({lang_collected_secs / 3600:.2f}h collected)")

        rng.shuffle(lang_cuts)
        sampled = []
        total_dur = 0.0
        for cut_json in lang_cuts:
            dur = cut_json.get("duration", 0)
            if dur <= 0:
                continue
            sampled.append(cut_json)
            total_dur += dur
            if total_dur >= secs_per_lang:
                break

        target_cuts_by_lang[lang] = sampled
        logging.info(f"  {lang}: sampled {len(sampled)} cuts, {total_dur / 3600:.2f}h")

    total_sampled = sum(len(v) for v in target_cuts_by_lang.values())
    total_hours = sum(sum(c.get("duration", 0) for c in v) for v in target_cuts_by_lang.values()) / 3600
    logging.info(f"[Stage 2] Total sampled: {total_sampled} cuts, {total_hours:.2f}h")
    return target_cuts_by_lang, dict(context_pool_by_speaker)


# ---------------------------------------------------------------------------
# Stage 3: Extract audio + write NeMo manifest
# ---------------------------------------------------------------------------


def run_stage3(
    target_cuts_by_lang: Dict[str, List[Dict]],
    context_pool_by_speaker: Dict[str, List],
    cross_lingual_map: Dict[str, Tuple[str, float]],
    speaker_embeddings: Dict[str, Dict],
    output_dir: str,
    sample_rate: int,
    seed: int,
):
    """
    For each sampled target cut, pick a context utterance from the matched
    cross-lingual speaker, extract both audios to disk, and write the manifest.
    """
    rng = random.Random(seed)
    audio_dir = os.path.join(output_dir, "extracted_audio")
    target_audio_dir = os.path.join(audio_dir, "target")
    context_audio_dir = os.path.join(audio_dir, "context")
    os.makedirs(target_audio_dir, exist_ok=True)
    os.makedirs(context_audio_dir, exist_ok=True)

    manifest_path = os.path.join(output_dir, "manifest.json")

    # Build a quick lookup: for each context cut we might need to load,
    # index by (shar_key, shard_idx, cut_id)
    # First, assign a context cut to each target
    assignments: List[Dict] = []
    for lang, cuts in target_cuts_by_lang.items():
        for cut_json in cuts:
            spk = cut_json["_speaker_str"]
            matched_spk, ssim = cross_lingual_map[spk]
            ctx_pool = context_pool_by_speaker.get(matched_spk, [])
            if not ctx_pool:
                logging.warning(
                    f"No context pool for matched speaker {matched_spk}, skipping cut {cut_json.get('id', '')}"
                )
                continue
            ctx_se, ctx_shard_idx, ctx_cut_json = rng.choice(ctx_pool)
            assignments.append(
                {
                    "target_cut_json": cut_json,
                    "target_shar_entry": cut_json["_shar_entry"],
                    "target_shard_idx": cut_json["_shard_idx"],
                    "target_speaker": spk,
                    "context_cut_json": ctx_cut_json,
                    "context_shar_entry": ctx_se,
                    "context_shard_idx": ctx_shard_idx,
                    "context_speaker": matched_spk,
                    "ssim": ssim,
                    "lang": lang,
                }
            )

    logging.info(f"[Stage 3] Total assignments: {len(assignments)}")

    # Group by (shar_key, shard_idx) for efficient loading
    # We need to load target and context audio from potentially different shards
    # Strategy: process all assignments, grouping audio loads by shard
    target_loads: Dict[Tuple[str, int], List[int]] = defaultdict(list)
    context_loads: Dict[Tuple[str, int], List[int]] = defaultdict(list)

    for idx, a in enumerate(assignments):
        t_shar_key = json.dumps(a["target_shar_entry"]["shar_path"], sort_keys=True)
        target_loads[(t_shar_key, a["target_shard_idx"])].append(idx)
        c_shar_key = json.dumps(a["context_shar_entry"]["shar_path"], sort_keys=True)
        context_loads[(c_shar_key, a["context_shard_idx"])].append(idx)

    # Arrays to hold extracted audio file paths
    target_audio_paths = [None] * len(assignments)
    context_audio_paths = [None] * len(assignments)

    def _save_audio_from_shard(
        shard_loads: Dict[Tuple[str, int], List[int]],
        assignments_list: List[Dict],
        cut_json_key: str,
        out_subdir: str,
        out_paths_array: List,
        audio_field: str,
    ):
        """Load cuts from shar tars and save individual audio files to disk."""
        total_shards = len(shard_loads)
        for (shar_key_str, shard_idx), indices in tqdm(
            shard_loads.items(), desc=f"[Stage 3] Extracting {audio_field}", total=total_shards
        ):
            se_shar_path = json.loads(shar_key_str)
            cuts_files = expand_shar_range(se_shar_path["cuts"])
            target_audio_files = expand_shar_range(se_shar_path.get("target_audio", ""))

            if shard_idx >= len(cuts_files) or shard_idx >= len(target_audio_files):
                logging.warning(f"Shard {shard_idx} out of range, skipping")
                continue

            cut_file = cuts_files[shard_idx]
            tar_file = target_audio_files[shard_idx]

            if not os.path.isfile(cut_file) or not os.path.isfile(tar_file):
                logging.warning(f"Missing files: {cut_file} or {tar_file}")
                continue

            needed_cut_ids = {}
            for i in indices:
                cj = assignments_list[i][cut_json_key]
                cid = cj.get("id", "")
                needed_cut_ids[cid] = i

            try:
                fields = {"cuts": [cut_file], "recording": [tar_file]}
                ctx_audio_files = expand_shar_range(se_shar_path.get("context_audio", ""))
                if ctx_audio_files and shard_idx < len(ctx_audio_files) and os.path.isfile(ctx_audio_files[shard_idx]):
                    fields["context_recording"] = [ctx_audio_files[shard_idx]]

                shard_cutset = CutSet.from_shar(fields=fields)
                for cut in shard_cutset:
                    if cut.id in needed_cut_ids:
                        assign_idx = needed_cut_ids[cut.id]
                        audio_np = cut.recording.resample(sample_rate).load_audio().squeeze(0)
                        safe_id = cut.id.replace("/", "_")
                        out_file = os.path.join(out_subdir, f"{safe_id}.wav")
                        sf.write(out_file, audio_np, sample_rate)
                        out_paths_array[assign_idx] = os.path.relpath(
                            out_file, os.path.join(output_dir, "extracted_audio")
                        )
                        del needed_cut_ids[cut.id]
                        if not needed_cut_ids:
                            break
            except Exception as e:
                logging.warning(f"Error processing shard {cut_file}: {e}")

    # Extract target audio
    logging.info(f"[Stage 3] Extracting target audio from {len(target_loads)} shards...")
    _save_audio_from_shard(
        target_loads,
        assignments,
        "target_cut_json",
        target_audio_dir,
        target_audio_paths,
        "target_audio",
    )

    # Extract context audio
    logging.info(f"[Stage 3] Extracting context audio from {len(context_loads)} shards...")
    _save_audio_from_shard(
        context_loads,
        assignments,
        "context_cut_json",
        context_audio_dir,
        context_audio_paths,
        "context_audio",
    )

    # Write manifest
    logging.info(f"[Stage 3] Writing manifest to {manifest_path}")
    written = 0
    skipped = 0
    with open(manifest_path, 'w', encoding='utf-8') as f:
        for idx, a in enumerate(assignments):
            if target_audio_paths[idx] is None or context_audio_paths[idx] is None:
                skipped += 1
                continue

            t_cut = a["target_cut_json"]
            c_cut = a["context_cut_json"]
            t_sup = t_cut.get("supervisions", [{}])[0]

            text = t_sup.get("text", "")
            normalized_text = t_sup.get("custom", {}).get("normalized_text", text)
            ipa = t_sup.get("custom", {}).get("ipa", "")
            speaker = t_sup.get("speaker", "")
            duration = t_cut.get("duration", 0)
            context_duration = c_cut.get("duration", 0)
            ctx_lang_parsed, _, _ = parse_speaker_field(a["context_speaker"])

            target_lang_parsed, _, _ = parse_speaker_field(speaker)

            entry = {
                "audio_filepath": target_audio_paths[idx],
                "text": text,
                "normalized_text": normalized_text,
                "speaker": speaker,
                "language": target_lang_parsed,
                "duration": duration,
                "context_audio_filepath": context_audio_paths[idx],
                "context_audio_duration": context_duration,
                "context_speaker_similarity": round(a["ssim"], 6),
                "context_language": ctx_lang_parsed,
                "context_speaker": a["context_speaker"],
            }
            if ipa:
                entry["ipa"] = ipa

            # Carry over any additional custom fields from the target supervision
            _exclude_custom_keys = {
                "target_audio_codes_path",
                "context_audio_codes_path",
                "context_audio_text",
                "context_audio_normalized_text",
                "context_audio_offset",
            }
            for k, v in t_sup.get("custom", {}).items():
                if k not in entry and k not in _exclude_custom_keys:
                    entry[k] = v

            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            written += 1

    logging.info(f"[Stage 3] Manifest written: {written} entries, {skipped} skipped")
    return manifest_path


# ---------------------------------------------------------------------------
# YAML config generation (post Stage 4)
# ---------------------------------------------------------------------------


def generate_yaml_config(lhotse_shar_dir: str, output_yaml_path: str, data_mount_prefix: str = "/data"):
    """
    Generate a lhotse YAML config pointing to the cross-lingual shar dataset.
    Call this after running create_lhotse_shar_from_nemo_manifest.py on the manifest.

    Args:
        lhotse_shar_dir: Absolute path to the lhotse_shar output directory
                         (containing cuts/, target_audio/, context_audio/).
        output_yaml_path: Path to write the YAML config file.
        data_mount_prefix: If shar_dir is under a mount, replace the host prefix
                           with this docker-internal prefix. Pass empty string to skip.
    """
    cuts_dir = os.path.join(lhotse_shar_dir, "cuts")
    target_audio_dir = os.path.join(lhotse_shar_dir, "target_audio")
    context_audio_dir = os.path.join(lhotse_shar_dir, "context_audio")

    cuts_files = sorted(glob_module.glob(os.path.join(cuts_dir, "cuts.*.jsonl.gz")))
    context_files = sorted(glob_module.glob(os.path.join(context_audio_dir, "recording.*.tar")))

    if not cuts_files:
        logging.error(f"No cut files found in {cuts_dir}")
        return

    # Determine shard range
    first_idx = int(re.search(r"cuts\.(\d+)\.jsonl\.gz$", cuts_files[0]).group(1))
    last_idx = int(re.search(r"cuts\.(\d+)\.jsonl\.gz$", cuts_files[-1]).group(1))
    width = len(re.search(r"cuts\.(\d+)\.jsonl\.gz$", cuts_files[0]).group(1))

    def _make_range_pattern(directory: str, prefix: str, ext: str) -> str:
        path = os.path.join(directory, f"{prefix}.{{{first_idx:0{width}d}..{last_idx:0{width}d}}}.{ext}")
        return path

    shar_path = {
        "cuts": _make_range_pattern(cuts_dir, "cuts", "jsonl.gz"),
        "target_audio": _make_range_pattern(target_audio_dir, "recording", "tar"),
    }
    if context_files:
        shar_path["context_audio"] = _make_range_pattern(context_audio_dir, "recording", "tar")

    # Check for codec codes
    for codec_dir_name in os.listdir(lhotse_shar_dir):
        codec_subdir = os.path.join(lhotse_shar_dir, codec_dir_name)
        if not os.path.isdir(codec_subdir):
            continue
        target_codes_dir = os.path.join(codec_subdir, "target_codes")
        context_codes_dir = os.path.join(codec_subdir, "context_codes")
        if os.path.isdir(target_codes_dir):
            tc_files = sorted(glob_module.glob(os.path.join(target_codes_dir, "codes.*.tar")))
            if tc_files:
                tc_first = int(re.search(r"codes\.(\d+)\.tar$", tc_files[0]).group(1))
                tc_last = int(re.search(r"codes\.(\d+)\.tar$", tc_files[-1]).group(1))
                tc_width = len(re.search(r"codes\.(\d+)\.tar$", tc_files[0]).group(1))
                shar_path["target_codes"] = os.path.join(
                    target_codes_dir, f"codes.{{{tc_first:0{tc_width}d}..{tc_last:0{tc_width}d}}}.tar"
                )
        if os.path.isdir(context_codes_dir):
            cc_files = sorted(glob_module.glob(os.path.join(context_codes_dir, "codes.*.tar")))
            if cc_files:
                cc_first = int(re.search(r"codes\.(\d+)\.tar$", cc_files[0]).group(1))
                cc_last = int(re.search(r"codes\.(\d+)\.tar$", cc_files[-1]).group(1))
                cc_width = len(re.search(r"codes\.(\d+)\.tar$", cc_files[0]).group(1))
                shar_path["context_codes"] = os.path.join(
                    context_codes_dir, f"codes.{{{cc_first:0{cc_width}d}..{cc_last:0{cc_width}d}}}.tar"
                )

    yaml_entry = [
        {
            "type": "lhotse_shar",
            "shar_path": shar_path,
            "weight": 1.0,
            "tags": {
                "task": "tts",
                "lang": "crosslingual",
                "tokenizer_names": ["nemotron_nano_30b"],
            },
        }
    ]

    os.makedirs(os.path.dirname(output_yaml_path) or ".", exist_ok=True)
    with open(output_yaml_path, 'w') as f:
        yaml.dump(yaml_entry, f, default_flow_style=False, sort_keys=False)
    logging.info(f"YAML config written to {output_yaml_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Create a cross-lingual context TTS dataset from multilingual lhotse shar data.",
    )
    parser.add_argument(
        "--master-yaml",
        required=True,
        type=str,
        help="Path to the master multilingual YAML (e.g. train_25fpsSpectralCodecBWE_en_de_es_fr_hi_it_vi_zh_with_ipa.yaml).",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=str,
        help="Base directory for all outputs (extracted audio, manifest, speaker index).",
    )
    parser.add_argument(
        "--target-hours",
        type=float,
        default=50.0,
        help="Total hours of target audio to sample (split equally across languages).",
    )
    parser.add_argument(
        "--samples-per-speaker",
        type=int,
        default=5,
        help="Number of utterances per speaker to use for computing the average TitaNet embedding.",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=24000,
        help="Sample rate for saving extracted audio files.",
    )
    parser.add_argument(
        "--embedding-batch-size",
        type=int,
        default=16,
        help="Batch size for TitaNet embedding computation.",
    )
    parser.add_argument(
        "--max-shards-per-dataset",
        type=int,
        default=0,
        help="Max number of .jsonl.gz shard files to scan per dataset during "
        "speaker discovery (Stage 1). 0 means scan all shards. "
        "Setting this to e.g. 10 dramatically speeds up discovery while "
        "still finding most speakers.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level.",
    )
    parser.add_argument(
        "--generate-yaml",
        type=str,
        default=None,
        help="If provided, skip stages 1-3 and instead generate a YAML config "
        "pointing to the lhotse shar in OUTPUT_DIR/lhotse_shar. "
        "Value is the output YAML file path.",
    )
    args = parser.parse_args()

    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
    )
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    # --- Generate YAML config mode (post Stage 4) ---
    if args.generate_yaml:
        lhotse_shar_dir = os.path.join(args.output_dir, "lhotse_shar")
        generate_yaml_config(lhotse_shar_dir, args.generate_yaml)
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # --- Parse master YAML ---
    logging.info(f"Parsing master YAML: {args.master_yaml}")
    lang_to_shar_entries = parse_master_yaml(args.master_yaml)
    if not lang_to_shar_entries:
        logging.error("No shar entries found. Check the master YAML path and contents.")
        return

    for lang, entries in sorted(lang_to_shar_entries.items()):
        logging.info(f"  Language '{lang}': {len(entries)} shar groups (with context_audio)")

    # --- Stage 1: Build speaker embedding index ---
    index_path = os.path.join(args.output_dir, "speaker_embedding_index.pkl")
    speaker_embeddings = run_stage1(
        lang_to_shar_entries,
        samples_per_speaker=args.samples_per_speaker,
        device=device,
        index_path=index_path,
        batch_size=args.embedding_batch_size,
        max_shards_per_dataset=args.max_shards_per_dataset,
    )

    # --- Stage 2: Cross-lingual matching + balanced sampling ---
    cross_lingual_map = build_crosslingual_map(speaker_embeddings)
    target_cuts_by_lang, context_pool_by_speaker = sample_balanced_cuts(
        lang_to_shar_entries,
        cross_lingual_map,
        target_hours=args.target_hours,
        seed=args.seed,
        max_shards_per_dataset=args.max_shards_per_dataset,
    )

    # --- Stage 3: Extract audio + write manifest ---
    manifest_path = run_stage3(
        target_cuts_by_lang,
        context_pool_by_speaker,
        cross_lingual_map,
        speaker_embeddings,
        args.output_dir,
        args.sample_rate,
        args.seed,
    )

    # --- Summary ---
    logging.info("=" * 60)
    logging.info("Cross-lingual context dataset creation complete!")
    logging.info(f"  Manifest: {manifest_path}")
    logging.info(f"  Audio dir: {os.path.join(args.output_dir, 'extracted_audio')}")
    logging.info("")
    logging.info("Next steps:")
    logging.info("  1. Convert to lhotse shar format:")
    logging.info(f"     python scripts/magpietts/create_lhotse_shar_from_nemo_manifest.py \\")
    logging.info(f"       --manifest-path {manifest_path} \\")
    logging.info(f"       --audio-base-dir {os.path.join(args.output_dir, 'extracted_audio')} \\")
    logging.info(f"       --output-dir {os.path.join(args.output_dir, 'lhotse_shar')} \\")
    logging.info(f"       --num-jobs 16 --processing-chunk-size 256 --audio-format flac --shuffle --shuffle-seed 42")
    logging.info("")
    logging.info("  2. (Optional) Add codec codes:")
    logging.info(f"     python scripts/magpietts/extend_lhotse_shards_with_audio_codes.py \\")
    logging.info(f"       --cuts-dir {os.path.join(args.output_dir, 'lhotse_shar', 'cuts')} \\")
    logging.info(f"       --target-audio-dir {os.path.join(args.output_dir, 'lhotse_shar', 'target_audio')} \\")
    logging.info(f"       --context-audio-dir {os.path.join(args.output_dir, 'lhotse_shar', 'context_audio')} \\")
    logging.info(f"       --output-dir {os.path.join(args.output_dir, 'lhotse_shar')} \\")
    logging.info(f"       --codec-model-path <YOUR_CODEC_MODEL_PATH>")
    logging.info("")
    yaml_out = os.path.join(args.output_dir, "crosslingual_context.yaml")
    logging.info("  3. Generate YAML config for training:")
    logging.info(f"     python scripts/magpietts/create_crosslingual_context_dataset.py \\")
    logging.info(f"       --master-yaml {args.master_yaml} \\")
    logging.info(f"       --output-dir {args.output_dir} \\")
    logging.info(f"       --generate-yaml {yaml_out}")
    logging.info("=" * 60)


if __name__ == "__main__":
    main()
