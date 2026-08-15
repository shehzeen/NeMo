# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

"""Convert synchronized David AI two-speaker conversations to Lhotse SHAR.

The input directory must contain ``0_metadata.csv``. Each conversation has two
metadata rows, one per isolated, postprocessed mono WAV track, and a transcript
JSON shared by both rows.
"""

import argparse
import csv
import json
import logging
import math
import os
import re
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import soundfile as sf
import yaml
from lhotse import AudioSource, MonoCut, Recording, SupervisionSegment, compute_num_samples, fastcopy
from lhotse.shar.writers import AudioTarWriter, JsonlShardWriter

MAX_CLIP_TOLERANCE = 0.020
MAX_GAP_DURATION_COLLAPSE_TURNS = 1.0
EXPLICITLY_REJECTED_CONVERSATIONS = {"abaa0bcf-7e5c-41e5-8677-f2f9b57d2944"}
ROLE_MAP = {
    "D7": {"Asker": "user", "Expert": "agent"},
    "D6b": {"User": "user", "Advisor": "agent"},
    "D6a": {"Host": "user", "Guest": "agent"},
}
CONVERSATION_ID_FIELDS = ("conference_id", "conversation_id", "conversation_uuid", "uuid")
TRANSCRIPT_PATH_FIELDS = (
    "machine_generated_transcription_file_path",
    "transcript_file_path",
    "transcript_path",
    "transcript_json_file_path",
    "transcript_json_path",
    "transcription_file_path",
)
AUDIO_PATH_FIELDS = ("postprocessed_audio_file_path",)
SPEAKER_ID_FIELDS = ("speaker_id", "participant_id", "speaker", "speaker_name")
ROLE_FIELDS = ("role", "participant_role")


class ConversationError(ValueError):
    """An expected data-quality rejection for one conversation."""


@dataclass(frozen=True)
class MetadataRow:
    conference_id: str
    sku: str
    speaker_id: str
    role: str | None
    audio_path: Path
    transcript_path: Path
    values: dict[str, Any] = field(compare=False)


@dataclass(frozen=True)
class ConversationMetadata:
    conference_id: str
    sku: str
    rows: tuple[MetadataRow, MetadataRow]


@dataclass(frozen=True)
class TranscriptTurn:
    start: float
    end: float
    text: str
    original_speaker: str
    speaker: str
    speaker_id: str
    demographics: dict[str, Any] = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass(frozen=True)
class PreparedConversation:
    cut: MonoCut
    user_audio: np.ndarray
    agent_audio: np.ndarray
    sampling_rate: int


@dataclass(frozen=True)
class ValidatedConversation:
    metadata: ConversationMetadata
    user_row: MetadataRow
    agent_row: MetadataRow
    turns: tuple[TranscriptTurn, ...]
    sampling_rate: int
    num_samples: int

    @property
    def duration(self) -> float:
        return self.num_samples / self.sampling_rate


def _first(row: Mapping[str, Any], names: Sequence[str], *, required: bool = True) -> Any:
    for name in names:
        value = row.get(name)
        if value is not None and str(value).strip():
            return value
    if required:
        raise ConversationError(f"missing required field (one of: {', '.join(names)})")
    return None


def _typed_value(value: str) -> Any:
    value = value.strip()
    if not value:
        return None
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    if value[0:1] in {"[", "{"}:
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    try:
        return int(value)
    except ValueError:
        try:
            number = float(value)
            return number if math.isfinite(number) else value
        except ValueError:
            return value


def resolve_path(root: Path, value: str | Path) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else root / path


def parse_metadata_csv(input_root: Path) -> list[ConversationMetadata]:
    metadata_path = input_root / "0_metadata.csv"
    with metadata_path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames:
            raise ConversationError(f"{metadata_path} has no header")
        parsed_rows: list[MetadataRow] = []
        for line_number, raw in enumerate(reader, start=2):
            values = {key: _typed_value(value or "") for key, value in raw.items()}
            try:
                conference_id = str(_first(values, CONVERSATION_ID_FIELDS))
                sku = str(_first(values, ("sku", "SKU", "dataset_sku")))
                speaker_id = str(_first(values, SPEAKER_ID_FIELDS))
                role_value = _first(values, ROLE_FIELDS, required=False)
                role = str(role_value) if role_value is not None else None
                audio_path = resolve_path(input_root, _first(values, AUDIO_PATH_FIELDS))
                transcript_path = resolve_path(input_root, _first(values, TRANSCRIPT_PATH_FIELDS))
            except ConversationError as error:
                raise ConversationError(f"{metadata_path}:{line_number}: {error}") from error
            parsed_rows.append(
                MetadataRow(
                    conference_id=conference_id,
                    sku=sku,
                    speaker_id=speaker_id,
                    role=role,
                    audio_path=audio_path,
                    transcript_path=transcript_path,
                    values=values,
                )
            )

    grouped: dict[str, list[MetadataRow]] = {}
    for row in parsed_rows:
        grouped.setdefault(row.conference_id, []).append(row)
    conversations = []
    for conference_id, rows in grouped.items():
        if len(rows) != 2:
            raise ConversationError(f"{conference_id}: expected exactly two metadata rows, found {len(rows)}")
        if rows[0].sku != rows[1].sku:
            raise ConversationError(f"{conference_id}: metadata rows disagree on sku")
        if rows[0].transcript_path != rows[1].transcript_path:
            raise ConversationError(f"{conference_id}: metadata rows disagree on transcript path")
        conversations.append(ConversationMetadata(conference_id, rows[0].sku, (rows[0], rows[1])))
    return sorted(conversations, key=lambda item: item.conference_id)


def sku_family(sku: str) -> str:
    match = re.match(r"(?i)^(D6[ab]|D7|D1)(?:\b|[_-])", sku.strip())
    if not match:
        raise ConversationError(f"unsupported sku: {sku}")
    value = match.group(1)
    return "D6a" if value.lower() == "d6a" else "D6b" if value.lower() == "d6b" else value.upper()


def canonical_speaker_map(
    sku: str, metadata_rows: Iterable[MetadataRow], transcript_speakers: Iterable[str]
) -> dict[str, str]:
    family = sku_family(sku)
    metadata_rows = list(metadata_rows)
    metadata_speaker_ids = {row.speaker_id for row in metadata_rows}
    transcript_speakers = list(transcript_speakers)
    if set(transcript_speakers) != metadata_speaker_ids:
        raise ConversationError(
            f"transcript speakers {sorted(set(transcript_speakers))} do not match metadata "
            f"speakers {sorted(metadata_speaker_ids)}"
        )
    if family == "D1":
        distinct = list(dict.fromkeys(str(speaker).strip() for speaker in transcript_speakers))
        if len(distinct) != 2:
            raise ConversationError(f"D1 transcript must contain exactly two distinct speakers, found {len(distinct)}")
        return {distinct[0]: "user", distinct[1]: "agent"}

    expected = ROLE_MAP[family]
    lookup = {key.casefold(): value for key, value in expected.items()}
    mapping = {}
    for row in metadata_rows:
        canonical = lookup.get(str(row.role or "").strip().casefold())
        if canonical is None:
            raise ConversationError(f"{family} has unexpected metadata speaker role: {row.role}")
        mapping[row.speaker_id] = canonical
    if set(mapping.values()) != {"user", "agent"}:
        raise ConversationError(f"{family} metadata does not contain one user and one agent row")
    return mapping


def _transcript_items(document: Any) -> list[Mapping[str, Any]]:
    if isinstance(document, list):
        items = document
    elif isinstance(document, dict):
        items = None
        for key in ("segments", "utterances", "turns", "transcript", "transcription"):
            if isinstance(document.get(key), list):
                items = document[key]
                break
        if items is None:
            raise ConversationError("transcript JSON has no segment list")
    else:
        raise ConversationError("transcript JSON must be an object or list")
    if not all(isinstance(item, dict) for item in items):
        raise ConversationError("transcript segments must be JSON objects")
    return items


def parse_transcript_document(document: Any) -> list[dict[str, Any]]:
    parsed = []
    for index, item in enumerate(_transcript_items(document)):
        speaker = _first(item, ("speaker", "speaker_name", "speaker_id", "speaker_label", "role"))
        text = _first(item, ("text", "transcript", "utterance"), required=False)
        start = _first(item, ("start", "start_time", "start_seconds", "start_time_seconds"))
        end = _first(item, ("end", "end_time", "end_seconds", "end_time_seconds"), required=False)
        duration = _first(item, ("duration", "duration_seconds"), required=False)
        try:
            start = float(start)
            end = float(end) if end is not None else start + float(duration)
        except (TypeError, ValueError) as error:
            raise ConversationError(f"transcript segment {index} has non-numeric timing") from error
        parsed.append(
            {
                "start": start,
                "end": end,
                "text": "" if text is None else str(text),
                "speaker": str(speaker).strip(),
                "_index": index,
            }
        )
    return parsed


def _row_demographics(row: MetadataRow) -> dict[str, Any]:
    demographic_fields = (
        "age_range",
        "gender",
        "country_of_birth",
        "state_of_residence",
        "country_of_residence",
        "locale",
        "channel_flag_scores",
        "redelivered",
    )
    demographics = row.values.get("demographics")
    result = dict(demographics) if isinstance(demographics, dict) else {}
    result.update({key: row.values[key] for key in demographic_fields if row.values.get(key) is not None})
    return result


def _conversation_metadata(metadata: ConversationMetadata) -> dict[str, Any]:
    explicit = [row.values.get("conversation_metadata") for row in metadata.rows]
    if any(value is not None for value in explicit):
        if explicit[0] != explicit[1] or not isinstance(explicit[0], dict):
            raise ConversationError(f"{metadata.conference_id}: inconsistent conversation_metadata")
        return dict(explicit[0])
    excluded = {
        *CONVERSATION_ID_FIELDS,
        *TRANSCRIPT_PATH_FIELDS,
        *AUDIO_PATH_FIELDS,
        *SPEAKER_ID_FIELDS,
        *ROLE_FIELDS,
        "speaker_id",
        "participant_id",
        "demographics",
        "sku",
        "SKU",
        "dataset_sku",
    }
    return {
        key: value
        for key, value in metadata.rows[0].values.items()
        if key not in excluded and value is not None and metadata.rows[1].values.get(key) == value
    }


def normalize_transcript(
    segments: Sequence[Mapping[str, Any]],
    duration: float,
    speaker_mapping: Mapping[str, str],
    speaker_metadata: Mapping[str, tuple[str, Mapping[str, Any]]] | None = None,
    max_gap: float = MAX_GAP_DURATION_COLLAPSE_TURNS,
) -> list[TranscriptTurn]:
    """Validate, clip, filter, sort, and collapse transcript segments."""
    if not math.isfinite(duration) or duration <= 0:
        raise ConversationError(f"audio duration must be finite and positive, got {duration}")
    normalized: list[tuple[int, TranscriptTurn]] = []
    for index, segment in enumerate(segments):
        try:
            start, end = float(segment["start"]), float(segment["end"])
        except (KeyError, TypeError, ValueError) as error:
            raise ConversationError(f"transcript segment {index} has invalid timing") from error
        if not math.isfinite(start) or not math.isfinite(end):
            raise ConversationError(f"transcript segment {index} has non-finite timing")
        if (
            start < -MAX_CLIP_TOLERANCE
            or start > duration + MAX_CLIP_TOLERANCE
            or end < -MAX_CLIP_TOLERANCE
            or end > duration + MAX_CLIP_TOLERANCE
        ):
            raise ConversationError(
                f"transcript segment {index} lies outside audio bounds: [{start}, {end}] vs {duration}"
            )
        start, end = min(duration, max(0.0, start)), min(duration, max(0.0, end))
        raw_text = segment.get("text", "")
        text = "" if raw_text is None else str(raw_text).strip()
        if end <= start or not any(character.isalnum() for character in text):
            continue
        original = str(segment.get("speaker", "")).strip()
        canonical = speaker_mapping.get(original)
        if canonical is None:
            folded = {key.casefold(): value for key, value in speaker_mapping.items()}
            canonical = folded.get(original.casefold())
        if canonical not in {"user", "agent"}:
            raise ConversationError(f"unknown transcript speaker: {original}")
        speaker_id, demographics = original, {}
        if speaker_metadata:
            metadata = speaker_metadata.get(original) or speaker_metadata.get(canonical)
            if metadata:
                speaker_id, demographics = metadata
        normalized.append(
            (
                int(segment.get("_index", index)),
                TranscriptTurn(start, end, text, original, canonical, str(speaker_id), dict(demographics)),
            )
        )

    normalized.sort(key=lambda item: (item[1].start, item[1].end, item[0]))
    collapsed: list[TranscriptTurn] = []
    for _, turn in normalized:
        if collapsed and collapsed[-1].speaker == turn.speaker and turn.start - collapsed[-1].end < max_gap:
            previous = collapsed[-1]
            collapsed[-1] = TranscriptTurn(
                start=previous.start,
                end=max(previous.end, turn.end),
                text=f"{previous.text} {turn.text}".strip(),
                original_speaker=previous.original_speaker,
                speaker=previous.speaker,
                speaker_id=previous.speaker_id,
                demographics=previous.demographics,
            )
        else:
            collapsed.append(turn)
    if not collapsed:
        raise ConversationError("transcript has no usable segments")
    if {turn.speaker for turn in collapsed} != {"user", "agent"}:
        raise ConversationError("transcript must contain usable speech from both speakers")
    # This catches gross unit/schema mistakes while permitting natural cross-talk.
    if sum(turn.duration for turn in collapsed) > duration * 2.5 + MAX_CLIP_TOLERANCE:
        raise ConversationError("implausible alignment: total supervision duration exceeds 2.5x audio duration")
    return collapsed


def load_synchronized_audio(user_path: Path, agent_path: Path) -> tuple[np.ndarray, np.ndarray, int]:
    user_audio, user_sr = sf.read(user_path, dtype="float32", always_2d=True)
    agent_audio, agent_sr = sf.read(agent_path, dtype="float32", always_2d=True)
    if user_audio.shape[1] != 1 or agent_audio.shape[1] != 1:
        raise ConversationError("postprocessed tracks must be mono")
    if user_sr != agent_sr:
        raise ConversationError(f"track sample rates differ: {user_sr} vs {agent_sr}")
    mismatch = abs(len(user_audio) - len(agent_audio))
    if mismatch > 1:
        raise ConversationError(f"synchronized tracks differ by {mismatch} samples")
    shared_length = min(len(user_audio), len(agent_audio))
    if shared_length == 0:
        raise ConversationError("audio tracks are empty")
    return user_audio[:shared_length, 0], agent_audio[:shared_length, 0], user_sr


def inspect_synchronized_audio(user_path: Path, agent_path: Path) -> tuple[int, int]:
    user_info, agent_info = sf.info(user_path), sf.info(agent_path)
    if user_info.channels != 1 or agent_info.channels != 1:
        raise ConversationError("postprocessed tracks must be mono")
    if user_info.samplerate != agent_info.samplerate:
        raise ConversationError(f"track sample rates differ: {user_info.samplerate} vs {agent_info.samplerate}")
    mismatch = abs(user_info.frames - agent_info.frames)
    if mismatch > 1:
        raise ConversationError(f"synchronized tracks differ by {mismatch} samples")
    num_samples = min(user_info.frames, agent_info.frames)
    if num_samples == 0:
        raise ConversationError("audio tracks are empty")
    return user_info.samplerate, num_samples


def to_shar_placeholder(recording: Recording, cut: MonoCut) -> Recording:
    return fastcopy(
        recording,
        id=cut.id,
        sources=[AudioSource(type="shar", channels=recording.channel_ids, source="")],
        transforms=None,
        duration=cut.duration,
        num_samples=compute_num_samples(cut.duration, recording.sampling_rate),
    )


def validate_conversation(metadata: ConversationMetadata) -> ValidatedConversation:
    if metadata.conference_id in EXPLICITLY_REJECTED_CONVERSATIONS:
        raise ConversationError("explicitly rejected known-bad conversation")
    with metadata.rows[0].transcript_path.open(encoding="utf-8") as stream:
        segments = parse_transcript_document(json.load(stream))
    transcript_speakers = [segment["speaker"] for segment in segments]
    mapping = canonical_speaker_map(metadata.sku, metadata.rows, transcript_speakers)
    row_by_role = {mapping[row.speaker_id]: row for row in metadata.rows}
    user_row, agent_row = row_by_role["user"], row_by_role["agent"]
    sampling_rate, num_samples = inspect_synchronized_audio(user_row.audio_path, agent_row.audio_path)
    duration = num_samples / sampling_rate
    speaker_metadata = {
        role: (
            row.speaker_id,
            _row_demographics(row),
        )
        for role, row in row_by_role.items()
    }
    for original, role in mapping.items():
        speaker_metadata[original] = speaker_metadata[role]
    turns = normalize_transcript(segments, duration, mapping, speaker_metadata)
    return ValidatedConversation(
        metadata=metadata,
        user_row=user_row,
        agent_row=agent_row,
        turns=tuple(turns),
        sampling_rate=sampling_rate,
        num_samples=num_samples,
    )


def prepare_conversation(metadata: ConversationMetadata) -> PreparedConversation:
    validated = validate_conversation(metadata)
    user_audio, agent_audio, sampling_rate = load_synchronized_audio(
        validated.user_row.audio_path, validated.agent_row.audio_path
    )
    if sampling_rate != validated.sampling_rate or len(user_audio) != validated.num_samples:
        raise ConversationError("audio changed between metadata inspection and decoding")
    duration = validated.duration
    mapping = canonical_speaker_map(
        metadata.sku,
        metadata.rows,
        [turn.original_speaker for turn in validated.turns],
    )

    cut_id = metadata.conference_id
    user_recording = fastcopy(
        Recording.from_file(validated.user_row.audio_path, recording_id=cut_id),
        duration=duration,
        num_samples=validated.num_samples,
    )
    agent_recording = fastcopy(
        Recording.from_file(validated.agent_row.audio_path, recording_id=f"{cut_id}-target"),
        duration=duration,
        num_samples=validated.num_samples,
    )
    conversation_custom = _conversation_metadata(metadata)
    supervisions = [
        SupervisionSegment(
            id=f"{cut_id}-{index:06d}",
            recording_id=cut_id,
            start=turn.start,
            duration=turn.duration,
            channel=0,
            text=turn.text,
            language="en",
            speaker=turn.speaker,
            custom={
                "speaker_id": turn.speaker_id,
                "original_speaker": turn.original_speaker,
                "role": turn.speaker,
                **turn.demographics,
            },
        )
        for index, turn in enumerate(validated.turns)
    ]
    cut = MonoCut(
        id=cut_id,
        start=0.0,
        duration=duration,
        channel=0,
        recording=user_recording,
        supervisions=supervisions,
        custom={
            "target_audio": agent_recording,
            "conference_id": metadata.conference_id,
            "sku": metadata.sku,
            **conversation_custom,
            "metadata": {"conferenceId": metadata.conference_id},
            "user_transcript_speaker": next(key for key, value in mapping.items() if value == "user"),
            "agent_transcript_speaker": next(key for key, value in mapping.items() if value == "agent"),
            "duration": duration,
            "user_audio_duration": duration,
            "agent_audio_duration": duration,
            "audio_path_field": "postprocessed_audio_file_path",
            "max_gap_duration_collapse_turns": MAX_GAP_DURATION_COLLAPSE_TURNS,
        },
    )
    return PreparedConversation(cut, user_audio, agent_audio, sampling_rate)


def _shard_paths(output_dir: Path, shard_index: int) -> tuple[Path, Path, Path]:
    return (
        output_dir / "cuts" / f"cuts.{shard_index:06d}.jsonl.gz",
        output_dir / "recording" / f"recording.{shard_index:06d}.tar",
        output_dir / "target_audio" / f"recording.{shard_index:06d}.tar",
    )


def _shard_result_path(output_dir: Path, shard_index: int) -> Path:
    return output_dir / "audit" / "shards" / f"shard.{shard_index:06d}.json"


def shard_is_complete(output_dir: Path, shard_index: int) -> bool:
    paths = (*_shard_paths(output_dir, shard_index), _shard_result_path(output_dir, shard_index))
    return all(path.is_file() and path.stat().st_size > 0 for path in paths)


def write_shard(
    shard_index: int, conversations: Sequence[ConversationMetadata], output_dir: Path
) -> dict[str, Any]:
    temporary = output_dir / ".tmp" / f"shard-{shard_index:06d}-{os.getpid()}"
    shutil.rmtree(temporary, ignore_errors=True)
    for name in ("cuts", "recording", "target_audio"):
        (temporary / name).mkdir(parents=True, exist_ok=True)
    (temporary / "audit" / "shards").mkdir(parents=True, exist_ok=True)
    accepted, rejected = [], []
    shard_size = len(conversations)
    with (
        JsonlShardWriter(
            str(temporary / "cuts" / "cuts.%06d.jsonl.gz"), shard_size=shard_size, shard_offset=shard_index
        ) as cut_writer,
        AudioTarWriter(
            str(temporary / "recording" / "recording.%06d.tar"),
            shard_size=shard_size,
            format="flac",
            shard_offset=shard_index,
        ) as user_writer,
        AudioTarWriter(
            str(temporary / "target_audio" / "recording.%06d.tar"),
            shard_size=shard_size,
            format="flac",
            shard_offset=shard_index,
        ) as agent_writer,
    ):
        for metadata in conversations:
            try:
                item = prepare_conversation(metadata)
            except Exception as error:
                rejected.append({"conference_id": metadata.conference_id, "reason": str(error)})
                continue
            cut = item.cut
            user_writer.write(
                key=cut.id,
                value=item.user_audio[np.newaxis, :],
                sampling_rate=item.sampling_rate,
                manifest=to_shar_placeholder(cut.recording, cut),
            )
            agent_recording = cut.custom["target_audio"]
            agent_writer.write(
                key=cut.id,
                value=item.agent_audio[np.newaxis, :],
                sampling_rate=item.sampling_rate,
                manifest=to_shar_placeholder(agent_recording, cut),
            )
            cut_writer.write(cut)
            accepted.append(metadata.conference_id)
    result = {"shard": shard_index, "accepted": accepted, "rejected": rejected, "published": bool(accepted)}
    if not accepted:
        shutil.rmtree(temporary, ignore_errors=True)
        return result
    temporary_result = _shard_result_path(temporary, shard_index)
    temporary_result.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for source, destination in zip(_shard_paths(temporary, shard_index), _shard_paths(output_dir, shard_index)):
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.replace(source, destination)
    result_destination = _shard_result_path(output_dir, shard_index)
    result_destination.parent.mkdir(parents=True, exist_ok=True)
    os.replace(temporary_result, result_destination)
    shutil.rmtree(temporary, ignore_errors=True)
    return result


def _write_reports(
    output_dir: Path,
    results: Sequence[Mapping[str, Any]],
    dry_run: bool,
    config_data_root: Path | None = None,
) -> None:
    accepted = sorted(item for result in results for item in result["accepted"])
    rejected = sorted(
        (item for result in results for item in result["rejected"]), key=lambda item: item["conference_id"]
    )
    audit_dir = output_dir / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    with (audit_dir / "accepted_conversations.jsonl").open("w", encoding="utf-8") as stream:
        for conference_id in accepted:
            stream.write(json.dumps({"conference_id": conference_id}) + "\n")
    with (audit_dir / "rejected_conversations.jsonl").open("w", encoding="utf-8") as stream:
        for item in rejected:
            stream.write(json.dumps(item, sort_keys=True) + "\n")
    summary = {
        "accepted": len(accepted),
        "rejected": len(rejected),
        "published_shards": sum(bool(result["published"]) for result in results),
        "dry_run": dry_run,
    }
    (audit_dir / "conversion_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    published = sorted(result["shard"] for result in results if result["published"])
    if published and not dry_run:
        if published != list(range(published[0], published[-1] + 1)):
            raise RuntimeError(f"published shard indices are not contiguous: {published}")
        shard_expression = (
            f"{published[0]:06d}" if len(published) == 1 else f"{{{published[0]:06d}..{published[-1]:06d}}}"
        )
        config_root = config_data_root or output_dir
        config = [
            {
                "type": "lhotse_shar",
                "shar_path": {
                    "cuts": str(config_root / "cuts" / f"cuts.{shard_expression}.jsonl.gz"),
                    "recording": str(config_root / "recording" / f"recording.{shard_expression}.tar"),
                    "target_audio": str(config_root / "target_audio" / f"recording.{shard_expression}.tar"),
                },
                "tags": {"task": "david ai", "tokenizer_names": ["nemotron_nano_30b"]},
            }
        ]
        (output_dir / "davidai_7speakers_train.yaml").write_text(
            yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
        )


def chunked(items: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def main() -> None:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Convert David AI multiturn conversations to sharded Lhotse SHAR.",
    )
    parser.add_argument("--input-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--config-data-root",
        type=Path,
        help="Optional path written into the generated YAML when it differs from the container output path.",
    )
    parser.add_argument("--shard-size", type=int, default=100)
    parser.add_argument("--num-jobs", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    parser.add_argument("--max-conversations", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s %(levelname)s %(message)s")
    if args.shard_size <= 0 or args.num_jobs <= 0:
        parser.error("--shard-size and --num-jobs must be positive")
    if args.max_conversations is not None and args.max_conversations < 0:
        parser.error("--max-conversations must be non-negative")

    conversations = parse_metadata_csv(args.input_root)
    if args.max_conversations is not None:
        conversations = conversations[: args.max_conversations]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    jobs = list(enumerate(chunked(conversations, args.shard_size)))
    results: list[dict[str, Any]] = []
    pending = []
    for shard_index, shard in jobs:
        if args.resume and shard_is_complete(args.output_dir, shard_index):
            results.append(json.loads(_shard_result_path(args.output_dir, shard_index).read_text(encoding="utf-8")))
        else:
            pending.append((shard_index, shard))

    if args.dry_run:
        for shard_index, shard in pending:
            accepted, rejected = [], []
            for metadata in shard:
                try:
                    validate_conversation(metadata)
                    accepted.append(metadata.conference_id)
                except Exception as error:
                    rejected.append({"conference_id": metadata.conference_id, "reason": str(error)})
            results.append({"shard": shard_index, "accepted": accepted, "rejected": rejected, "published": False})
    elif args.num_jobs == 1:
        results.extend(write_shard(index, shard, args.output_dir) for index, shard in pending)
    else:
        with ProcessPoolExecutor(max_workers=args.num_jobs) as executor:
            futures = {
                executor.submit(write_shard, index, shard, args.output_dir): index for index, shard in pending
            }
            for future in as_completed(futures):
                results.append(future.result())
    temporary_root = args.output_dir / ".tmp"
    if temporary_root.is_dir():
        try:
            temporary_root.rmdir()
        except OSError:
            logging.warning("Temporary directory is not empty and was retained: %s", temporary_root)
    (args.output_dir / "audit").mkdir(parents=True, exist_ok=True)
    (args.output_dir / "audit" / "run_arguments.json").write_text(
        json.dumps(
            {
                "input_root": str(args.input_root.resolve()),
                "output_dir": str(args.output_dir.resolve()),
                "config_data_root": str(args.config_data_root) if args.config_data_root else None,
                "shard_size": args.shard_size,
                "num_jobs": args.num_jobs,
                "max_conversations": args.max_conversations,
                "dry_run": args.dry_run,
                "resume": args.resume,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    _write_reports(args.output_dir, results, args.dry_run, config_data_root=args.config_data_root)
    rejected = sum(len(result["rejected"]) for result in results)
    logging.info("Finished: %d conversations rejected; audit reports are in %s", rejected, args.output_dir / "audit")


if __name__ == "__main__":
    main()
