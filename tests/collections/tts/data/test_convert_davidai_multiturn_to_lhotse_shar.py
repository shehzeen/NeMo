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

import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

pytestmark = pytest.mark.unit

SCRIPT = (
    Path(__file__).parents[4] / "scripts" / "magpietts" / "convert_davidai_multiturn_to_lhotse_shar.py"
)
SPEC = importlib.util.spec_from_file_location("convert_davidai_multiturn_to_lhotse_shar", SCRIPT)
converter = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(converter)


def _segments():
    return [
        {"speaker": "speaker-u", "start": 0.0, "end": 0.20, "text": "Hi", "_index": 0},
        {"speaker": "speaker-a", "start": 0.18, "end": 0.35, "text": "Hello", "_index": 1},
        {"speaker": "speaker-u", "start": 0.40, "end": 0.50, "text": "Question", "_index": 2},
    ]


def _write_fixture(root, *, sku="D7", mismatch=0, conference_id="00000000-0000-0000-0000-000000000001"):
    sample_rate = 8000
    user = np.linspace(-0.1, 0.1, sample_rate, dtype=np.float32)
    agent = np.linspace(0.1, -0.1, sample_rate + mismatch, dtype=np.float32)
    sf.write(root / "user.wav", user, sample_rate, subtype="PCM_16")
    sf.write(root / "agent.wav", agent, sample_rate, subtype="PCM_16")
    (root / "transcript.json").write_text(json.dumps({"segments": _segments()}), encoding="utf-8")
    fieldnames = [
        "conversation_id",
        "sku",
        "role",
        "speaker_id",
        "age_range",
        "postprocessed_audio_file_path",
        "machine_generated_transcription_file_path",
    ]
    with (root / "0_metadata.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(
            {
                "conversation_id": conference_id,
                "sku": sku,
                "role": "Asker",
                "speaker_id": "speaker-u",
                "age_range": "30-34",
                "postprocessed_audio_file_path": "user.wav",
                "machine_generated_transcription_file_path": "transcript.json",
            }
        )
        writer.writerow(
            {
                "conversation_id": conference_id,
                "sku": sku,
                "role": "Expert",
                "speaker_id": "speaker-a",
                "age_range": "40-44",
                "postprocessed_audio_file_path": "agent.wav",
                "machine_generated_transcription_file_path": "transcript.json",
            }
        )
    return conference_id


@pytest.mark.parametrize(
    "sku,roles,expected",
    [
        ("D7", ["Asker", "Expert"], {"speaker-0": "user", "speaker-1": "agent"}),
        ("D6b-topic", ["User", "Advisor"], {"speaker-0": "user", "speaker-1": "agent"}),
        ("D6a_topic", ["Host", "Guest"], {"speaker-0": "user", "speaker-1": "agent"}),
    ],
)
def test_canonical_speaker_map_for_typed_skus(sku, roles, expected):
    rows = [
        converter.MetadataRow(
            conference_id="conversation",
            sku=sku,
            speaker_id=f"speaker-{index}",
            role=role,
            audio_path=Path(f"{index}.wav"),
            transcript_path=Path("transcript.json"),
            values={},
        )
        for index, role in enumerate(roles)
    ]
    assert converter.canonical_speaker_map(sku, rows, expected) == expected


def test_d1_uses_first_two_distinct_transcript_speakers():
    rows = [
        converter.MetadataRow("conversation", "D1", "alpha", None, Path("a.wav"), Path("t.json"), {}),
        converter.MetadataRow("conversation", "D1", "beta", None, Path("b.wav"), Path("t.json"), {}),
    ]
    mapping = converter.canonical_speaker_map("D1", rows, ["beta", "beta", "alpha"])
    assert mapping == {"beta": "user", "alpha": "agent"}


def test_normalize_clips_filters_and_stably_orders():
    segments = [
        {"speaker": "speaker-u", "start": 0.4, "end": 0.5, "text": "...", "_index": 0},
        {"speaker": "speaker-a", "start": -0.01, "end": 0.2, "text": "agent", "_index": 1},
        {"speaker": "speaker-u", "start": 0.2, "end": 0.2, "text": "zero", "_index": 2},
        {"speaker": "speaker-u", "start": 0.2, "end": 0.3, "text": "user", "_index": 3},
    ]
    turns = converter.normalize_transcript(
        segments, 0.5, {"speaker-u": "user", "speaker-a": "agent"}, max_gap=0.0
    )
    assert [(turn.speaker, turn.start, turn.end, turn.text) for turn in turns] == [
        ("agent", 0.0, 0.2, "agent"),
        ("user", 0.2, 0.3, "user"),
    ]


def test_merge_requires_global_adjacency_and_preserves_cross_talk():
    turns = converter.normalize_transcript(
        _segments(), 1.0, {"speaker-u": "user", "speaker-a": "agent"}
    )
    assert [(turn.speaker, turn.text) for turn in turns] == [
        ("user", "Hi"),
        ("agent", "Hello"),
        ("user", "Question"),
    ]
    assert turns[0].end > turns[1].start


def test_adjacent_same_speaker_merges_only_below_threshold():
    segments = [
        {"speaker": "speaker-u", "start": 0.0, "end": 0.1, "text": "one"},
        {"speaker": "speaker-u", "start": 1.09, "end": 1.2, "text": "two"},
        {"speaker": "speaker-a", "start": 1.3, "end": 1.4, "text": "answer"},
    ]
    turns = converter.normalize_transcript(
        segments, 2.0, {"speaker-u": "user", "speaker-a": "agent"}
    )
    assert [(turn.speaker, turn.text) for turn in turns] == [
        ("user", "one two"),
        ("agent", "answer"),
    ]
    segments[1]["start"] = 1.1
    turns = converter.normalize_transcript(
        segments, 2.0, {"speaker-u": "user", "speaker-a": "agent"}
    )
    assert [turn.text for turn in turns] == ["one", "two", "answer"]


@pytest.mark.parametrize(
    "start,end",
    [
        (float("nan"), 0.2),
        (0.0, float("inf")),
        (-0.021, 0.2),
        (0.0, 1.021),
    ],
)
def test_normalize_rejects_invalid_or_out_of_bounds_times(start, end):
    segments = [
        {"speaker": "speaker-u", "start": start, "end": end, "text": "hello"},
        {"speaker": "speaker-a", "start": 0.3, "end": 0.4, "text": "answer"},
    ]
    with pytest.raises(converter.ConversationError):
        converter.normalize_transcript(segments, 1.0, {"speaker-u": "user", "speaker-a": "agent"})


def test_parse_metadata_and_prepare_conversation(tmp_path):
    conference_id = _write_fixture(tmp_path)
    metadata = converter.parse_metadata_csv(tmp_path)
    assert [item.conference_id for item in metadata] == [conference_id]
    assert metadata[0].rows[0].values["age_range"] == "30-34"

    prepared = converter.prepare_conversation(metadata[0])
    assert prepared.sampling_rate == 8000
    assert prepared.cut.id == conference_id
    assert prepared.cut.duration == 1.0
    assert prepared.cut.custom["target_audio"].sampling_rate == 8000
    assert prepared.cut.custom["audio_path_field"] == "postprocessed_audio_file_path"
    assert [supervision.speaker for supervision in prepared.cut.supervisions] == [
        "user",
        "agent",
        "user",
    ]
    assert prepared.cut.supervisions[0].custom == {
        "speaker_id": "speaker-u",
        "original_speaker": "speaker-u",
        "role": "user",
        "age_range": "30-34",
    }


def test_audio_allows_one_sample_mismatch_but_rejects_larger(tmp_path):
    one_sample = tmp_path / "one"
    one_sample.mkdir()
    _write_fixture(one_sample, mismatch=1)
    metadata = converter.parse_metadata_csv(one_sample)[0]
    prepared = converter.prepare_conversation(metadata)
    assert len(prepared.user_audio) == len(prepared.agent_audio) == 8000

    two_samples = tmp_path / "two"
    two_samples.mkdir()
    _write_fixture(two_samples, mismatch=2)
    metadata = converter.parse_metadata_csv(two_samples)[0]
    with pytest.raises(converter.ConversationError, match="differ by 2 samples"):
        converter.prepare_conversation(metadata)


def test_explicit_known_bad_conversation_is_rejected(tmp_path):
    _write_fixture(tmp_path, conference_id="abaa0bcf-7e5c-41e5-8677-f2f9b57d2944")
    metadata = converter.parse_metadata_csv(tmp_path)[0]
    with pytest.raises(converter.ConversationError, match="known-bad"):
        converter.prepare_conversation(metadata)


def test_write_shard_and_resume_marker(tmp_path):
    input_root = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_root.mkdir()
    _write_fixture(input_root)
    metadata = converter.parse_metadata_csv(input_root)

    result = converter.write_shard(0, metadata, output_dir)

    assert result["accepted"] == [metadata[0].conference_id]
    assert result["rejected"] == []
    assert converter.shard_is_complete(output_dir, 0)
    assert (output_dir / "cuts" / "cuts.000000.jsonl.gz").is_file()
    assert (output_dir / "recording" / "recording.000000.tar").is_file()
    assert (output_dir / "target_audio" / "recording.000000.tar").is_file()
