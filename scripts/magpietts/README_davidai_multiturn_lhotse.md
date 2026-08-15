# DavidAI Multiturn to EMTTS Lhotse

This document describes how to reproduce the conversion of the synchronized
DavidAI delivery into dual-channel Lhotse SHAR for EMTTS multi-turn training.

## Paths

Source data:

```text
/lustre/fsw/portfolios/llmservice/projects/llmservice_nemo_speechlm/data/duplex/DavidAI_7speakers_en-2026-08-10
```

Converter:

```text
scripts/magpietts/convert_davidai_multiturn_to_lhotse_shar.py
```

Staging output:

```text
/lustre/fsw/portfolios/nemotron/projects/nemotron_speech_tts/data/lhotse_datasets/en_multiturn/DavidAI_7speakers_en-2026-08-10_lhotse_s2s_format
```

The staging output can be copied to AIStore after validation.

## Input contract

The source root contains `0_metadata.csv`. It has two rows per conversation,
one for each synchronized speaker track. Each row references:

- A postprocessed full-conversation mono WAV.
- A shared `machine_transcription.json`.
- A speaker UUID, contextual role, SKU, and demographic metadata.

The transcript contains segment-level `text`, `start`, `end`, and `speaker`
fields. Times are relative to the beginning of the synchronized tracks.
Opposite-speaker overlaps and backchannels are expected.

The converter uses `postprocessed_audio_file_path`; it does not use the
preprocessed tracks.

## Speaker mapping

The primary `recording` SHAR field is the user/source channel.
`target_audio` is the agent/target channel.

Speaker UUIDs are mapped as follows:

- D7: `Asker` is user; `Expert` is agent.
- D6b: `User` is user; `Advisor` is agent.
- D6a: `Host` is user; `Guest` is agent.
- D1: the first distinct transcript speaker is user; the second is agent.

The role-bearing SKUs use the CSV role even when the agent speaks first.

## Transcript normalization

The converter:

1. Validates finite timestamps and synchronized mono audio headers.
2. Allows at most 20 ms of timestamp rounding outside the audio boundary and
   clips that small discrepancy.
3. Drops zero-duration and punctuation-only transcript records.
4. Stably sorts records by start, end, and original record index.
5. Merges only globally adjacent records from the same speaker when the gap is
   strictly less than one second.
6. Preserves all cross-speaker overlap and never forces strict turn
   alternation.

Conversation `abaa0bcf-7e5c-41e5-8677-f2f9b57d2944` is explicitly rejected
because its transcript contains a 759.1-second alignment for “Exactly.” and
does not provide a recoverable remainder for that speaker.

## Output contract

The output is sorted by conversation UUID and sharded at 100 cuts:

```text
cuts/cuts.000000.jsonl.gz
recording/recording.000000.tar
target_audio/recording.000000.tar
audit/
davidai_7speakers_train.yaml
```

Each manifest record is a Lhotse `MonoCut` with:

- A full user/source `recording`.
- A same-length `target_audio` recording.
- Timestamped `user` and `agent` supervisions.
- `speaker_id`, original speaker, canonical role, and available demographics.
- Conversation metadata and the conversion settings.

Audio is stored as FLAC inside uncompressed tar shards. Original sample rates
(24, 32, and 44.1 kHz) are preserved. EMTTS resamples at load time.

No `ipa` field is generated. IPA enrichment is a separate downstream step.

## Interactive container

Only one interactive allocation should be active at a time. Before launching,
check for an existing session:

```bash
squeue -u "$USER"
```

The conversion was run with the image and resource pattern from:

```text
/lustre/fs12/portfolios/nemotron/users/pneekhara/launch_scripts/debug.sh
```

That script hard-codes another NeMo checkout and does not mount this input or
output. Either copy and adapt it, or launch with these essential mounts:

```bash
CODE=/lustre/fs12/portfolios/nemotron/users/shehzeenh/mountdir/NeMo
INPUT=/lustre/fsw/portfolios/llmservice/projects/llmservice_nemo_speechlm/data/duplex/DavidAI_7speakers_en-2026-08-10
PROJECT_DATA=/lustre/fsw/portfolios/nemotron/projects/nemotron_speech_tts/data
CONTAINER=/lustre/fsw/portfolios/nemotron/projects/nemotron_speech_tts/containers/nemo_26.06_rc3.sqsh

srun \
  --account=nemotron_speech_tts \
  --partition=interactive \
  --job-name=davidai-lhotse \
  --nodes=1 \
  --ntasks-per-node=1 \
  --gpus-per-node=1 \
  --cpus-per-task=30 \
  --time=4:00:00 \
  --mem=220G \
  --overcommit \
  --export=ALL \
  --no-container-mount-home \
  --container-image="$CONTAINER" \
  --container-mounts="$CODE:/code,$INPUT:/data/input,$PROJECT_DATA:/data/output" \
  --pty bash -lc 'cd /code && export PYTHONPATH="/code:${PYTHONPATH:-}" && exec bash -i'
```

The commands below assume they are run inside that container.

## Dry run

The dry run reads CSV, JSON, and WAV headers but does not decode or write audio:

```bash
ROOT=/data/output/lhotse_datasets/en_multiturn/DavidAI_7speakers_en-2026-08-10_lhotse_s2s_format

python scripts/magpietts/convert_davidai_multiturn_to_lhotse_shar.py \
  --input-root /data/input \
  --output-dir "$ROOT" \
  --shard-size 100 \
  --num-jobs 15 \
  --dry-run
```

Review:

```text
audit/conversion_summary.json
audit/rejected_conversations.jsonl
```

The expected dry-run result for this delivery is 4,794 accepted and one
rejected conversation.

## Optional pilot

Use a separate temporary directory so the pilot cannot be mistaken for the
full dataset:

```bash
python scripts/magpietts/convert_davidai_multiturn_to_lhotse_shar.py \
  --input-root /data/input \
  --output-dir "${ROOT}_pilot" \
  --shard-size 100 \
  --num-jobs 1 \
  --max-conversations 100
```

Delete the pilot after it has been validated.

## Full conversion

`--config-data-root` writes host-visible paths into the generated YAML instead
of the container mount path:

```bash
HOST_ROOT=/lustre/fsw/portfolios/nemotron/projects/nemotron_speech_tts/data/lhotse_datasets/en_multiturn/DavidAI_7speakers_en-2026-08-10_lhotse_s2s_format

python scripts/magpietts/convert_davidai_multiturn_to_lhotse_shar.py \
  --input-root /data/input \
  --output-dir "$ROOT" \
  --config-data-root "$HOST_ROOT" \
  --shard-size 100 \
  --num-jobs 15 \
  --resume
```

The converter publishes each shard atomically. `--resume` skips shards that
have all three data files plus their per-shard audit record. After an
interrupted allocation, remove only incomplete directories under `.tmp/`
before resuming.

Expected result:

- 4,794 cuts in 48 shards (`000000` through `000047`).
- Approximately 76 GB.
- Approximately 789.13 hours.
- 434,786 supervisions.
- One rejected conversation.

## Tests and validation

Run the converter tests:

```bash
pytest -q tests/collections/tts/data/test_convert_davidai_multiturn_to_lhotse_shar.py
pytest -q tests/collections/tts/data/test_magpietts_dataset_lhotse.py -k multiturn
```

The completed staging dataset includes:

- `audit/validation_summary.json`: all 48 shards and every source/target audio
  member were round-tripped and decoded.
- `audit/emtts_smoke_test.json`: an EMTTS batch was loaded with IPA disabled.
- `audit/overlap_inspection.json`: channel-energy and overlap inspection.
- `audit/overlap_audio_example.wav`: stereo listening sample; left is the
  user/source track and right is the agent/target track.

Before publishing, confirm:

- There are 48 files in each of `cuts`, `recording`, and `target_audio`.
- There are 4,794 unique cut IDs.
- Source and target sample counts match for every cut.
- Every supervision has a valid role, text, bounds, and `speaker_id`.
- IPA is absent.
- The EMTTS smoke test produces source/target audio, user/agent masks, text
  channels, and extracted user turns.

## Training integration note

The generated `davidai_7speakers_train.yaml` includes:

```yaml
tags:
  task: david ai
  tokenizer_names:
  - nemotron_nano_30b
```

The existing EMTTS launcher applies `excluded_speaker_ids` to its DavidAI
mixture. If all 4,794 converted cuts should be sampled, load this dataset
without that filter. Otherwise 221 cuts remain on disk but are excluded at
runtime.
