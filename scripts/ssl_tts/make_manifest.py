import glob
import json
import os

import librosa

root_dir = "/data/shehzeen/AudioDatasets/LibriTTS/LibriTTS/dev-clean"

# recursively find .wav files
records = []
spk_idx = 0
for speaker in os.listdir(root_dir):
    speaker_dir = os.path.join(root_dir, speaker)
    for chapter in os.listdir(speaker_dir):
        chapter_dir = os.path.join(speaker_dir, chapter)
        for file in os.listdir(chapter_dir):
            if file.endswith(".wav"):
                file_path = os.path.join(chapter_dir, file)
                duration = librosa.get_duration(filename=file_path)
                records.append(
                    {"audio_filepath": file_path, "duration": duration, "speaker": spk_idx, "speaker_name": speaker,}
                )
    spk_idx += 1

manifest_path = "/data/shehzeen/SSLTTS/manifests/libri_test.json"
with open(manifest_path, "w") as f:
    for record in records:
        print(record)
        f.write(json.dumps(record) + "\n")
