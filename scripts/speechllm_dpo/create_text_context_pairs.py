import json
import random
import copy
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--challenging_texts", type=str, default="/Data/challenging_texts_nemollm.txt")
parser.add_argument("--riva_manifest", type=str, default="/Data/CodecDatasets/updatedcodecs/manifests/RivattsEnglish_train_nemo_codec_bw_6.0_phoneme_tts_highsimilarity3.json")
parser.add_argument("--libri_manifest", type=str, default="/Data/CodecDatasets/updatedcodecs/manifests/LibriTTSCorrectContext_train_nemo_codec_bw_6.0_phoneme_tts_highsimilarity2.json")
parser.add_argument("--output_manifest", type=str, default="/Data/CodecDatasets/updatedcodecs/manifests/challenging_nemo21Hz_textcontextpairs.json")
args = parser.parse_args()

challenging_texts = args.challenging_texts
riva_manifest = args.riva_manifest
libri_manifest = args.libri_manifest
output_manifest = args.output_manifest

def read_records(manifest_path):
    with open(manifest_path, 'r') as f:
        lines = f.readlines()
        records = []
        for line in lines:
            records.append(json.loads(line.strip()))
    return records

def write_records(fp, records):
    with open(fp, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")
    print("Wrote {} records to: {}".format(len(records), fp))

def convert_to_tts(records):
    records_copy = copy.deepcopy(records)
    for record in records_copy:
        record['question'] = record['question'].replace("Phoneme TTS", "Text to speech this")
    return records_copy

riva_records = read_records(riva_manifest)
libri_records = read_records(libri_manifest)

# libri_records_longer_than_8 = [ record for record in libri_records if record['answer_duration'] > 8 ]
# riva_records_longer_than_8 = [ record for record in riva_records if record['answer_duration'] > 8 ]
libri_records_longer_than_1 = [ record for record in libri_records if record['answer_duration'] > 2 ]
riva_records_longer_than_1 = [ record for record in riva_records if record['answer_duration'] > 2 ]

with open(challenging_texts, 'r') as f:
    challenging_texts = f.readlines()

challenging_records = []
num_contexts_per_sample = 10
for challenging_text in challenging_texts:
    challenging_text = challenging_text.strip()
    for _ in range(num_contexts_per_sample):
        riva_record = copy.deepcopy(random.choice(riva_records))
        libri_record = copy.deepcopy(random.choice(libri_records))
        riva_record['question'] = "Phoneme TTS " + challenging_text
        libri_record['question'] = "Phoneme TTS " + challenging_text
        riva_record['text'] = challenging_text
        libri_record['text'] = challenging_text
        challenging_records.append(riva_record)
        challenging_records.append(libri_record)

# regular libri records 50% of the challenging records
libri_subset_records = random.sample(libri_records_longer_than_1, int(len(challenging_records)/2.0) )
for libri_subset_record in libri_subset_records:
    context_record = random.choice(libri_records)
    libri_subset_record['context'] = context_record['context']
    libri_subset_record['context_type'] = context_record['context_type']
    libri_subset_record['context_duration'] = context_record['context_duration']

# regular riva records 20% of the challenging records
riva_subset_records = random.sample(riva_records_longer_than_1, int(len(challenging_records)/5.0))
for riva_subset_record in riva_subset_records:
    context_record = random.choice(riva_records)
    riva_subset_record['context'] = context_record['context']
    riva_subset_record['context_type'] = context_record['context_type']
    riva_subset_record['context_duration'] = context_record['context_duration']

phoneme_records = challenging_records + libri_subset_records + riva_subset_records
tts_records = convert_to_tts(phoneme_records)

all_records = phoneme_records + tts_records

random.shuffle(all_records)

write_records(output_manifest, all_records)