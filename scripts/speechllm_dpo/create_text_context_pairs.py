import json
import random
import copy
import argparse


def corrupt_text(question_text):
    # randomly repeat word or delete a word from the question
    question_words = question_text.split(" ")
    if random.random() < 0.5:
        # repeat a word
        word_idx = random.randint(0, len(question_words) - 1)
        word = question_words[word_idx]
        # Repeat one occurence of the word
        question_text = question_text.replace(word, word + " " + word, 1)
    else:
        # delete a word
        word_idx = random.randint(0, len(question_words) - 1)
        word = question_words[word_idx]
        question_text = question_text.replace(word, "", 1)
    
    return question_text


parser = argparse.ArgumentParser()
parser.add_argument("--challenging_texts", type=str, default="/Data/challenging_texts_nemollm.txt")
parser.add_argument("--riva_manifest", type=str, default="/datap/misc/speechllm_codecdatasets_new/manifests/rivaLindyRodney__phoneme__nemo_audio_21fps_8codebooks_2kcodes_v2bWithWavLM.json")
parser.add_argument("--libri_manifest", type=str, default="/datap/misc/speechllm_codecdatasets_new/manifests/libri360__phoneme__nemo_audio_21fps_8codebooks_2kcodes_v2bWithWavLM.json")
parser.add_argument("--riva_textcontext_manifest", type=str, default="/datap/misc/speechllm_codecdatasets_new/manifests/rivaLindyRodneyTextContext__phoneme__nemo_audio_21fps_8codebooks_2kcodes_v2bWithWavLM.json")
parser.add_argument("--tts_records", type=bool, default=False)
parser.add_argument("--output_manifest", type=str, default="/datap/misc/speechllm_codecdatasets_new/manifests/dpo_textcontext_pairs_21Hz_NoElizWithWavLM.json")
args = parser.parse_args()

challenging_texts = args.challenging_texts
riva_manifest = args.riva_manifest
libri_manifest = args.libri_manifest
riva_textcontext_manifest = args.riva_textcontext_manifest
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
riva_textcontext_records = read_records(riva_textcontext_manifest)

# libri_records_longer_than_8 = [ record for record in libri_records if record['answer_duration'] > 8 ]
# riva_records_longer_than_8 = [ record for record in riva_records if record['answer_duration'] > 8 ]
libri_records_longer_than_2 = [ record for record in libri_records if record['answer_duration'] > 2 ]
riva_records_longer_than_2 = [ record for record in riva_records if record['answer_duration'] > 2 ]

with open(challenging_texts, 'r') as f:
    challenging_texts = f.readlines()

challenging_records = []
num_contexts_per_sample = 12
for challenging_text in challenging_texts:
    challenging_text = challenging_text.strip()
    for ci in range(num_contexts_per_sample):
        if ci >= num_contexts_per_sample - 2:
            # For last 20% of the challenging texts, make it more challenging by corrupting the text
            # Randomly drops a word or repeats a word
            print("Corrupting text: {}".format(challenging_text))
            challenging_text = corrupt_text(challenging_text)
            print("Corrupted text: {}".format(challenging_text))
        riva_record = copy.deepcopy(random.choice(riva_records))
        libri_record = copy.deepcopy(random.choice(libri_records))
        riva_textcontext_record = copy.deepcopy(random.choice(riva_textcontext_records))
        riva_record['question'] = "Phoneme TTS " + challenging_text
        libri_record['question'] = "Phoneme TTS " + challenging_text
        riva_textcontext_record['question'] = "Phoneme TTS " + challenging_text
        riva_record['text'] = challenging_text
        libri_record['text'] = challenging_text
        riva_textcontext_record['text'] = challenging_text
        challenging_records.append(riva_record)
        challenging_records.append(libri_record)
        if ci == 0:
            # dont need too many text context examples
            challenging_records.append(riva_textcontext_record)

# regular libri records 50% of the challenging records
libri_subset_records = random.sample(libri_records_longer_than_2, int(len(challenging_records)/2.0) )
for libri_subset_record in libri_subset_records:
    context_record = random.choice(libri_records)
    libri_subset_record['context'] = context_record['context']
    libri_subset_record['context_type'] = context_record['context_type']
    libri_subset_record['context_duration'] = context_record['context_duration']

# regular riva records 20% of the challenging records
riva_subset_records = random.sample(riva_records_longer_than_2, int(len(challenging_records)/5.0))
for riva_subset_record in riva_subset_records:
    context_record = random.choice(riva_records)
    riva_subset_record['context'] = context_record['context']
    riva_subset_record['context_type'] = context_record['context_type']
    riva_subset_record['context_duration'] = context_record['context_duration']

# riva textcontext records 5% of the challenging records
riva_textcontext_subset_records = random.sample(riva_textcontext_records, int(len(challenging_records)/20.0))
for riva_textcontext_subset_record in riva_textcontext_subset_records:
    context_record = random.choice(riva_textcontext_records)
    riva_textcontext_subset_record['context'] = context_record['context']
    riva_textcontext_subset_record['context_type'] = context_record['context_type']
    riva_textcontext_subset_record['context_duration'] = context_record['context_duration']

phoneme_records = challenging_records + libri_subset_records + riva_subset_records + riva_textcontext_subset_records

if args.tts_records:
    tts_records = convert_to_tts(phoneme_records)
else:
    tts_records = []

all_records = phoneme_records + tts_records

random.shuffle(all_records)

write_records(output_manifest, all_records)