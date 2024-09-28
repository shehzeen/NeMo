import json
import torch
from nemo.collections.tts.models import AudioCodecModel
import nemo.collections.asr as nemo_asr
import soundfile as sf
import string
from nemo.collections.asr.metrics.wer import word_error_rate
import numpy as np
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--codec_model_path", type=str, default="/Data/Checkpoints/AudioCodec_21Hz-2k-codes_updated.nemo")
parser.add_argument("--generated_manifest", type=str, default="/Data/Experiments/DPO_GenerationsDebug/DPO21Hz_90kGenerations/rlhf_generations/generated_outputs_manifest.json")
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--sample_rate", type=int, default=22050)
parser.add_argument("--group_size", type=int, default=6)
parser.add_argument("--val_size", type=int, default=256)
args = parser.parse_args()


batch_size = args.batch_size
sample_rate = args.sample_rate
group_size = args.group_size
val_size = args.val_size
codec_model_path = args.codec_model_path
generated_manifest = args.generated_manifest


def create_chosen_rejected_records(records, group_size=4):
    assert len(records) % group_size == 0
    num_groups = len(records) // group_size
    best_records = []
    worst_records = []
    for gidx in range(num_groups):
        gsi = gidx * group_size
        gei = (gidx + 1) * group_size
        group = records[gsi:gei]
        cer_sim_indices = []
        for sidx, record in enumerate(group):
            cer_sim_indices.append((record['cer_gts'], -record['pred_context_similarity'], sidx))
        cer_sim_indices = sorted(cer_sim_indices)
        best_record = group[cer_sim_indices[0][2]]
        worst_record = group[cer_sim_indices[-1][2]]
        best_record['reward'] = 1
        if worst_record['pred_context_similarity'] > best_record['pred_context_similarity']:
            reward_delta = (worst_record['cer_gts'] - best_record['cer_gts'])
        else:
            reward_delta = (worst_record['cer_gts'] - best_record['cer_gts']) + (best_record['pred_context_similarity'] - worst_record['pred_context_similarity'])
        
        if not (reward_delta > 0):
            # Make sure reward delta is not negative
            print("Warning reward_delta is not positive", reward_delta)
            print(best_record, worst_record)
        
        reward_delta = max(0.001, reward_delta)
        worst_record['reward'] = 1.0 - reward_delta
        best_records.append(best_record)
        worst_records.append(worst_record)
    
    return best_records, worst_records

def filter_best_and_worst_records(best_records, worst_records, cer_threshold=0.02):
    ridx = 0
    filtered_best_records = []
    filtered_worst_records = []
    best_cer_avg = 0.0
    worst_cer_avg = 0.0
    skipped_records = 0
    while ridx < len(best_records):
        # print(ridx, len(best_records))
        best_record = best_records[ridx]
        if best_record['cer_gts'] < cer_threshold:
            worst_record = worst_records[ridx]
            if (worst_record['answer_duration'] > 19.0 or best_record['answer_duration'] > 19.0) or (worst_record['answer_duration'] < 1.5 or best_record['answer_duration'] < 1.5):
                skipped_records += 1
                print("Skipping record with answer duration > 20.0", ridx, skipped_records)
                ridx += 1
                continue
            assert best_record['cer_gts'] <= worst_record['cer_gts']
            if worst_record['cer_gts'] == best_record['cer_gts']:
                assert worst_record['pred_context_similarity'] <= best_record['pred_context_similarity']
            
            filtered_best_records.append(best_record)
            filtered_worst_records.append(worst_record)
            best_cer_avg += best_record['cer_gts']
            worst_cer_avg += worst_record['cer_gts']
        ridx += 1
    
    best_cer_avg /= len(filtered_best_records)
    worst_cer_avg /= len(filtered_worst_records)
    print(f"Best CER avg: {best_cer_avg}, Worst CER avg: {worst_cer_avg}")
    return filtered_best_records, filtered_worst_records

def process_text(input_text):
    """
    Normalizes text for CER/WER calculation.
    Taken from hallucination_eval.py
    """
    # Convert text to lowercase
    lower_case_text = input_text.lower()
    
    # Remove commas from text
    no_comma_text = lower_case_text.replace(",", "")
    
    # Replace "-" with spaces
    no_dash_text = no_comma_text.replace("-", " ")

    no_dash_text = no_dash_text.replace("'", "")
    no_dash_text = no_dash_text.replace(";", "")
    no_dash_text = no_dash_text.replace(".", "")
    
    # Replace double spaces with single space
    single_space_text = " ".join(no_dash_text.split())

    single_space_text = single_space_text.translate(str.maketrans('', '', string.punctuation))

    single_space_text.replace("h t t p", "http")
    single_space_text.replace("w w w", "www")

    return single_space_text

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
        print(f"Wrote {len(records)} records to: {fp}")

def decode_audio(codec_model, codec_paths):
    codec_batch = []
    codec_lens = []
    max_len = 0
    for codec_path in codec_paths:
        codec = torch.load(codec_path)
        codec = codec.to('cuda')
        codec = codec.unsqueeze(0)
        codec_lens.append(codec.shape[2])
        codec_batch.append(codec)
        max_len = max(max_len, codec.shape[2])
    
    for idx in range(len(codec_batch)):
        codec = codec_batch[idx]
        codec_len = codec_lens[idx]
        codec = torch.cat([codec, torch.zeros(1, 8, max_len - codec_len).long().cuda()], dim=2)
        codec_batch[idx] = codec
    
    codec_batch = torch.cat(codec_batch, dim=0).long().cuda()
    codec_lens = torch.Tensor(codec_lens).long().cuda()
    
    with torch.no_grad():
        codec_decoded_audios, codec_decoded_audio_lens = codec_model.decode(tokens=codec_batch, tokens_len=codec_lens)
    
    return codec_decoded_audios, codec_decoded_audio_lens

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


generated_records = read_records(generated_manifest)

codec_model = AudioCodecModel.restore_from(codec_model_path)
codec_model.to(device)
codec_model.eval()

asr_model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained(model_name="nvidia/parakeet-tdt-1.1b")
asr_model = asr_model.to(device)
asr_model.eval()

nemo_sv_model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(model_name='titanet_large')
nemo_sv_model = nemo_sv_model.to(device)
nemo_sv_model.eval()

# ceil division
num_batches = (len(generated_records) + batch_size - 1) // batch_size

st = time.time()
print("num_batches: ", num_batches)
for batch_idx in range(num_batches):
    si = batch_idx * batch_size
    ei = min((batch_idx + 1) * batch_size, len(generated_records))
    answer_paths = [generated_records[idx]['answer'] for idx in range(si, ei)]
    context_paths = [generated_records[idx]['context'] for idx in range(si, ei)]
    answer_audios, answer_audio_lens = decode_audio(codec_model, answer_paths)
    context_audios, context_audio_lens = decode_audio(codec_model, context_paths)
    answer_audio_paths = []
    context_audio_paths = []
    gt_transcripts = []
    for idx in range(si, ei):
        answer_audio, answer_audio_len = answer_audios[idx - si], answer_audio_lens[idx - si]
        context_audio,  context_audio_len = context_audios[idx - si], context_audio_lens[idx - si]
        answer_audio = answer_audio[:answer_audio_len]
        context_audio = context_audio[:context_audio_len]
        
        # Overwrite previous batch audio files to save space
        answer_file_name = answer_paths[idx - si].split("/")[-1]
        answer_audio_path = answer_paths[idx - si].replace(answer_file_name, "{}_decoded_answer_audio.wav".format(idx - si))
        context_audio_path = answer_paths[idx - si].replace(answer_file_name, "{}_decoded_context_audio.wav".format(idx - si))
        sf.write(answer_audio_path, answer_audio.cpu().numpy(), sample_rate)
        sf.write(context_audio_path, context_audio.cpu().numpy(), sample_rate)
        answer_audio_paths.append(answer_audio_path)
        context_audio_paths.append(context_audio_path)
        gt_transcripts.append(process_text(generated_records[idx]['text']))
    
    with torch.no_grad():
        pred_transcripts = asr_model.transcribe(answer_audio_paths)[0]
    
    pred_transcripts = [process_text(transcript) for transcript in pred_transcripts]

    for idx in range(si, ei):
        cer_gt = word_error_rate([pred_transcripts[idx - si]], [gt_transcripts[idx - si]], use_cer=True)
        wer_gt = word_error_rate([pred_transcripts[idx - si]], [gt_transcripts[idx - si]], use_cer=False)
        answer_audio_path = answer_audio_paths[idx - si]
        context_audio_path = context_audio_paths[idx - si]
        with torch.no_grad():
            spk_embedding_pred = nemo_sv_model.get_embedding(answer_audio_path).cpu().detach().numpy().flatten()
            spk_embedding_gt = nemo_sv_model.get_embedding(context_audio_path).cpu().detach().numpy().flatten()

        similarity = np.dot(spk_embedding_pred, spk_embedding_gt) / (
            np.linalg.norm(spk_embedding_pred) * np.linalg.norm(spk_embedding_gt)
        )
        generated_records[idx]['cer_gts'] = float(cer_gt)
        generated_records[idx]['wer_gts'] = float(wer_gt)
        generated_records[idx]['pred_context_similarity'] = float(similarity)
        generated_records[idx]['transcript_pred'] = pred_transcripts[idx - si]
        generated_records[idx]['transcript_gt'] = gt_transcripts[idx - si]
        print("Done idx", idx)

    ct = time.time()
    print(f"Processed batch {batch_idx + 1}/{num_batches} in {ct - st} seconds")

out_manifest = generated_manifest.replace(".json", "_with_metrics.json")
write_records(out_manifest, generated_records)

all_best_records, all_worst_records = create_chosen_rejected_records(generated_records, group_size)
print("Len all_best_records: ", len(all_best_records))
print("Len all_worst_records: ", len(all_worst_records))
best_records, worst_records = filter_best_and_worst_records(all_best_records, all_worst_records)
print("Len filtered best_records: ", len(best_records))
print("Len filtered worst_records: ", len(worst_records))

ridx = 0
final_records = []
while ridx + 1 < len(best_records):
    best_record1 = best_records[ridx]
    best_record2 = best_records[ridx+1]
    worst_record1 = worst_records[ridx]
    worst_record2 = worst_records[ridx+1]
    assert best_record1['reward'] == 1
    assert best_record2['reward'] == 1
    assert worst_record1['reward'] < 1
    assert worst_record2['reward'] < 1
    final_records.append(best_record1)
    final_records.append(best_record2)
    final_records.append(worst_record1)
    final_records.append(worst_record2)
    ridx += 2

final_records_val = final_records[:val_size]
final_records_train = final_records[val_size:]

final_records_train_manifest = generated_manifest.replace(".json", "_final_records_train.json")
final_records_val_manifest = generated_manifest.replace(".json", "_final_records_val.json")

write_records(final_records_train_manifest, final_records_train)
write_records(final_records_val_manifest, final_records_val)