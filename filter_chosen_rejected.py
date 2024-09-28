import json
import string

BATCH_SIZE = 8
SAMPLE_RATE = 22050
GROUP_SIZE = 6
# codec_model_path = "/Data/Checkpoints/SpeechCodec_2402.nemo"
codec_model_path = "/Data/Checkpoints/AudioCodec_21Hz-2k-codes_updated.nemo"
generated_manifest = "/Data/Experiments/DPO_Generations/DPO21Hz_90kGenerations/rlhf_generations/generated_outputs_manifest.json"


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
        if (best_record['cer_gts'] == 0 and best_record['wer_gts'] == 0):
            worst_record = worst_records[ridx]
            if (worst_record['cer_gts'] > 0 and worst_record['wer_gts'] > 0):
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


out_manifest = generated_manifest.replace(".json", "_with_metrics.json")

generated_records = read_records(out_manifest)

all_best_records, all_worst_records = create_chosen_rejected_records(generated_records, GROUP_SIZE)
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

final_records_val = final_records[:256]
final_records_train = final_records[256:]

final_records_train_manifest = generated_manifest.replace(".json", "_zerocer_train.json")
final_records_val_manifest = generated_manifest.replace(".json", "_zerower_val.json")

write_records(final_records_train_manifest, final_records_train)
write_records(final_records_val_manifest, final_records_val)