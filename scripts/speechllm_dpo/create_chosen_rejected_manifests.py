import argparse
from nemo.collections.asr.parts.utils import manifest_utils
import random

parser = argparse.ArgumentParser()
parser.add_argument("--metrics_with_manifest", type=str, default="/Data/Experiments/DPO_Generations_RivaRelease/NoYoutube_21Hz_WithWavlm/rlhf_generations/generated_outputs_manifest_with_metrics_emotion.json")
parser.add_argument("--group_size", type=int, default=6)
parser.add_argument("--uncorrupted_group_size", type=int, default=5, help="First N records in each group are from uncorrupted text")
parser.add_argument("--corruption_prob", type=float, default=0.0, help="Percentage of train records in which rejected example can be a corrupted text")
parser.add_argument("--cer_threshold", type=float, default=0.01)
parser.add_argument("--val_size", type=int, default=256)
args = parser.parse_args()


def create_chosen_rejected_records(records, group_size=6, uncorrupted_group_size=5, corruption_prob=0.2):
    assert len(records) % group_size == 0
    num_groups = len(records) // group_size
    best_records = []
    worst_records = []
    for gidx in range(num_groups):
        gsi = gidx * group_size
        gei = (gidx + 1) * group_size
        group = records[gsi:gei]
        group_for_best_candidates = group[:uncorrupted_group_size]
        if random.random() > corruption_prob:
            group = group[:uncorrupted_group_size]
        cer_sim_indices = []
        for sidx, record in enumerate(group):
            cer_sim_indices.append((record['cer_gts'], -record['pred_context_similarity'], sidx))
        
        cer_sim_indices_for_best = []
        for sidx, record in enumerate(group_for_best_candidates):
            cer_sim_indices_for_best.append((record['cer_gts'], -record['pred_context_similarity'], sidx))

        cer_sim_indices = sorted(cer_sim_indices)
        cer_sim_indices_for_best = sorted(cer_sim_indices_for_best)

        best_record = group[cer_sim_indices_for_best[0][2]]
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

records = manifest_utils.read_manifest(args.metrics_with_manifest)
group_size = args.group_size
uncorrupted_group_size = args.uncorrupted_group_size
corruption_prob = args.corruption_prob
cer_threshold = args.cer_threshold
val_size = args.val_size

all_best_records, all_worst_records = create_chosen_rejected_records(records, group_size, uncorrupted_group_size, corruption_prob)
print("Len all_best_records: ", len(all_best_records))
print("Len all_worst_records: ", len(all_worst_records))
best_records, worst_records = filter_best_and_worst_records(all_best_records, all_worst_records, args.cer_threshold)
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

train_manifest = args.metrics_with_manifest.replace(".json", "_corruptionProb_{}__cerTh_{}_train.json".format( corruption_prob, cer_threshold))
val_manifest = args.metrics_with_manifest.replace(".json", "_corruptionProb_{}__cerTh_{}_val.json".format( corruption_prob, cer_threshold))

manifest_utils.write_manifest(train_manifest, final_records_train)
print("Train manifest written to: ", train_manifest)
manifest_utils.write_manifest(val_manifest, final_records_val)
print("Val manifest written to: ", val_manifest)
