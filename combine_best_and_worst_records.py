import json
import os

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
    print(f"Written {len(records)} records to {fp}")



def filter_best_and_worst_records(best_records, worst_records, worst_records_alternate, cer_threshold=0.01):
    ridx = 0
    filtered_best_records = []
    filtered_worst_records = []
    best_cer_avg = 0.0
    worst_cer_avg = 0.0
    while ridx < len(best_records):
        # print(ridx, len(best_records))
        best_record = best_records[ridx]
        if best_record['cer_gts'] < cer_threshold:
            worst_record = worst_records[ridx]
            worst_record_alternate = worst_records_alternate[ridx]
            if worst_record_alternate['cer_gts'] >= worst_record['cer_gts']:
                # If temp 0.85 is giving a worst record, choose that. 
                worst_record = worst_record_alternate
            if (worst_record['answer_duration'] > 19.0 or best_record['answer_duration'] > 19.0) or (worst_record['answer_duration'] < 1.5 or best_record['answer_duration'] < 1.5):
                print("Skipping record with answer duration > 20.0")
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

best_records_manifest = "/Data/Experiments/DPO_GenerationGoodExamplesTemp0.85/DPOGoodExamples/rlhf_generations/best_records.json"
worst_records_manifest = "/Data/Experiments/DPO_GenerationBadExamplesTemp1.5/DPOBadExamples/rlhf_generations/worst_records.json"
worst_records_manifest2 = "/Data/Experiments/DPO_GenerationGoodExamplesTemp0.85/DPOGoodExamples/rlhf_generations/worst_records.json"

best_records = read_records(best_records_manifest)
worst_records = read_records(worst_records_manifest)
worst_records_85 = read_records(worst_records_manifest2)

print(len(best_records), len(worst_records))
best_records, worst_records = filter_best_and_worst_records(best_records, worst_records, worst_records_85)
print(len(best_records), len(worst_records))
# import ipdb; ipdb.set_trace()



assert len(best_records) == len(worst_records)

ridx = 0
final_records = []
while ridx < len(best_records):
    best_record1 = best_records[ridx]
    best_record2 = best_records[ridx+1]
    worst_record1 = worst_records[ridx]
    worst_record2 = worst_records[ridx+1]
    assert best_record1['reward'] == 1
    assert best_record2['reward'] == 1
    assert worst_record1['reward'] == 0
    assert worst_record2['reward'] == 0
    final_records.append(best_record1)
    final_records.append(best_record2)
    final_records.append(worst_record1)
    final_records.append(worst_record2)
    ridx += 2



final_records_val = final_records[:400]
final_records_train = final_records[400:]

out_dir = "/Data/Experiments/DPO_Data/"
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

write_records(os.path.join(out_dir, "final_records_val.json"), final_records_val)
write_records(os.path.join(out_dir, "final_records_train.json"), final_records_train)