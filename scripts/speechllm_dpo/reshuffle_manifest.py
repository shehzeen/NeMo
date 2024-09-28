import json
import os

def read_manifest(manifest_path):
    records = []
    with open(manifest_path, 'r') as f:
        all_lines = f.readlines()
        for line in all_lines:
            line = line.strip()
            records.append(json.loads(line))
    return records

def write_manifest(manifest_path, records):
    with open(manifest_path, 'w') as f:
        file_str = ""
        for record in records:
            file_str += json.dumps(record) + "\n"
        file_str = file_str.strip()
        f.write(file_str)
        print("Wrote {} records to: {}".format(len(records), manifest_path))

# manifest = "/Data/Experiments/DPO_Data_NewLongerthan3s/final_records_train.json"
manifest = "/Data/Experiments/DPO_Generations/DPO21Hz_90kGenerations/rlhf_generations/generated_outputs_manifest_zerower_val.json"
records = read_manifest(manifest)

micro_batch_size = 4
assert len(records) % micro_batch_size == 0
n_micro_batches = len(records) // micro_batch_size
new_records = [None for _ in range(len(records))]

filled_targets = {}
filling_complete = False
target_start_idx = 0
for bidx in range(n_micro_batches):
    si = bidx * micro_batch_size
    ei = si + micro_batch_size
    if bidx % 16 == 0:
        target_start_idx = bidx * 4
    else:
        if bidx > 0:
            target_start_idx = target_start_idx + 1
    # if bidx not in filled_targets:
    #     target_idx = bidx
    # else:
    #     # find the next available target index
    #     max_filled_target = max(filled_targets.keys())
    #     target_idx = max_filled_target + 1
    #     print("max_filled_target", max_filled_target, "target_idx", target_idx)
    target_idx = target_start_idx
    if target_idx + 48 >= len(records):
        filling_complete = True
        print("Filling complete")
        break 
    for ridx in range(si, ei):
        if target_idx >= len(records):
            filling_complete = True
            break
        assert target_idx not in filled_targets
        new_records[target_idx] = records[ridx]
        filled_targets[target_idx] = True
        print("bidx", bidx, "ridx", ridx, "target_idx", target_idx)
        target_idx += 16
        assert target_idx not in filled_targets
    
    if filling_complete:
        print("Filling complete")
        break

new_records = [r for r in new_records if r is not None]
out_manifest = manifest.replace(".json", "_2nodes.json")
write_manifest(out_manifest, new_records)

        