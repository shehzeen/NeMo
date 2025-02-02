import random
from pathlib import Path
from nemo.collections.tts.parts.utils.tts_dataset_utils import get_base_dir
import os
import tqdm
import torch
import argparse
from nemo.collections.asr.parts.utils.manifest_utils import read_manifest, write_manifest
import random
import copy

def find_best_candidate(record, candidate_records, audio_base_dir, base_dir_for_file_id, embeddings_save_dir):
    """
    Find the best candidate for a given record from a list of candidate records, in terms of cosine similarity of embeddings.
    Returns the best similarity and the best candidate record.
    """
    record_audio_file_path = os.path.join(audio_base_dir, record['audio_filepath'])
    record_rel_audio_path = Path(record_audio_file_path).relative_to(base_dir_for_file_id).with_suffix("")
    record_rel_audio_path_as_text_id = str(record_rel_audio_path).replace("/", "_")
    record_embedding_fp = os.path.join(embeddings_save_dir, "{}.pt".format(record_rel_audio_path_as_text_id))
    record_embedding = torch.load(record_embedding_fp).cuda()

    cand_embeddings = []
    for candidate_record in candidate_records:
        candidate_audio_file_path = os.path.join(audio_base_dir, candidate_record['audio_filepath'])
        cand_rel_audio_path = Path(candidate_audio_file_path).relative_to(base_dir_for_file_id).with_suffix("")
        cand_rel_audio_path_as_text_id = str(cand_rel_audio_path).replace("/", "_")
        cand_embedding_fp = os.path.join(embeddings_save_dir, "{}.pt".format(cand_rel_audio_path_as_text_id))
        cand_embedding = torch.load(cand_embedding_fp).cuda()
        cand_embeddings.append(cand_embedding)
    cand_embeddings = torch.stack(cand_embeddings)

    with torch.no_grad():
        similarities = torch.nn.functional.cosine_similarity(record_embedding.unsqueeze(0), cand_embeddings, dim=1)
    
    similarity_and_records = []
    for cidx, candidate_record in enumerate(candidate_records):
        similarity_and_records.append((similarities[cidx].item(), candidate_record))
    
    similarity_and_records.sort(key=lambda x: x[0], reverse=True)
    best_similarity, best_candidate_record = similarity_and_records[0]
    return best_similarity, best_candidate_record


if __name__ == "__main__":
    """
    python scripts/t5tts/make_high_similarity_manifest.py \
        --manifest /home/shehzeenh/Code/NewT5TTS/manifests/libri360_val.json \
        --audio_base_dir /Data/LibriTTS \
        --embeddings_save_dir /Data/tempspeakerembeddings \
        --n_candidates_per_record 100 \
        --similarity_threshold 0.6 ;
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=str)
    parser.add_argument("--audio_base_dir", type=str)
    parser.add_argument("--embeddings_save_dir", type=str)
    parser.add_argument("--n_candidates_per_record", type=int, default=100)
    parser.add_argument("--similarity_threshold", type=float, default=0.6)
    args = parser.parse_args()


    records = read_manifest(args.manifest)
    speakerwise_records = {}
    
    audio_filepaths = [] 
    for record in records:
        audio_filepaths.append(os.path.join(args.audio_base_dir, record['audio_filepath']))
        if record['speaker'] not in speakerwise_records:
            speakerwise_records[record['speaker']] = []
        speakerwise_records[record['speaker']].append(record)
    base_dir_for_file_id = get_base_dir(audio_filepaths)

    filtered_records = []
    for ridx, record in enumerate(tqdm.tqdm(records)):
        speaker_records = speakerwise_records[record['speaker']]
        candidate_records = random.sample(speaker_records, args.n_candidates_per_record)
        candidate_records = [r for r in candidate_records if r['audio_filepath'] != record['audio_filepath']]
        
        if len(candidate_records) == 0:
            # Only one record for this speaker
            continue

        best_candidate_similarity, best_candidate_record = find_best_candidate(record, candidate_records, args.audio_base_dir, base_dir_for_file_id, args.embeddings_save_dir)
        if best_candidate_similarity > args.similarity_threshold:
            record_copy = copy.deepcopy(record)
            record_copy['context_similarity'] = best_candidate_similarity
            record_copy['context_audio_filepath'] = best_candidate_record['audio_filepath']
            record_copy['context_duration'] = best_candidate_record['duration']
            filtered_records.append(record_copy)

    out_manifest_path = args.manifest.replace(".json", "_with_high_similarity_contexts.json")
    write_manifest(out_manifest_path, filtered_records)
    print("Length of original manifest: ", len(records))
    print("Length of filtered manifest: ", len(filtered_records))
    print("Written to ", out_manifest_path)