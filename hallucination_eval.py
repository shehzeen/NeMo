import os
import json
import argparse
import nemo.collections.asr as nemo_asr
from nemo.collections.asr.metrics.wer import word_error_rate_detail
import string
import pprint

def find_sample_audios(exp_name, exp_base_dir):
    exp_dir = os.path.join(exp_base_dir, exp_name)
    sub_dirs = [x[0] for x in os.walk(exp_dir)]
    
    
    for sub_dir in sub_dirs:
        if "Sample_Audios" in sub_dir:
            audio_file_lists = {
                'gt' : [],
                'pred' : []
            }
            for f in os.listdir(sub_dir):
                if f.endswith(".wav") and "16khz" not in f:
                    audio_number = int(f.split("_")[-1].split(".wav")[0])
                    if "dec_input" in f:
                        audio_file_lists['gt'].append((audio_number, os.path.join(sub_dir, f)))
                    elif 'predicted' in f:
                        audio_file_lists['pred'].append((audio_number, os.path.join(sub_dir, f)))

    audio_file_lists['gt'].sort()
    audio_file_lists['pred'].sort()
    audio_file_lists['gt'] = [t[1] for t in audio_file_lists['gt']]
    audio_file_lists['pred'] = [t[1] for t in audio_file_lists['pred']]
    return audio_file_lists


def read_manifest(manifest_path):
    records = []
    with open(manifest_path, 'r') as f:
        all_lines = f.readlines()
        for line in all_lines:
            line = line.strip()
            records.append(json.loads(line))
    return records

def process_text(input_text):
    # Convert text to lowercase
    lower_case_text = input_text.lower()
    
    # Remove commas from text
    no_comma_text = lower_case_text.replace(",", "")
    
    # Replace "-" with spaces
    no_dash_text = no_comma_text.replace("-", " ")
    
    # Replace double spaces with single space
    single_space_text = " ".join(no_dash_text.split())

    single_space_text = single_space_text.translate(str.maketrans('', '', string.punctuation))
    
    return single_space_text

def contains_invalid_text(text):
    invalid_substrings = [
        "one b two zero four nine two eight zero zero zero",
    ]
    for invalid_substring in invalid_substrings:
        if invalid_substring in text:
            return True
    return False

def main():
    parser = argparse.ArgumentParser(description='Evaluate on challenging texts')
    parser.add_argument('--exp_name', type=str, default="dpp1e-6_step1800_CHALLENGING")
    parser.add_argument('--exp_base_dir', type=str, default="/Data/Experiments/DPO_EOS_InferenceTestsContinuousEval/")
    parser.add_argument('--manifest_path', type=str, default="/Data/CodecDatasets/updatedcodecs/manifests/challenging_interspeech.json")
    args = parser.parse_args()

    audio_file_lists = find_sample_audios(args.exp_name, args.exp_base_dir)
    pred_audio_files = audio_file_lists['pred']

    manifest_records = read_manifest(args.manifest_path)

    print("Len pred_audio_files:", len(pred_audio_files))
    print("Len manifest_records:", len(manifest_records))
    
    max_answer_duration = 0
    for record in manifest_records:
        if record['answer_duration'] > max_answer_duration:
            max_answer_duration = record['answer_duration']
    print("Max answer duration:", max_answer_duration)
    # import ipdb; ipdb.set_trace()


    # assert len(pred_audio_files) == len(manifest_records)

    device = "cuda"
    asr_model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained(
                    model_name="nvidia/parakeet-tdt-1.1b"
                )
    asr_model = asr_model.to(device)
    asr_model.eval()

    pred_texts = []
    gt_texts = []
    wer_ranked_list = []
    all_cers = []
    for ridx, record in enumerate(manifest_records[:len(pred_audio_files)]):
        gt_text = process_text(record['text'])
        if contains_invalid_text(gt_text):
            continue
        pred_text = asr_model.transcribe([pred_audio_files[ridx]])[0][0]
        pred_text = process_text(pred_text)
        pred_texts.append(pred_text)
        gt_texts.append(gt_text)

        print ("Ridx", ridx)
        print ("GT:", gt_text)
        print ("Pred:", pred_text)

        detailed_cer = word_error_rate_detail(hypotheses=[pred_text], references=[gt_text], use_cer=True)
        detailed_wer = word_error_rate_detail(hypotheses=[pred_text], references=[gt_text], use_cer=False)
        print("CER:", detailed_cer[0])
        wer_ranked_list.append(
            (detailed_cer[0], detailed_wer[0], gt_text, pred_text, pred_audio_files[ridx])
        )
    
    # Reverse sort by CER
    # Print challenging texts with highest CER
    print("*"*50)
    print("Top 10 Challenge Texts")
    print("*"*50)
    wer_ranked_list.sort(key=lambda x: x[0], reverse=True)
    for item in wer_ranked_list[:10]:
        print ("CER:", item[0])
        print ("WER:", item[1])
        print ("GT:", item[2])
        print ("Pred:", item[3])
        print ("Audio:", item[4])
        print ("-"*50)
    
    cumulative_cer_metrics = word_error_rate_detail(hypotheses=pred_texts, references=gt_texts, use_cer=True)
    cumulative_wer_metrics = word_error_rate_detail(hypotheses=pred_texts, references=gt_texts, use_cer=False)
    cer, words, ins_rate, del_rate, sub_rate = cumulative_cer_metrics
    wer, words_wer, ins_rate_wer, del_rate_wer, sub_rate_wer = cumulative_wer_metrics
    print("*"*50)
    print("Cumulative CER Metrics")
    print("*"*50)
    print ("CER:", cer)
    print ("WER:", wer)
    print ("Words:", words)
    print ("Ins:", ins_rate)
    print ("Del:", del_rate)
    print ("Sub:", sub_rate)
    

    out_dir = os.path.join(args.exp_base_dir, args.exp_name)
    all_metrics = {
        'average' : {
            'cer' : cer,
            'wer' : wer,
            'words' : words,
            'ins' : ins_rate,
            'del' : del_rate,
            'sub' : sub_rate
        },
        'detailed' : wer_ranked_list
    }


    pprint.pprint(all_metrics['average'])
    
    with open(os.path.join(out_dir, "hallucination_metrics.json"), 'w') as f:
        json.dump(all_metrics, f, indent=4)
        print("Wrote metrics to:", os.path.join(out_dir, "hallucination_metrics.json"))

if __name__ == "__main__":
    main()