import argparse
import json
import os
import pprint
import string
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch
import librosa
import nemo.collections.asr as nemo_asr
from nemo.collections.asr.metrics.wer import word_error_rate_detail

LOCAL_EVALSETS = {
    'riva_challenging': {
        'manifest': '/datap/misc/Datasets/riva/riva_interspeech.json',
        'audio_dir': '/datap/misc/Datasets/riva'
    },
    'vctk': {
        'manifest': '/home/pneekhara/2023/SimpleT5NeMo/manifests/smallvctk__phoneme__nemo_audio_21fps_8codebooks_2kcodes_v2bWithWavLM_simplet5.json',
        'audio_dir': '/datap/misc/Datasets/VCTK-Corpus'
    }
}
   

def transcribe_with_whisper(whisper_model, whisper_processor, audio_path, language="fr"):

    """Load and preprocess audio file."""
    speech_array, sampling_rate = librosa.load(audio_path, sr=16000)
   
    """Transcribe non-English audio using Whisper model."""

    # Set the language task (optional, improves performance for specific languages)
    forced_decoder_ids = whisper_processor.get_decoder_prompt_ids(language=language) if language else None

    # Preprocess inputs for the model
    inputs = whisper_processor(speech_array, sampling_rate=sampling_rate, return_tensors="pt").input_features

    # Generate transcription
    with torch.no_grad():
        predicted_ids = whisper_model.generate(inputs, forced_decoder_ids=forced_decoder_ids)

    # Decode transcription
    transcription = whisper_processor.batch_decode(predicted_ids, skip_special_tokens=True)

    result = transcription[0]
    
    return result


def find_sample_audios(audio_dir):
    file_list = []
    for f in os.listdir(audio_dir):
        if "predicted_audio" in f and f.endswith(".wav"):
            audio_number = int(f.split("_")[-1].split(".wav")[0])
            file_list.append((audio_number, os.path.join(audio_dir, f)))
    file_list.sort()
    file_list = [t[1] for t in file_list]
    return file_list


def read_manifest(manifest_path):
    records = []
    with open(manifest_path, 'r') as f:
        all_lines = f.readlines()
        for line in all_lines:
            line = line.strip()
            record = json.loads(line)
            if record['duration'] < 20.0 and record['duration'] > 0.5:
                records.append(record)
    return records


def process_text(input_text):
    lower_case_text = input_text.lower()
    no_punctuation_text = lower_case_text.translate(str.maketrans('', '', string.punctuation))
    return " ".join(no_punctuation_text.split())


def evaluate(manifest_path, audio_dir, generated_audio_dir, language="fr"):
    audio_file_lists = find_sample_audios(generated_audio_dir)
    records = read_manifest(manifest_path)
    assert len(audio_file_lists) == len(records)

    device = "cuda"
    # Load Speaker Verification Models
    speaker_verification_model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(model_name='titanet_large')
    speaker_verification_model = speaker_verification_model.to(device)
    speaker_verification_model.eval()

    speaker_verification_model_alternate = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(model_name='titanet_small')
    speaker_verification_model_alternate = speaker_verification_model_alternate.to(device)
    speaker_verification_model_alternate.eval()

    # Load Whisper model and processor
    whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3")
    whisper_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3")

    filewise_metrics = []
    pred_texts = []
    gt_texts = []

    for ridx, record in enumerate(records):
        gt_audio_filepath = record['audio_filepath']
        context_audio_filepath = record.get('context_audio_filepath', None)
        if audio_dir is not None:
            gt_audio_filepath = os.path.join(audio_dir, gt_audio_filepath)
            if context_audio_filepath is not None:
                context_audio_filepath = os.path.join(audio_dir, context_audio_filepath)

        pred_audio_filepath = audio_file_lists[ridx]

        # Use Whisper for transcription
        pred_text = transcribe_with_whisper(whisper_model, whisper_processor, pred_audio_filepath, language=language)
        pred_text = process_text(pred_text)

        if 'normalized_text' in record:
            gt_text = process_text(record['normalized_text'])
        else:
            gt_text = process_text(record['text'])

        detailed_cer = word_error_rate_detail(hypotheses=[pred_text], references=[gt_text], use_cer=True)
        detailed_wer = word_error_rate_detail(hypotheses=[pred_text], references=[gt_text], use_cer=False)

        print(f"{ridx} GT Text: {gt_text}")
        print(f"{ridx} Pr Text: {pred_text}")
        print("CER:", "{:.4f} | WER: {:.4f}".format(detailed_cer[0], detailed_wer[0]))

        # Speaker similarity evaluation
        with torch.no_grad():
            gt_speaker_embedding = speaker_verification_model.get_embedding(gt_audio_filepath).squeeze()
            pred_speaker_embedding = speaker_verification_model.get_embedding(pred_audio_filepath).squeeze()
            pred_gt_ssim = torch.nn.functional.cosine_similarity(gt_speaker_embedding, pred_speaker_embedding, dim=0).item()

            gt_speaker_embedding_alternate = speaker_verification_model_alternate.get_embedding(gt_audio_filepath).squeeze()
            pred_speaker_embedding_alternate = speaker_verification_model_alternate.get_embedding(pred_audio_filepath).squeeze()
            pred_gt_ssim_alternate = torch.nn.functional.cosine_similarity(gt_speaker_embedding_alternate, pred_speaker_embedding_alternate, dim=0).item()

            pred_context_ssim = gt_context_ssim = 0.0
            if context_audio_filepath is not None:
                context_speaker_embedding = speaker_verification_model.get_embedding(context_audio_filepath).squeeze()
                context_speaker_embedding_alternate = speaker_verification_model_alternate.get_embedding(context_audio_filepath).squeeze()

                pred_context_ssim = torch.nn.functional.cosine_similarity(pred_speaker_embedding, context_speaker_embedding, dim=0).item()
                gt_context_ssim = torch.nn.functional.cosine_similarity(gt_speaker_embedding, context_speaker_embedding, dim=0).item()

                pred_context_ssim_alternate = torch.nn.functional.cosine_similarity(pred_speaker_embedding_alternate, context_speaker_embedding_alternate, dim=0).item()
                gt_context_ssim_alternate = torch.nn.functional.cosine_similarity(gt_speaker_embedding_alternate, context_speaker_embedding_alternate, dim=0).item()
    

        filewise_metrics.append({
            'gt_text': gt_text,
            'pred_text': pred_text,
            'detailed_cer': detailed_cer,
            'detailed_wer': detailed_wer,
            'pred_gt_ssim': pred_gt_ssim,
            'pred_context_ssim': pred_context_ssim,
            'gt_context_ssim': gt_context_ssim,
            'pred_gt_ssim_alternate': pred_gt_ssim_alternate,
            'pred_context_ssim_alternate': pred_context_ssim_alternate,
            'gt_context_ssim_alternate': gt_context_ssim_alternate
        })

        pred_texts.append(pred_text)
        gt_texts.append(gt_text)

    avg_metrics = {
        'cer_filewise_avg': sum([ min(m['detailed_cer'][0],1.0) for m in filewise_metrics]) / len(filewise_metrics),
        'wer_filewise_avg': sum([ min(m['detailed_wer'][0],1.0) for m in filewise_metrics]) / len(filewise_metrics),
        'cer_cumulative': word_error_rate_detail(hypotheses=pred_texts, references=gt_texts, use_cer=True)[0],
        'wer_cumulative': word_error_rate_detail(hypotheses=pred_texts, references=gt_texts, use_cer=False)[0],
        'ssim_pred_gt_avg': sum([m['pred_gt_ssim'] for m in filewise_metrics]) / len(filewise_metrics),
        'ssim_pred_context_avg': sum([m['pred_context_ssim'] for m in filewise_metrics]) / len(filewise_metrics),
        'ssim_gt_context_avg': sum([m['gt_context_ssim'] for m in filewise_metrics]) / len(filewise_metrics),
        'ssim_pred_gt_avg_alternate': sum([m['pred_gt_ssim_alternate'] for m in filewise_metrics]) / len(filewise_metrics),
        'ssim_pred_context_avg_alternate': sum([m['pred_context_ssim_alternate'] for m in filewise_metrics]) / len(filewise_metrics),
        'ssim_gt_context_avg_alternate': sum([m['gt_context_ssim_alternate'] for m in filewise_metrics]) / len(filewise_metrics)
    }

    pprint.pprint(avg_metrics)
    return avg_metrics


def main():
    parser = argparse.ArgumentParser(description='Evaluate Generated Audio')
    parser.add_argument('--manifest_path', type=str, default=None)
    parser.add_argument('--audio_dir', type=str, default=None)
    parser.add_argument('--generated_audio_dir', type=str, default=None)
    parser.add_argument('--evalset', type=str, default=None)
    parser.add_argument('--language', type=str, default="fr", help="Language for Whisper transcription")
    args = parser.parse_args()

    if args.evalset is not None:
        assert args.evalset in LOCAL_EVALSETS
        args.manifest_path = LOCAL_EVALSETS[args.evalset]['manifest']
        args.audio_dir = LOCAL_EVALSETS[args.evalset]['audio_dir']

    evaluate(args.manifest_path, args.audio_dir, args.generated_audio_dir, language=args.language)


if __name__ == "__main__":
    main()
