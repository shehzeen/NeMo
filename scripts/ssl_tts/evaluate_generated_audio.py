import argparse
import fnmatch
import json
import os
import pickle
import random

import editdistance
import librosa
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import soundfile
import torch
import torch.nn.functional as F
import torchaudio
from numpy import dot
from numpy.linalg import norm
from scipy import linalg
from sklearn.manifold import TSNE
from sklearn.metrics import auc, roc_curve
from tqdm import tqdm

import nemo.collections.asr as nemo_asr
from nemo.collections.asr.models import label_models
from nemo.collections.asr.parts.preprocessing.features import WaveformFeaturizer
from nemo.collections.tts.models import fastpitch_ssl, hifigan, ssl_tts
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import jiwer


nemo_transcriptions = {} # key: audio_path, value: transcription (Cache)
def transcribe_nemo(asr_model, audio_path):
    global nemo_transcriptions
    if audio_path in nemo_transcriptions:
        return nemo_transcriptions[audio_path]
    transcription = asr_model.transcribe([audio_path])[0]
    nemo_transcriptions[audio_path] = transcription

    return transcription




wav2vec2_transcriptions = {} # key: audio_path, value: transcription (Cache)
def transcribe_wav2vec2(wav2vec2_processor, wav2vec2_model, audio_path):
    global wav2vec2_transcriptions

    if audio_path in wav2vec2_transcriptions:
        return wav2vec2_transcriptions[audio_path]
    
    audio_input, _ = librosa.load(audio_path, sr=16000)
    audio_torch = wav2vec2_processor(audio_input, return_tensors="pt", sampling_rate=16000).input_values

    with torch.no_grad():
        logits = wav2vec2_model(audio_torch).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = wav2vec2_processor.batch_decode(predicted_ids)[0]
        wav2vec2_transcriptions[audio_path] = transcription

    return transcription
    

def segment_wav(wav, segment_length=32000, hop_size=16000, min_segment_size=16000):
    if len(wav) < segment_length:
        pad = torch.zeros(segment_length - len(wav))
        segment = torch.cat([wav, pad])
        return [segment]
    else:
        si = 0
        segments = []
        while si < len(wav) - min_segment_size:
            segment = wav[si : si + segment_length]
            if len(segment) < segment_length:
                pad = torch.zeros(segment_length - len(segment))
                segment = torch.cat([segment, pad])

            segments.append(segment)
            si += hop_size
        return segments

def get_speaker_embedding(nemo_sv_model, wav_featurizer, audio_paths, duration=None, device="cpu"):
    all_segments = []
    all_wavs = []
    for audio_path in audio_paths:
        wav = wav_featurizer.process(audio_path)
        segments = segment_wav(wav)
        all_segments += segments
        all_wavs.append(wav)
        if duration is not None and len(all_segments) >= duration:
            # each segment is 2 seconds with one second overlap.
            # so 10 segments would mean 0 to 2, 1 to 3.. 9 to 11 (11 seconds.)
            all_segments = all_segments[: int(duration)]
            break

    with torch.no_grad():
        signal_batch = torch.stack(all_segments)
        signal_length_batch = torch.stack([torch.tensor(signal_batch.shape[1]) for _ in range(len(all_segments))])
        signal_batch = signal_batch.to(device)
        signal_length_batch = signal_length_batch.to(device)

        _, speaker_embeddings = nemo_sv_model(input_signal=signal_batch, input_signal_length=signal_length_batch)

    speaker_embedding_list = [speaker_embeddings[i].cpu().detach().numpy().flatten() for i in range(speaker_embeddings.shape[0])]

    return speaker_embedding_list

def calculatePitchMetrics(ref_pitch_contour_np, gen_pitch_contour_np):
    # F0 Frame Error - FFE #
    NFOE_error = 0.0 
    NUV_ref = 0.0
    NVU_gen = 0.0
    NVV = 0.0
    N_total = 0.0

    for i in range(min(len(ref_pitch_contour_np), len(gen_pitch_contour_np))):
        N_total += 1.
        if (ref_pitch_contour_np[i] != 0 and gen_pitch_contour_np[i] != 0):        
            non_zero_count +=1
        if (ref_pitch_contour_np[i] == 0 and gen_pitch_contour_np[i] != 0):
            NUV_ref += 1
        if (gen_pitch_contour_np[i] == 0 and ref_pitch_contour_np[i] != 0):
            NVU_gen += 1

        if (ref_pitch_contour_np[i] > 0 and  gen_pitch_contour_np[i] > 0):
            NVV += 1 
            ratio = (gen_pitch_contour_np[i]/ref_pitch_contour_np[i])
            if (ratio > 1.2 or ratio < 0.8):
                NFOE_error +=1

    FFE = ((NFOE_error + NUV_ref + NVU_gen)/N_total)
    
    if NVV == 0:
        GPE = 1.0
    else:
        GPE =  (NFOE_error/NVV)

    
    VDE = ((NUV_ref + NVU_gen)/N_total)
    metric_dict = {'FFE': FFE, "GPE" : GPE, "VDE" : VDE}
    
    return metric_dict

def get_similarity(emb1, emb2):
    similarity = dot(emb1, emb2) / (norm(emb1) * norm(emb2))
    return similarity

def calculate_cer(source_transcripts, generated_transcripts, source_paths, generated_paths):
    assert len(source_transcripts) == len(generated_transcripts)
    cer_mean = 0.0
    pairwise_cers = []
    for i in range(len(source_transcripts)):
        t1 = source_transcripts[i]
        t2 = generated_transcripts[i]
        cer = jiwer.cer(t1, t2)
        pairwise_cers.append((source_paths[i], generated_paths[i], cer))
        cer_mean += cer
    cer_mean /= len(source_transcripts)
    return cer_mean, pairwise_cers

def calculate_eer(speaker_embeddings, mode="generated"):
    generated_embeddings = {}
    real_embeddings = {}
    for key in speaker_embeddings:
        speaker = key.split("_")[1]
        if speaker not in generated_embeddings:
            generated_embeddings[speaker] = []
        if speaker not in real_embeddings:
            real_embeddings[speaker] = []
        if "generated" in key:
            generated_embeddings[speaker] += speaker_embeddings[key]
        else:
            real_embeddings[speaker] += speaker_embeddings[key]

    y_score = []
    y_true = []
    if mode == "generated":
        anchor_embeddings = generated_embeddings
    else:
        anchor_embeddings = real_embeddings

    speaker_similarities = {}
    kidx = 0
    for key in anchor_embeddings:
        speaker_sim_score = []
        alternate_keys = [k for k in real_embeddings if k != key]
        alt_indices = []
        for altkey in alternate_keys:
            for aid in range(len(real_embeddings[altkey])):
                alt_indices.append((altkey, aid))
        
        random.seed(kidx)
        random.shuffle(alt_indices)

        for aidx, anchor_embedding in enumerate(anchor_embeddings[key]):
            for ridx, real_same_embedding in enumerate(real_embeddings[key]):
                if mode == "real" and ridx == aidx:
                    # skip if same speaker and same utterance
                    continue

                same_score = get_similarity(anchor_embedding, real_same_embedding)
                y_score.append(same_score)
                speaker_sim_score.append(same_score)
                y_true.append(1)

                alternate_speaker = alt_indices[ridx][0]
                alternate_audio_idx = alt_indices[ridx][1]
                
                alternate_embedding = real_embeddings[alternate_speaker][alternate_audio_idx]
                y_score.append(get_similarity(anchor_embedding, alternate_embedding))
                y_true.append(0)

        speaker_sim_score = np.mean(speaker_sim_score)
        speaker_similarities[key] = float(speaker_sim_score)

        kidx += 1

    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    _auc = auc(fpr, tpr)
    fnr = 1 - tpr
    eer_threshold = thresholds[np.nanargmin(np.absolute((fnr - fpr)))]
    eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    eer_verify = fnr[np.nanargmin(np.absolute((fnr - fpr)))]

    assert abs(eer - eer_verify) < 1.0

    return {
        'mean_speaker_similarity': float(np.mean(list(speaker_similarities.values()))),
        'speaker_similarities': speaker_similarities,
        'eer': eer,
        'auc': _auc,
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate the model')
    parser.add_argument("--generated_audio_dir", type=str, required=True, default="/data/shehzeen/SSLTTS/ACEVCTRIALS/")
    parser.add_argument(
        "--gt_audio_dir", type=str, required=True, default="/data/shehzeen/SSLTTS/EVAL_SEEN_SPEAKERS_VCTK"
    )
    parser.add_argument(
        "--gt_source_audio_dir",
        type=str,
        required=True,
        default="/data/shehzeen/SSLTTS/EVAL_SEEN_SPEAKERS_VCTK_SOURCE",
    )
    parser.add_argument("--base_exp_dir", type=str, required=False, default="/data/shehzeen/SSLTTS/GENDER_EXP/RESULTS")

    args = parser.parse_args()

    generated_audio_dir = args.generated_audio_dir
    gt_audio_dir = args.gt_audio_dir
    gt_source_audio_dir = args.gt_source_audio_dir
    base_exp_dir = args.base_exp_dir

    exp_name = generated_audio_dir.split('/')[-1]
    device = "cpu"

    generated_audio_files = {}
    source_audio_files = {}

    wav_featurizer_sv = WaveformFeaturizer(sample_rate=16000, int_values=False, augmentor=None)

    for f in os.listdir(generated_audio_dir):
        if 'targetspeaker' in f and f.endswith('.wav'):
            # f = source_1_targetspeaker_6_0.wav
            speaker = f.split('_')[3]
            if ".wav" in speaker:
                speaker = speaker.split(".")[0]
            if "TO" in f:
                speaker = f.split('_')[2]
            if speaker not in generated_audio_files:
                generated_audio_files[speaker] = []
                source_audio_files[speaker] = []
            source_fname = "source_{}.wav".format(f.split('_')[1])
            if "TO" in f:
                source_num = f.split('_')[1].split("TO")[0]
                source_fname = "source_{}.wav".format(source_num)
            source_path = os.path.join(gt_source_audio_dir, source_fname)
            assert os.path.exists(source_path), "{} does not exist".format(source_path)

            generated_audio_files[speaker].append(os.path.join(generated_audio_dir, f))
            source_audio_files[speaker].append(source_path)

    gt_audio_files = {}
    for f in os.listdir(gt_audio_dir):
        if 'targetspeaker' in f and f.endswith('.wav'):
            # f = targetspeaker_19_4.wav
            speaker = f.split('_')[1]
            if ".wav" in speaker:
                speaker = speaker.split(".")[0]
            if speaker not in gt_audio_files:
                gt_audio_files[speaker] = []
            gt_audio_files[speaker].append(os.path.join(gt_audio_dir, f))

    print("generated_audio_files", generated_audio_files.keys())
    keys_to_remove = []
    for key in gt_audio_files:
        if key not in generated_audio_files:
            print("{} not in generated_audio_files".format(key))
            keys_to_remove.append(key)
    
    for key in keys_to_remove:
        del gt_audio_files[key]
            

    speaker_embeddings = {}
    transcriptions = {}

    sv_model_name = "speakerverification_speakernet"
    nemo_sv_model = label_models.EncDecSpeakerLabelModel.from_pretrained(sv_model_name)
    nemo_sv_model = nemo_sv_model.to(device)
    nemo_sv_model.eval()

    asr_model_name = "stt_en_quartznet15x5"
    asr_model = nemo_asr.models.EncDecCTCModel.from_pretrained(model_name=asr_model_name)
    asr_model = asr_model.to(device)
    asr_model.eval()

    wav2vec2_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-xlsr-53-espeak-cv-ft")
    wav2vec2_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-xlsr-53-espeak-cv-ft")
    wav2vec2_model = wav2vec2_model.to(device)
    wav2vec2_model.eval()

    kidx = 0
    source_transcriptions_nemo = []
    source_transcriptions_wav2vec2 = []
    generated_transcriptions_nemo = []
    generated_transcriptions_wav2vec2 = []
    source_paths = []
    generated_paths = []

    for key in gt_audio_files:
        speaker_embeddings["generated_{}".format(key)] = []
        speaker_embeddings["original_{}".format(key)] = []

        for fp in generated_audio_files[key]:
            embedding = nemo_sv_model.get_embedding(fp)
            embedding = embedding.cpu().detach().numpy().flatten()
            speaker_embeddings["generated_{}".format(key)].append(embedding)
            trancription = transcribe_nemo(asr_model, fp)
            transcription_wav2vec2 = transcribe_wav2vec2(wav2vec2_processor, wav2vec2_model, fp)

            generated_paths.append(fp)
            generated_transcriptions_nemo.append(trancription)
            generated_transcriptions_wav2vec2.append(transcription_wav2vec2)

        for fp in source_audio_files[key]:
            trancription = transcribe_nemo(asr_model, fp)
            trancription_wav2vec2 = transcribe_wav2vec2(wav2vec2_processor, wav2vec2_model, fp)
            
            source_paths.append(fp)
            source_transcriptions_nemo.append(trancription)
            source_transcriptions_wav2vec2.append(trancription_wav2vec2)

        for fp in gt_audio_files[key]:
            # Split the original audio into segments to get the embedding
            embeddings = get_speaker_embedding(nemo_sv_model, wav_featurizer_sv, [fp], device=device)
            speaker_embeddings["original_{}".format(key)] += embeddings

        kidx += 1
        # if kidx == 2:
        #     break

    generated_metrics = calculate_eer(speaker_embeddings, mode="generated")
    real_metrics = calculate_eer(speaker_embeddings, mode="real")
    cer_nemo, nemo_pairwise_cers = calculate_cer(source_transcriptions_nemo, generated_transcriptions_nemo, source_paths, generated_paths)
    cer_wav2vec2, wav2vec2_pairwise_cers = calculate_cer(source_transcriptions_wav2vec2, generated_transcriptions_wav2vec2, source_paths, generated_paths)

    generated_metrics['cer_nemo'] = cer_nemo
    generated_metrics['cer_wav2vec2'] = cer_wav2vec2
    generated_metrics['real_eer'] = real_metrics['eer']
    generated_metrics['real_auc'] = real_metrics['auc']
    generated_metrics['real_mean_speaker_similarity'] = real_metrics['mean_speaker_similarity']

    print(generated_metrics)

    detailed_cers = {
        'nemo': nemo_pairwise_cers,
        'wav2vec2': wav2vec2_pairwise_cers,
    }

    out_file_path = os.path.join(base_exp_dir, 'results_{}.json'.format(exp_name))
    with open(out_file_path, 'w') as f:
        json.dump(generated_metrics, f, indent=4)
    
    detailed_out_path = os.path.join(base_exp_dir, 'detailed_results_{}.json'.format(exp_name))
    with open(detailed_out_path, 'w') as f:
        json.dump(detailed_cers, f, indent=4)
    
    print("Saved to {}".format(out_file_path))
    print("Saved Detailed CERs to {}".format(detailed_out_path))


if __name__ == '__main__':
    main()
