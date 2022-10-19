import argparse
import fnmatch
import json
import os
import pickle
import random

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
from transformers import Wav2Vec2Model, Wav2Vec2Processor

from nemo.collections.asr.models import label_models
from nemo.collections.asr.parts.preprocessing.features import WaveformFeaturizer
from nemo.collections.tts.models import fastpitch_ssl, hifigan, ssl_tts
import nemo.collections.asr as nemo_asr
import editdistance

def get_similarity(emb1, emb2):
    similarity = dot(emb1, emb2) / (norm(emb1) * norm(emb2))
    return similarity


def calculate_cer(transcriptions):
    generated_transcriptions = []
    real_transcriptions = []

    for key in transcriptions:
        if 'generated' in key:
            real_key = key.replace('generated', 'original')
            assert real_key in transcriptions, "{} not in transcriptions".format(real_key)
            real_transcriptions += transcriptions[real_key]
            generated_transcriptions += transcriptions[key]
    # print(len(generated_transcriptions))
    # print(len(real_transcriptions))
    mean_cer = 0.0
    ctr = 0
    for idx in range(len(generated_transcriptions)):
        t1 = generated_transcriptions[idx]
        t2 = real_transcriptions[idx]
        if len(t1) > 0 and len(t2) > 0:
            cer = (1.0*editdistance.eval(t1, t2)) / max(len(t1), len(t2))
        else:
            cer = 1.0
        ctr += 1 
        mean_cer += cer
    mean_cer /= ctr

    return mean_cer
    

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
    for key in anchor_embeddings:
        speaker_sim_score = []
        alternate_keys = [k for k in real_embeddings if k != key]
        for aidx, anchor_embedding in enumerate(anchor_embeddings[key]):
            for ridx, real_same_embedding in enumerate(real_embeddings[key]):
                if mode == "real" and ridx == aidx:
                    # skip if same speaker and same utterance
                    continue

                same_score = get_similarity(anchor_embedding, real_same_embedding)
                y_score.append(same_score)
                speaker_sim_score.append(same_score)
                y_true.append(1)

                alternate_speaker = random.choice(alternate_keys)
                alternate_audio_idx = random.randint(0, len(real_embeddings[alternate_speaker]) - 1)
                alternate_embedding = real_embeddings[alternate_speaker][alternate_audio_idx]
                y_score.append(get_similarity(anchor_embedding, alternate_embedding))
                y_true.append(0)

        speaker_sim_score = np.mean(speaker_sim_score)
        speaker_similarities[key] = float(speaker_sim_score)

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
    generated_audio_dir = '/data/shehzeen/AutoVC/autovc_converted'
    # gt_audio_dir = '/data/shehzeen/SSLTTS/EVAL_SEEN_SPEAKERS_ALL/'
    gt_audio_dir = "/data/shehzeen/SSLTTS/EVALDATA"
    # gt_audio_dir = '/data/shehzeen/SSLTTS/EVAL_SEEN_SPEAKERS_VCTK'
    gt_source_audio_dir = '/data/shehzeen/SSLTTS/EVALDATA/'
    base_exp_dir = '/data/shehzeen/SSLTTS/'
    exp_name = generated_audio_dir.split('/')[-1]
    device = "cpu"

    generated_audio_files = {}
    source_audio_files = {}
    for f in os.listdir(generated_audio_dir):
        if 'targetspeaker' in f and f.endswith('.wav'):
            # f = source_1_targetspeaker_6_0.wav
            speaker = f.split('_')[3]
            if speaker not in generated_audio_files:
                generated_audio_files[speaker] = []
                source_audio_files[speaker] = []
            source_fname = "source_{}.wav".format(f.split('_')[1])
            source_path = os.path.join(gt_source_audio_dir, source_fname)
            assert os.path.exists(source_path), "{} does not exist".format(source_path)

            generated_audio_files[speaker].append(os.path.join(generated_audio_dir, f))
            source_audio_files[speaker].append(source_path)
    
    gt_audio_files = {}
    for f in os.listdir(gt_audio_dir):
        if 'targetspeaker' in f and f.endswith('.wav'):
            # f = targetspeaker_19_4.wav
            speaker = f.split('_')[1]
            if speaker not in gt_audio_files:
                gt_audio_files[speaker] = []
            gt_audio_files[speaker].append(os.path.join(gt_audio_dir, f))

    for key in gt_audio_files:
        assert key in generated_audio_files, "{} not in generated_audio_files".format(key)
    
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

    kidx = 0
    for key in gt_audio_files:
        speaker_embeddings["generated_{}".format(key)] = []
        speaker_embeddings["original_{}".format(key)] = []
        transcriptions["generated_{}".format(key)] = []
        transcriptions["original_{}".format(key)] = []

        for fp in generated_audio_files[key]:
            print("getting embedding for {}".format(fp))
            embedding = nemo_sv_model.get_embedding(fp)
            embedding = embedding.cpu().detach().numpy().flatten()
            speaker_embeddings["generated_{}".format(key)].append(embedding)
            trancription = asr_model.transcribe([fp])[0]
            print("Transcription generated: {}".format(trancription))
            transcriptions["generated_{}".format(key)].append(trancription)

        for fp in source_audio_files[key]:
            trancription = asr_model.transcribe([fp])[0]
            print("Transcription original: {}".format(trancription))
            transcriptions["original_{}".format(key)].append(trancription)

        for fp in gt_audio_files[key]:
            print("getting embedding for {}".format(fp))
            embedding = nemo_sv_model.get_embedding(fp)
            embedding = embedding.cpu().detach().numpy().flatten()
            speaker_embeddings["original_{}".format(key)].append(embedding)
        
        kidx += 1
        # if kidx == 2:
        #     break

    for key in transcriptions:
        print (key, len(transcriptions[key]))
    print("results real")

    # real_eer = calculate_eer(speaker_embeddings, mode="real")
    # print(real_eer)
    print("---------------------------------------------------------------------------------")
    print("results generated")
    generated_metrics = calculate_eer(speaker_embeddings, mode="generated")
    generated_metrics['cer'] = calculate_cer(transcriptions)

    print(generated_metrics)

    out_file_path = os.path.join(base_exp_dir, 'results_{}.json'.format(exp_name))
    
    with open(out_file_path, 'w') as f:
        json.dump(generated_metrics, f, indent=4)
    
    

if __name__ == '__main__':
    main()