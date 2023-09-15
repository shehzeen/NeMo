# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Example Run Command: python ssl_tts_vc.py --ssl_model_ckpt_path <PATH TO CKPT> --hifi_ckpt_path <PATH TO CKPT> \
# --fastpitch_ckpt_path <PATH TO CKPT> --source_audio_path <SOURCE CONTENT WAV PATH> --target_audio_path \
# <TARGET SPEAKER WAV PATH> --out_path <PATH TO OUTPUT WAV>

import argparse
import os

import librosa
import soundfile
import torch

import nemo.collections.asr as nemo_asr
from nemo.collections.asr.models import label_models
from nemo.collections.asr.parts.preprocessing.features import WaveformFeaturizer
from nemo.collections.tts.models import fastpitch_ssl, hifigan, ssl_tts
from nemo.collections.tts.torch.helpers import get_base_dir
import editdistance
import json

from numpy import dot
from numpy.linalg import norm
from scipy import linalg
from sklearn.manifold import TSNE
from sklearn.metrics import auc, roc_curve
import random
import numpy as np

def get_similarity(emb1, emb2):
    similarity = dot(emb1, emb2) / (norm(emb1) * norm(emb2))
    return similarity

def calculate_eer(generated_embeddings, real_embeddings, mode="generated"):
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
        'speaker_similarities': speaker_similarities,
        'eer': eer,
        'auc': _auc,
    }

def calculatePitchMetrics(ref_pitch_contour_np, gen_pitch_contour_np):
    # F0 Frame Error - FFE #
    NFOE_error = 0.0 
    NUV_ref = 0.0
    NVU_gen = 0.0
    NVV = 0.0
    N_total = 0.0
    non_zero_count = 0.0
    print(ref_pitch_contour_np.shape, gen_pitch_contour_np.shape)
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
    print("metric", metric_dict)
    
    return metric_dict

def compute_pitch_errors(wav_featurizer, source_paths, out_paths):
    assert len(source_paths) == len(out_paths)
    all_metrics = {
        "FFE": [],
        "GPE": [],
        "VDE": [],
    }
    for sidx, source_path in enumerate(source_paths):
        source_wav = load_wav(source_path, wav_featurizer, pad_multiple=1)
        source_pitch_contour = get_pitch_contour(source_wav)
        out_wav = load_wav(out_paths[sidx], wav_featurizer, pad_multiple=1)
        out_pitch_contour = get_pitch_contour(out_wav)
        pitch_metrics = calculatePitchMetrics(source_pitch_contour, out_pitch_contour)
        if pitch_metrics['GPE'] > 0.5:
            print("GPE > 0.5, skipping", source_path, out_paths[sidx])
            continue
        for metric_name in all_metrics:
            all_metrics[metric_name].append(pitch_metrics[metric_name])
    
    for metric_name in all_metrics:
        all_metrics[metric_name] = sum(all_metrics[metric_name]) / (1.0 * len(all_metrics[metric_name]))
    
    return all_metrics

def compute_CER(asr_model, source_paths, out_paths):
    assert len(source_paths) == len(out_paths)
    mean_cer = 0.0
    ctr = 0
    language_wise_cer = {}
    for sidx, source_path in enumerate(source_paths):
        # language = source_path.split("CSS10/")[1].split("/")[0]
        language = "en"
        if language not in language_wise_cer:
            language_wise_cer[language] = []
        t1 = asr_model.transcribe([source_path])[0]
        t2 = asr_model.transcribe([out_paths[sidx]])[0]
        print("t1", t1)
        print("t2", t2)
        if len(t1) > 0 and len(t2) > 0:
            cer = (1.0 * editdistance.eval(t1, t2)) / max(len(t1), len(t2))
        else:
            cer = 1.0
        # if cer > 0.2:
        print("CER", language, cer)
        #     continue
        ctr += 1
        language_wise_cer[language].append(cer)
        mean_cer += cer

    for language in language_wise_cer:
        language_wise_cer[language] = sum(language_wise_cer[language]) / (1.0 * len(language_wise_cer[language]))
    print("Language wise CER")
    print(language_wise_cer)
    mean_cer /= ctr
    return mean_cer

def load_wav(wav_path, wav_featurizer, pad_multiple=1024):
    wav = wav_featurizer.process(wav_path)
    # if wav is multi-channel, take the mean
    if wav.ndim > 1:
        wav = wav.mean(axis=1)

    if wav.shape[0] % pad_multiple != 0:
        wav = torch.cat([wav, torch.zeros(pad_multiple - wav.shape[0] % pad_multiple, dtype=torch.float)])
    wav = wav[:-1]

    return wav


def get_pitch_contour(wav, pitch_mean=None, pitch_std=None, compute_mean_std=False, sample_rate=22050):
    f0, _, _ = librosa.pyin(
        wav.numpy(),
        fmin=librosa.note_to_hz('C2'),
        fmax=librosa.note_to_hz('C7'),
        frame_length=1024,
        hop_length=256,
        sr=sample_rate,
        center=True,
        fill_na=0.0,
    )
    pitch_contour = torch.tensor(f0, dtype=torch.float32)
    _pitch_mean = pitch_contour.mean().item()
    _pitch_std = pitch_contour.std().item()
    if compute_mean_std:
        pitch_mean = _pitch_mean
        pitch_std = _pitch_std
    if (pitch_mean is not None) and (pitch_std is not None):
        pitch_contour = pitch_contour - pitch_mean
        pitch_contour[pitch_contour == -pitch_mean] = 0.0
        pitch_contour = pitch_contour / pitch_std

    return pitch_contour


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
        wav = load_wav(audio_path, wav_featurizer)
        segments = segment_wav(wav)
        all_segments += segments
        all_wavs.append(wav)
        if duration is not None and len(all_segments) >= duration:
            # each segment is 2 seconds with one second overlap.
            # so 10 segments would mean 0 to 2, 1 to 3.. 9 to 11 (11 seconds.)
            all_segments = all_segments[: int(duration)]
            break

    signal_batch = torch.stack(all_segments)
    signal_length_batch = torch.stack([torch.tensor(signal_batch.shape[1]) for _ in range(len(all_segments))])
    signal_batch = signal_batch.to(device)
    signal_length_batch = signal_length_batch.to(device)

    _, speaker_embeddings = nemo_sv_model(input_signal=signal_batch, input_signal_length=signal_length_batch)
    speaker_embedding = torch.mean(speaker_embeddings, dim=0)
    l2_norm = torch.norm(speaker_embedding, p=2)
    speaker_embedding = speaker_embedding / l2_norm

    return speaker_embedding[None]


def group_content_embeddings(content_embedding, duration, emb_similarity_threshold=0.925):
    # content_embedding: (256, n_timesteps)
    grouped_content_embeddings = [ content_embedding[:, 0] ]
    grouped_durations = [ duration[0] ]
    group_size = 1
    for _tidx in range(1, content_embedding.shape[1]):
        prev_embedding = grouped_content_embeddings[-1]
        curr_embedding = content_embedding[:, _tidx]
        emb_similarity = torch.cosine_similarity(prev_embedding, curr_embedding, dim=0)
        if emb_similarity < emb_similarity_threshold:
            grouped_content_embeddings.append(curr_embedding)
            grouped_durations.append(duration[_tidx])
        else:
            # group with previous embedding
            grouped_content_embeddings[-1] = (grouped_content_embeddings[-1] * group_size + curr_embedding) / (group_size + 1)
            grouped_durations[-1] += duration[_tidx]
    
    grouped_content_embeddings = torch.stack(grouped_content_embeddings, dim=1)
    grouped_durations = torch.stack(grouped_durations, dim=0)

    return grouped_content_embeddings, grouped_durations

def get_ssl_features_disentangled(ssl_model, wav_featurizer, audio_path, use_unique_tokens=False, device="cpu"):
    """
    Extracts content embedding, speaker embedding and duration tokens to be used as inputs for FastPitchModel_SSL 
    synthesizer. Content embedding and speaker embedding extracted using SSLDisentangler model.
    Args:
        ssl_model: SSLDisentangler model
        wav_featurizer: WaveformFeaturizer object
        audio_path: path to audio file
        device: device to run the model on
    Returns:
        content_embedding, speaker_embedding, duration
    """
    wav = load_wav(audio_path, wav_featurizer)
    audio_signal = wav[None]
    audio_signal_length = torch.tensor([wav.shape[0]])
    audio_signal = audio_signal.to(device)
    audio_signal_length = audio_signal_length.to(device)

    processed_signal, processed_signal_length = ssl_model.preprocessor(
        input_signal=audio_signal, length=audio_signal_length,
    )

    batch_content_embedding, batch_encoded_len = ssl_model.encoder(
        audio_signal=processed_signal, length=processed_signal_length
    )
    if ssl_model._cfg.get("normalize_content_encoding", False):
        batch_content_embedding = ssl_model._normalize_encoding(batch_content_embedding)

    content_embedding = batch_content_embedding[0, :, : batch_encoded_len[0]]
    ssl_downsampling_factor = ssl_model._cfg.encoder.subsampling_factor
    duration = torch.ones(content_embedding.shape[1]) * ssl_downsampling_factor
    
    if use_unique_tokens:
        print("Grouping..")
        emb_similarity_threshold = ssl_model._cfg.get("emb_similarity_threshold", 0.925)
        final_content_embedding, final_duration = group_content_embeddings(content_embedding, duration, emb_similarity_threshold)
        print("Grouped duration", final_duration)
    else:
        final_content_embedding, final_duration = content_embedding, duration
        
    final_content_embedding = final_content_embedding.to(device)
    final_duration = final_duration.to(device)

    return final_content_embedding[None], final_duration[None]


def main():
    parser = argparse.ArgumentParser(description='Evaluate the model')
    parser.add_argument('--ssl_model_ckpt_path', type=str)
    parser.add_argument('--hifi_ckpt_path', type=str)
    parser.add_argument('--fastpitch_ckpt_path', type=str)
    parser.add_argument('--manifest_path', type=str)
    parser.add_argument('--compute_pitch', type=int, default=1)
    parser.add_argument('--compute_duration', type=int, default=1)
    parser.add_argument('--max_input_length_sec', type=int, default=20)
    parser.add_argument('--segment_length_seconds', type=int, default=16)
    parser.add_argument('--use_unique_tokens', type=int, default=0)
    parser.add_argument('--duration', type=float, default=None)
    parser.add_argument('--base_out_dir', type=str, default="/data/shehzeen/SSLTTS/ReconTest")
    args = parser.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    combine_outpaths = False

    manifest_name = args.manifest_path.split("/")[-1].split(".")[0]
    fastpitch_name = args.fastpitch_ckpt_path.split("/")[-1].split(".")[0]
    if args.compute_duration == 1:
        fastpitch_name += "_predictive"
    else:
        fastpitch_name += "_guided"
    exp_name = "{}_{}".format(manifest_name, fastpitch_name)

    out_dir = os.path.join(args.base_out_dir, exp_name)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    with open(args.manifest_path) as f:
        all_lines = f.readlines()
        speaker_wise_data = {}
        for line in all_lines:
            record = json.loads(line)
            if record['duration'] > 5.0 and record['duration'] < 16.0:
                speaker = record["speaker"]
                if speaker not in speaker_wise_data:
                    speaker_wise_data[speaker] = []
                speaker_wise_data[speaker].append(record)

    valid_speakers = [ spk for spk in speaker_wise_data if len(speaker_wise_data[spk]) >= 10 ]
    valid_speakers.sort()
    target_speakers = valid_speakers[:10]
    source_target_out_pairs = []

    source_paths = []
    out_paths = []
    speaker_wise_outpaths = {}
    for speaker in target_speakers:
        speaker_wise_outpaths[speaker] = []
        for ridx, record in enumerate(speaker_wise_data[speaker][:10]):
            source_path = record['audio_filepath']
            target_path = record['audio_filepath']
            out_path = os.path.join(out_dir, "{}_{}.wav".format(speaker, ridx))
            source_paths.append(source_path)
            out_paths.append(out_path)
            source_target_out_pairs.append((source_path, target_path, out_path))
            speaker_wise_outpaths[speaker].append(out_path)

    print("source_target_out_pairs")
    print(source_target_out_pairs)

    ssl_model = nemo_asr.models.ssl_models.SpeechEncDecSelfSupervisedModel.load_from_checkpoint(
        args.ssl_model_ckpt_path
    )
    ssl_model.eval()
    ssl_model.to(device)

    nemo_sv_model = label_models.EncDecSpeakerLabelModel.from_pretrained("titanet_large")
    nemo_sv_model = nemo_sv_model.to(device)
    nemo_sv_model.eval()
    sv_sample_rate = nemo_sv_model._cfg.preprocessor.sample_rate

    nemo_sv_model_eval = label_models.EncDecSpeakerLabelModel.from_pretrained("speakerverification_speakernet")
    nemo_sv_model_eval = nemo_sv_model_eval.to(device)
    nemo_sv_model_eval.eval()

    vocoder = hifigan.HifiGanModel.load_from_checkpoint(args.hifi_ckpt_path).to(device)
    vocoder.eval()

    fastpitch_model = fastpitch_ssl.FastPitchModel_SSL.load_from_checkpoint(args.fastpitch_ckpt_path, strict=False)
    fastpitch_model = fastpitch_model.to(device)
    fastpitch_model.eval()
    fastpitch_model.non_trainable_models = {'vocoder': vocoder}
    fpssl_sample_rate = fastpitch_model._cfg.sample_rate

    asr_model_name = "stt_en_quartznet15x5"
    asr_model = nemo_asr.models.EncDecCTCModel.from_pretrained(model_name=asr_model_name)
    asr_model = asr_model.to(device)
    asr_model.eval()

    wav_featurizer = WaveformFeaturizer(sample_rate=fpssl_sample_rate, int_values=False, augmentor=None)
    wav_featurizer_sv = WaveformFeaturizer(sample_rate=sv_sample_rate, int_values=False, augmentor=None)

    use_unique_tokens = args.use_unique_tokens == 1
    compute_pitch = args.compute_pitch == 1
    compute_duration = args.compute_duration == 1


    if False:
        for source_target_out in source_target_out_pairs:
            source_audio_path = source_target_out[0]
            source_audio_length = wav_featurizer.process(source_audio_path).shape[0]
            target_audio_paths = source_target_out[1].split(",")
            out_path = source_target_out[2]

            with torch.no_grad():
                content_embedding1, duration1 = get_ssl_features_disentangled(
                    ssl_model, wav_featurizer, source_audio_path, use_unique_tokens, device=device,
                )

                speaker_embedding2 = get_speaker_embedding(
                    nemo_sv_model, wav_featurizer_sv, target_audio_paths, duration=args.duration, device=device
                )

                pitch_contour1 = None
                if not compute_pitch:
                    pitch_contour1 = get_pitch_contour(
                        load_wav(source_audio_path, wav_featurizer), compute_mean_std=True, sample_rate=fpssl_sample_rate
                    )[None]
                    pitch_contour1 = pitch_contour1.to(device)

                wav_generated = fastpitch_model.generate_wav(
                    content_embedding1,
                    speaker_embedding2,
                    pitch_contour=pitch_contour1,
                    compute_pitch=compute_pitch,
                    compute_duration=compute_duration,
                    durs_gt=duration1,
                    dataset_id=0,
                )
                wav_generated = wav_generated[0][0][:source_audio_length]
                soundfile.write(out_path, wav_generated, fpssl_sample_rate)

    if combine_outpaths:
        print("Combining segments into one file")
        out_paths = [r[2] for r in source_target_out_pairs]
        out_wavs = [wav_featurizer.process(out_path) for out_path in out_paths]
        out_wav = torch.cat(out_wavs, dim=0).cpu().numpy()
        soundfile.write(args.out_path, out_wav, fpssl_sample_rate)

    cer = compute_CER(asr_model, source_paths, out_paths)
    print("CER: {}".format(cer))
    
    pitch_metrics = compute_pitch_errors(wav_featurizer, source_paths, out_paths)
    print("Pitch Metrics", pitch_metrics)


    generated_embeddings = {}
    real_embeddings = {}
    for spk in speaker_wise_outpaths:
        generated_embeddings[spk] = []
        for path in speaker_wise_outpaths[spk]:
            embedding = nemo_sv_model_eval.get_embedding(path)
            embedding = embedding.cpu().detach().numpy().flatten()
            generated_embeddings[spk].append(embedding)
        
        real_embeddings[spk] = []
        for path in speaker_wise_data[spk][:10]:
            embedding = nemo_sv_model_eval.get_embedding(path['audio_filepath'])
            embedding = embedding.cpu().detach().numpy().flatten()
            real_embeddings[spk].append(embedding)
    
    eer_generated = calculate_eer(generated_embeddings, real_embeddings, mode="generated")
    eer_real = calculate_eer(generated_embeddings, real_embeddings, mode="real")

    print("EER Generated", eer_generated)
    print("EER Real", eer_real)

    all_results = {
        'cer': cer,
        'pitch_metrics': pitch_metrics,
        'eer_generated': eer_generated,
        'eer_real': eer_real,
    }

    print(os.path.join(out_dir, "results.json"))
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(all_results, f, indent=4)

    

if __name__ == "__main__":
    main()
