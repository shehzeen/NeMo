from flask import Flask
from flask_cors import CORS
from nemo.collections.tts.models import fastpitch_ssl, hifigan
import nemo.collections.asr as nemo_asr
from nemo.collections.asr.models import label_models
import torch
from nemo.collections.asr.parts.preprocessing.features import WaveformFeaturizer
from flask import request
import base64
import numpy as np
import soundfile as sf
import librosa
import json
import time
import os
import io
# ssl_model_ckpt_path = "/data/shehzeen/SSLTTS/PretrainingExperiments/AugLossAlpha100/Conformer-SSL/2023-01-24_00-42-05/checkpoints/Epoch68.ckpt"
ssl_model_ckpt_path = "/data/shehzeen/SSLTTS/PretrainingExperiments/MultiLing256/FPMEL_AllFixed_Unnorm/2023-02-19_20-59-04/checkpoints/Epoch43_8915.ckpt"
# hifi_ckpt_path = "/data/shehzeen/SSLTTS/HiFiLibriEpoch334.ckpt"
# fastpitch_ckpt_path = "/data/shehzeen/SSLTTS/TextlessFastPitchExperiments/AugmentedTraining/2023-01-26_14-00-48/checkpoints/Epoch89.ckpt"
# target_audio_paths = ["/data/shehzeen/SSLTTS/EVALDATA/source_2.wav"]

temp_dir = "/data/shehzeen/temp_vc_audio/"
if not os.path.exists(temp_dir):
    os.makedirs(temp_dir)

target_audio_paths = {
    'obama' : [
        "/data/shehzeen/SSLTTS/CelebrityData/YoutubeChunkedAudio/obama_2_335.wav",
        "/data/shehzeen/SSLTTS/CelebrityData/YoutubeChunkedAudio/obama_1_335.wav",
        "/data/shehzeen/SSLTTS/CelebrityData/YoutubeChunkedAudio/obama_1_336.wav",
        "/data/shehzeen/SSLTTS/CelebrityData/YoutubeChunkedAudio/obama_1_337.wav"
    ],
    'modi' : [
        "/data/shehzeen/SSLTTS/CelebrityData/YoutubeChunkedAudio/modi_1_334.wav",
        "/data/shehzeen/SSLTTS/CelebrityData/YoutubeChunkedAudio/modi_1_335.wav",
        "/data/shehzeen/SSLTTS/CelebrityData/YoutubeChunkedAudio/modi_1_336.wav",
        "/data/shehzeen/SSLTTS/CelebrityData/YoutubeChunkedAudio/modi_1_337.wav",
    ],
    'ravish' : [
        "/data/shehzeen/SSLTTS/CelebrityData/YoutubeChunkedAudio/ravish_2_141.wav",
        "/data/shehzeen/SSLTTS/CelebrityData/YoutubeChunkedAudio/ravish_2_142.wav",
        "/data/shehzeen/SSLTTS/CelebrityData/YoutubeChunkedAudio/ravish_2_143.wav",
        "/data/shehzeen/SSLTTS/CelebrityData/YoutubeChunkedAudio/ravish_2_144.wav",
    ],
    'lex' : [
        "/data/shehzeen/SSLTTS/CelebrityData/YoutubeChunkedAudio/lex_1_290.wav",
        "/data/shehzeen/SSLTTS/CelebrityData/YoutubeChunkedAudio/lex_1_291.wav",
        "/data/shehzeen/SSLTTS/CelebrityData/YoutubeChunkedAudio/lex_1_292.wav",
        "/data/shehzeen/SSLTTS/CelebrityData/YoutubeChunkedAudio/lex_1_293.wav",
    ],
    'oprah' : [
        "/data/shehzeen/SSLTTS/CelebrityData/YoutubeChunkedAudio/oprah_2_151.wav",
        "/data/shehzeen/SSLTTS/CelebrityData/YoutubeChunkedAudio/oprah_2_152.wav",
        "/data/shehzeen/SSLTTS/CelebrityData/YoutubeChunkedAudio/oprah_2_153.wav",
        "/data/shehzeen/SSLTTS/CelebrityData/YoutubeChunkedAudio/oprah_2_154.wav",
    ],
    'emma' : [
        "/data/shehzeen/SSLTTS/CelebrityData/YoutubeChunkedAudio/emma_1_2.wav",
        "/data/shehzeen/SSLTTS/CelebrityData/YoutubeChunkedAudio/emma_1_5.wav",
        "/data/shehzeen/SSLTTS/CelebrityData/YoutubeChunkedAudio/emma_1_8.wav",
    ],
    'priyanka' : [
        "/data/shehzeen/SSLTTS/CelebrityData/YoutubeChunkedAudio/priyanka_1_2.wav",
        "/data/shehzeen/SSLTTS/CelebrityData/YoutubeChunkedAudio/priyanka_1_8.wav",
        "/data/shehzeen/SSLTTS/CelebrityData/YoutubeChunkedAudio/priyanka_1_11.wav",
    ],
    'miley' : [
        "/data/shehzeen/SSLTTS/CelebrityData/YoutubeChunkedAudio/miley_1_2.wav",
        "/data/shehzeen/SSLTTS/CelebrityData/YoutubeChunkedAudio/miley_1_5.wav",
        "/data/shehzeen/SSLTTS/CelebrityData/YoutubeChunkedAudio/miley_1_11.wav",
    ],
    'aubrey' : [
        "/data/shehzeen/SSLTTS/CelebrityData/YoutubeChunkedAudio/aubrey_1_5.wav",
        "/data/shehzeen/SSLTTS/CelebrityData/YoutubeChunkedAudio/aubrey_1_8.wav",
        "/data/shehzeen/SSLTTS/CelebrityData/YoutubeChunkedAudio/aubrey_1_11.wav",
    ],
    'sundar' : [
        "/data/shehzeen/SSLTTS/CelebrityData/YoutubeChunkedAudio/sundar_1_5.wav",
        "/data/shehzeen/SSLTTS/CelebrityData/YoutubeChunkedAudio/sundar_1_8.wav",
        "/data/shehzeen/SSLTTS/CelebrityData/YoutubeChunkedAudio/sundar_1_11.wav",
    ],
    'ahmadCorrect' : [
        "/data/shehzeen/SSLTTS/CelebrityData/YoutubeChunkedAudio/ahmadCorrect_1_5.wav",
        "/data/shehzeen/SSLTTS/CelebrityData/YoutubeChunkedAudio/ahmadCorrect_1_8.wav",
        "/data/shehzeen/SSLTTS/CelebrityData/YoutubeChunkedAudio/ahmadCorrect_1_11.wav",
    ]
}

# fastpitch_ckpt_path = "/data/shehzeen/SSLTTS/CelebrityFastPitch/CelebrityLexOprah/2023-01-29_16-31-54/checkpoints/Epoch167.ckpt"
# fastpitch_ckpt_path = "/data/shehzeen/SSLTTS/CelebrityFastPitch/CelebrityFemailSpeakers/2023-01-30_14-41-11/checkpoints/Epoch300.ckpt"
fastpitch_ckpt_path = "/data/shehzeen/SSLTTS/TextlessFastPitchExperiments2/FPCompNew_MLM_Auh100_NoNorm/2023-02-23_12-51-14/checkpoints/Epoch500_5797.ckpt"
# hifi_ckpt_path = "/data/shehzeen/SSLTTS/HifiGANObama/HifiGan/2023-01-28_19-02-46/checkpoints/Epoch909.ckpt"
# hifi_ckpt_path = "/data/shehzeen/SSLTTS/HifiGANCelebrity/HifiGan/2023-01-29_16-23-01/checkpoints/Epoch69.ckpt"
# hifi_ckpt_path = "/data/shehzeen/SSLTTS/HifiGANOnSynth/HifiGan/2023-01-29_21-35-32/checkpoints/Epoch799.ckpt"
# hifi_ckpt_path = "/data/shehzeen/SSLTTS/HifiGANOnSynthNewCelebs/HifiGan/2023-01-30_18-23-17/checkpoints/Epoch1019.ckpt"
hifi_ckpt_path = "/data/shehzeen/SSLTTS/TextlessFastPitchExperimentsCeleb/HifiGAN_finetuned/HifiGan/2023-03-07_08-58-05/checkpoints/Epoch7659_3494.ckpt"
# hifi_ckpt_path = "/data/shehzeen/SSLTTS/HiFiLibriEpoch334.ckpt"
device = "cuda:0" if torch.cuda.is_available() else "cpu"

print("Loading SSL Model")
ssl_model = nemo_asr.models.ssl_models.SpeechEncDecSelfSupervisedModel.load_from_checkpoint(ssl_model_ckpt_path)
ssl_model.eval()
ssl_model.to(device)

print("Loading SV Model")
nemo_sv_model = label_models.EncDecSpeakerLabelModel.from_pretrained("titanet_large")
nemo_sv_model = nemo_sv_model.to(device)
nemo_sv_model.eval()
sv_sample_rate = nemo_sv_model._cfg.preprocessor.sample_rate

print("Loading Vocoder Model")
vocoder = hifigan.HifiGanModel.load_from_checkpoint(hifi_ckpt_path).to(device)
vocoder.eval()

print("Loading FP Model")
fastpitch_model = fastpitch_ssl.FastPitchModel_SSL.load_from_checkpoint(fastpitch_ckpt_path, strict=False)
fastpitch_model = fastpitch_model.to(device)
fastpitch_model.eval()
fastpitch_model.non_trainable_models = {'vocoder': vocoder}
fpssl_sample_rate = fastpitch_model._cfg.sample_rate

wav_featurizer_sv = WaveformFeaturizer(sample_rate=sv_sample_rate, int_values=False, augmentor=None)
wav_featurizer_fp = WaveformFeaturizer(sample_rate=fpssl_sample_rate, int_values=False, augmentor=None)

print("Loading VAD Model")
vad_model, vad_utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                              model='silero_vad',
                              force_reload=True,
                              onnx=False)

(vad_get_speech_timestamps,
 vad_save_audio,
 vad_read_audio,
 vad_VADIterator,
 vad_collect_chunks) = vad_utils

vad_model = vad_model.to("cpu")

def load_wav(wav_path, wav_featurizer, pad_multiple=1024):
    wav = wav_featurizer.process(wav_path)
    if (wav.shape[0] % pad_multiple) != 0:
        wav = torch.cat([wav, torch.zeros(pad_multiple - wav.shape[0] % pad_multiple, dtype=torch.float)])
    wav = wav[:-1]

    return wav

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
    
    _, speaker_embeddings = nemo_sv_model(
        input_signal=signal_batch, input_signal_length=signal_length_batch
    )
    speaker_embedding = torch.mean(speaker_embeddings, dim=0)
    l2_norm = torch.norm(speaker_embedding, p=2)
    speaker_embedding = speaker_embedding / l2_norm

    return speaker_embedding[None]


with torch.no_grad():
    speaker_embeddings = {}
    for speaker in target_audio_paths:
        paths = target_audio_paths[speaker]
        speaker_embedding = get_speaker_embedding(
            nemo_sv_model, wav_featurizer_sv, paths, duration=None, device=device
        )
        speaker_embeddings[speaker] = speaker_embedding

app = Flask(__name__)
CORS(app)

@app.route('/test_connection')
def test_connection():
    return 'works!'


# Write a POST view to accept audio base64 and process it through the vc model
@app.route('/convert_recordings', methods=['POST'])
def convert_recordings():
    total_wavs = int(request.values.get('total_wavs'))


    results = []
    for wav_no in range(total_wavs):
        with torch.no_grad():
            source_type = request.values.get('input_type_{}'.format(wav_no))
            if source_type == "recording":
                audio_data = request.files['audio_data_{}'.format(wav_no)]
            else:
                audio_data = request.files['custom_source_audio_{}'.format(wav_no)]
            
            # source_audio_path = os.path.join(temp_dir, 'source_audio_{}.wav'.format(wav_no))
            # print("Audio Saved to: {}".format(source_audio_path))
            # audio_data.save(source_audio_path)
            source_wav = load_wav(audio_data, wav_featurizer_fp)
            audio_np = source_wav.cpu().numpy()

            speaker = request.values.get('speaker_{}'.format(wav_no))
            if speaker == "custom":
                target_audio_data = request.files['target_speaker_audio_{}'.format(wav_no)]
                # target_audio_path = os.path.join(temp_dir, 'target_speaker_audio_{}.wav'.format(wav_no))
                # target_audio_data.save(target_audio_path)
                # print("Audio Saved to: {}".format(target_audio_path))
                target_speaker_embedding = get_speaker_embedding(
                    nemo_sv_model, wav_featurizer_sv, [target_audio_data], duration=None, device=device
                )
            else:
                target_speaker_embedding = speaker_embeddings[speaker]

            # audio_np, _ = librosa.load(audio_data, sr=22050)
            segment_length = int(16*20480)
            seg_num = 0
            num_segments = int(np.ceil(len(audio_np) / segment_length))
            generated_wavs = []
            for seg_num in range(num_segments):
                print("segment {}/{}".format(seg_num, num_segments))
                start = seg_num * segment_length
                end = min((seg_num + 1) * segment_length, len(audio_np))
                audio_signal = torch.from_numpy(audio_np[start:end]).to(device)[None]
                audio_signal_length = torch.tensor([audio_signal.shape[1]]).to(device)
                processed_signal, processed_signal_length = ssl_model.preprocessor(
                    input_signal=audio_signal, length=audio_signal_length,
                )

                batch_content_embedding, batch_encoded_len = ssl_model.encoder(audio_signal=processed_signal, length=processed_signal_length)
                final_content_embedding = batch_content_embedding[0,:,:batch_encoded_len[0]]
                ssl_downsampling_factor = ssl_model._cfg.encoder.subsampling_factor
                duration = torch.ones(final_content_embedding.shape[1]) * ssl_downsampling_factor
                duration = duration.to(device)
                final_content_embedding = final_content_embedding[None]
                duration = duration[None]

                wav_generated = fastpitch_model.generate_wav(
                    final_content_embedding,
                    target_speaker_embedding,
                    pitch_contour=None,
                    compute_pitch=True,
                    compute_duration=False,
                    durs_gt=duration,
                    dataset_id=0,
                )

                wav_generated = wav_generated[0][0]
                generated_wavs.append(wav_generated)
            
            wav_generated = np.concatenate(generated_wavs)
            # temp_wav_path = os.path.join(temp_dir, 'temp.wav')
            temp_buffer = io.BytesIO()
            # sf.write(temp_wav_path, wav_generated, 22050)
            sf.write(temp_buffer, wav_generated, 22050, format='WAV')
            # print("Converted", temp_wav_path)
            # with open(temp_wav_path, 'rb') as f:
            #     audio_base64_converted = base64.b64encode(f.read())
            audio_base64_converted = base64.b64encode(temp_buffer.getvalue())
            
            results.append({
                'audio_converted': audio_base64_converted.decode('ascii'),
            })
    
    return json.dumps({
        'total_wavs': total_wavs,
        'results': results,
    })


@app.route('/convert_voice', methods=['POST'])
def convert_voice():
    # get base64 audio from request
    # convert base64 to wav
    # get speaker embedding
    # run vc model

    audio_base64 = request.values.get('audio')
    
    speaker = request.values.get('speaker')
    if speaker not in speaker_embeddings:
        print("speaker not found, using default speaker")
        speaker = speaker_embeddings.keys()[0]
        print("chose speaker", speaker)

    audio_base64 = base64.b64decode(audio_base64)
    audio_np = np.frombuffer(audio_base64, dtype=np.float32)
    

    audio_np_16000 = librosa.resample(audio_np, 22050, 16000)

    audio_torch_16000 = torch.from_numpy(audio_np_16000)
    speech_timestamps = vad_get_speech_timestamps(audio_torch_16000, vad_model, sampling_rate=16000)
    print("speech timestamps ", speech_timestamps)
    if len(speech_timestamps) == 0:
        silence = audio_np * 0
        return json.dumps({
            'audio_converted': base64.b64encode(silence).decode('utf-8'),
        })
    
    st = time.time()
    with torch.no_grad():
        audio_np = audio_np[:-1]
        audio_signal = torch.from_numpy(audio_np).to(device)[None]

        audio_signal_length = torch.tensor([audio_signal.shape[1]]).to(device)

        processed_signal, processed_signal_length = ssl_model.preprocessor(
            input_signal=audio_signal, length=audio_signal_length,
        )
        batch_content_embedding, batch_encoded_len = ssl_model.encoder(audio_signal=processed_signal, length=processed_signal_length)
        final_content_embedding = batch_content_embedding[0,:,:batch_encoded_len[0]]
        ssl_downsampling_factor = ssl_model._cfg.encoder.subsampling_factor
        duration = torch.ones(final_content_embedding.shape[1]) * ssl_downsampling_factor
        duration = duration.to(device)
        final_content_embedding = final_content_embedding[None]
        duration = duration[None]

        wav_generated = fastpitch_model.generate_wav(
            final_content_embedding,
            speaker_embeddings[speaker],
            pitch_contour=None,
            compute_pitch=True,
            compute_duration=False,
            durs_gt=duration,
            dataset_id=0,
        )

        wav_generated = wav_generated[0][0]
        print("wav generated ", wav_generated.shape, wav_generated.dtype)
        print("***\n***\n***\n***\n***\n***\n")
        # wav_pitch_shifted = librosa.effects.pitch_shift(audio_np, 22050, n_steps=-2)
    audio_time = time.time() - st
    print("audio time ", audio_time)

    return json.dumps({
        'audio_converted': base64.b64encode(wav_generated).decode('utf-8'),
    })
    

if __name__ == '__main__':
    app.run()