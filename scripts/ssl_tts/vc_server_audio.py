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
from flask_ngrok import run_with_ngrok
import random

temp_dir = "/data/shehzeen/TextlessVCDemoTempDir"
os.makedirs(temp_dir, exist_ok=True)

data_dir_path = "/data/shehzeen/SSLTTS/CelebrityData"

ssl_model_ckpt_path = "/data/shehzeen/TextlessVCDemoCkpts/Epoch43_8915.ckpt"

avatar_paths = {
    'default' : "/home/shehzeen/first-order-model/obama.jpg",
    'obama' : "/home/shehzeen/first-order-model/obama.jpg",
    'ahmadCorrect' : "/home/shehzeen/ahmad.png",
    'sundar' : "/home/shehzeen/sundar.png",
    'modi' : "/home/shehzeen/modi.png",
    'emma' : "/home/shehzeen/emma.png",
    'priyanka' : "/home/shehzeen/priyanka.png",
    'ravish' : "/data/shehzeen/portraits/ravish.png",
    'aubrey' : "/data/shehzeen/portraits/aubrey.png",
    'lex' : "/data/shehzeen/portraits/lex.png",
    'oprah' : "/data/shehzeen/portraits/oprah.png",
    'miley' : "/data/shehzeen/portraits/miley.png",
}

target_audio_paths = {
    'obama' : [
        "{}/YoutubeChunkedAudio/obama_2_335.wav".format(data_dir_path),
        "{}/YoutubeChunkedAudio/obama_1_335.wav".format(data_dir_path),
        "{}/YoutubeChunkedAudio/obama_1_336.wav".format(data_dir_path),
        "{}/YoutubeChunkedAudio/obama_1_337.wav".format(data_dir_path)
    ],
    'modi' : [
        "{}/YoutubeChunkedAudio/modi_1_334.wav".format(data_dir_path),
        "{}/YoutubeChunkedAudio/modi_1_335.wav".format(data_dir_path),
        "{}/YoutubeChunkedAudio/modi_1_336.wav".format(data_dir_path),
        "{}/YoutubeChunkedAudio/modi_1_337.wav".format(data_dir_path),
    ],
    'ravish' : [
        "{}/YoutubeChunkedAudio/ravish_2_141.wav".format(data_dir_path),
        "{}/YoutubeChunkedAudio/ravish_2_142.wav".format(data_dir_path),
        "{}/YoutubeChunkedAudio/ravish_2_143.wav".format(data_dir_path),
        "{}/YoutubeChunkedAudio/ravish_2_144.wav".format(data_dir_path),
    ],
    'lex' : [
        "{}/YoutubeChunkedAudio/lex_1_290.wav".format(data_dir_path),
        "{}/YoutubeChunkedAudio/lex_1_291.wav".format(data_dir_path),
        "{}/YoutubeChunkedAudio/lex_1_292.wav".format(data_dir_path),
        "{}/YoutubeChunkedAudio/lex_1_293.wav".format(data_dir_path),
    ],
    'oprah' : [
        "{}/YoutubeChunkedAudio/oprah_2_151.wav".format(data_dir_path),
        "{}/YoutubeChunkedAudio/oprah_2_152.wav".format(data_dir_path),
        "{}/YoutubeChunkedAudio/oprah_2_153.wav".format(data_dir_path),
        "{}/YoutubeChunkedAudio/oprah_2_154.wav".format(data_dir_path),
    ],
    'emma' : [
        "{}/YoutubeChunkedAudio/emma_1_2.wav".format(data_dir_path),
        "{}/YoutubeChunkedAudio/emma_1_5.wav".format(data_dir_path),
        "{}/YoutubeChunkedAudio/emma_1_8.wav".format(data_dir_path),
    ],
    'priyanka' : [
        "{}/YoutubeChunkedAudio/priyanka_1_2.wav".format(data_dir_path),
        "{}/YoutubeChunkedAudio/priyanka_1_8.wav".format(data_dir_path),
        "{}/YoutubeChunkedAudio/priyanka_1_11.wav".format(data_dir_path),
    ],
    'miley' : [
        "{}/YoutubeChunkedAudio/miley_1_2.wav".format(data_dir_path),
        "{}/YoutubeChunkedAudio/miley_1_5.wav".format(data_dir_path),
        "{}/YoutubeChunkedAudio/miley_1_11.wav".format(data_dir_path),
    ],
    'aubrey' : [
        "{}/YoutubeChunkedAudio/aubrey_1_5.wav".format(data_dir_path),
        "{}/YoutubeChunkedAudio/aubrey_1_8.wav".format(data_dir_path),
        "{}/YoutubeChunkedAudio/aubrey_1_11.wav".format(data_dir_path),
    ],
    'sundar' : [
        "{}/YoutubeChunkedAudio/sundar_1_5.wav".format(data_dir_path),
        "{}/YoutubeChunkedAudio/sundar_1_8.wav".format(data_dir_path),
        "{}/YoutubeChunkedAudio/sundar_1_11.wav".format(data_dir_path),
    ],
    'ahmadCorrect' : [
        "{}/YoutubeChunkedAudio/ahmadCorrect_1_5.wav".format(data_dir_path),
        "{}/YoutubeChunkedAudio/ahmadCorrect_1_8.wav".format(data_dir_path),
        "{}/YoutubeChunkedAudio/ahmadCorrect_1_11.wav".format(data_dir_path),
    ]
}


fastpitch_ckpt_path = "/data/shehzeen/TextlessVCDemoCkpts/Epoch500_5797.ckpt".format(data_dir_path)
fastpitch_ckpt_path_finetuned = "/data/shehzeen/TextlessVCDemoCkpts/Epoch43_7464.ckpt".format(data_dir_path)

hifi_ckpt_path = "/data/shehzeen/TextlessVCDemoCkpts/HiFiLibriEpoch334.ckpt".format(data_dir_path)
hifi_ckpt_path_finetuned = "/data/shehzeen/TextlessVCDemoCkpts/Epoch7659_3494.ckpt".format(data_dir_path)

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

print("Loading Vocoder Model")
vocoder_finetuned = hifigan.HifiGanModel.load_from_checkpoint(hifi_ckpt_path_finetuned).to(device)
vocoder_finetuned.eval()

print("Loading FP Model")
fastpitch_model = fastpitch_ssl.FastPitchModel_SSL.load_from_checkpoint(fastpitch_ckpt_path, strict=False)
fastpitch_model = fastpitch_model.to(device)
fastpitch_model.eval()
fastpitch_model.non_trainable_models = {'vocoder': vocoder}
fpssl_sample_rate = fastpitch_model._cfg.sample_rate

print("Loading FP Model")
fastpitch_model_finetuned = fastpitch_ssl.FastPitchModel_SSL.load_from_checkpoint(fastpitch_ckpt_path_finetuned, strict=False)
fastpitch_model_finetuned = fastpitch_model_finetuned.to(device)
fastpitch_model_finetuned.eval()
fastpitch_model_finetuned.non_trainable_models = {'vocoder': vocoder_finetuned}

wav_featurizer_sv = WaveformFeaturizer(sample_rate=sv_sample_rate, int_values=False, augmentor=None)
wav_featurizer_fp = WaveformFeaturizer(sample_rate=fpssl_sample_rate, int_values=False, augmentor=None)

# print("Loading VAD Model")
# vad_model, vad_utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
#                               model='silero_vad',
#                               force_reload=True,
#                               onnx=False)

# (vad_get_speech_timestamps,
#  vad_save_audio,
#  vad_read_audio,
#  vad_VADIterator,
#  vad_collect_chunks) = vad_utils

# vad_model = vad_model.to("cpu")
def cleanup_temp_dir():
    # Remove files in temp dir older than one hour
    try:
        for f in os.listdir(temp_dir):
            if os.path.getmtime(os.path.join(temp_dir, f)) < time.time() - 3600:
                os.remove(os.path.join(temp_dir, f))
    except:
        pass


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

run_with_ngrok(app)  # Start ngrok when app is run
CORS(app)

@app.route('/test_connection')
def test_connection():
    return 'works!'


# Write a POST view to accept audio base64 and process it through the vc model
@app.route('/get_avatar', methods=['POST'])
def get_avatar():
    cleanup_temp_dir()
    audio_data_base64 = request.values.get('audio_data_base64')
    audio_data = base64.b64decode(audio_data_base64)
    speaker = request.values.get('speaker')
    session_number = random.randint(0, 1000000)
    temp_wav_fp = os.path.join(temp_dir, "temp_{}.wav".format(session_number))
    print("Temp wav fp", temp_wav_fp)
    temp_json_fp = os.path.join(temp_dir, "temp_{}.json".format(session_number))
    with open(temp_wav_fp, "wb") as f:
        f.write(audio_data)

    if speaker in avatar_paths:
        avatar_fp = avatar_paths[speaker]
    else:
        avatar_fp = avatar_paths["default"]
    
    command_string = '''pocketsphinx -phone_align yes single WAVFILE $text | jq '[.w[]|{word: (.t | ascii_upcase | sub("<S>"; "sil") | sub("<SIL>"; "sil") | sub("\\\(2\\\)"; "") | sub("\\\(3\\\)"; "") | sub("\\\(4\\\)"; "") | sub("\\\[SPEECH\\\]"; "SIL") | sub("\\\[NOISE\\\]"; "SIL")), phones: [.w[]|{ph: .t | sub("\\\+SPN\\\+"; "SIL") | sub("\\\+NSN\\\+"; "SIL"), bg: (.b*100)|floor, ed: (.b*100+.d*100)|floor}]}]' > JSONFILE'''
    command_string = command_string.replace("WAVFILE", temp_wav_fp)
    command_string = command_string.replace("JSONFILE", temp_json_fp)
    # print("Command", command_string)
    os.system(command_string)
    os.system("cd /home/shehzeen/one-shot-talking-face-colab/content/one-shot-talking-face; CUDA_VISIBLE_DEVICES=2 python test_script.py --img_path {} --audio_path {} --phoneme_path {} --save_dir {}".format(avatar_fp, temp_wav_fp, temp_json_fp, temp_dir) )

    output_video_name = os.path.basename(avatar_fp)[:-4] + "_" + os.path.basename(temp_wav_fp)[:-4] + ".mp4"
    print(os.path.join(temp_dir, output_video_name))

    video_base64 = base64.b64encode(open(os.path.join(temp_dir, output_video_name), "rb").read()).decode('utf-8')
    
    return json.dumps({'video_base64': video_base64})

    


@app.route('/convert_recordings', methods=['POST'])
def convert_recordings():
    total_wavs = int(request.values.get('total_wavs'))
    results = []
    _fastpitch_model = fastpitch_model
    for wav_no in range(total_wavs):
        with torch.no_grad():
            source_type = request.values.get('input_type_{}'.format(wav_no))
            if source_type == "recording":
                audio_data = request.files['audio_data_{}'.format(wav_no)]
            else:
                audio_data = request.files['custom_source_audio_{}'.format(wav_no)]
            
            source_wav = load_wav(audio_data, wav_featurizer_fp)
            audio_np = source_wav.cpu().numpy()

            speaker = request.values.get('speaker_{}'.format(wav_no))
            if speaker == "custom":
                target_audio_data = request.files['target_speaker_audio_{}'.format(wav_no)]
                target_speaker_embedding = get_speaker_embedding(
                    nemo_sv_model, wav_featurizer_sv, [target_audio_data], duration=None, device=device
                )
                _fastpitch_model = fastpitch_model
            else:
                target_speaker_embedding = speaker_embeddings[speaker]
                _fastpitch_model = fastpitch_model_finetuned

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

                wav_generated = _fastpitch_model.generate_wav(
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
            temp_buffer = io.BytesIO()
            sf.write(temp_buffer, wav_generated, 22050, format='WAV')
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
    # app.run(host='0.0.0.0', port=8000, ssl_context=('/home/bossbwtw/server.crt', '/home/bossbwtw/server.key'))
    app.run()