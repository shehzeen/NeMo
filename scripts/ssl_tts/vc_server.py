import base64
import json

# import wav2lip_infer
import time

import librosa
import numpy as np
import soundfile as sf
import torch
from flask import Flask, request
from flask_cors import CORS

import nemo.collections.asr as nemo_asr
from nemo.collections.asr.models import label_models
from nemo.collections.asr.parts.preprocessing.features import WaveformFeaturizer
from nemo.collections.tts.models import fastpitch_ssl, hifigan

# ssl_model_ckpt_path = "/data/shehzeen/SSLTTS/PretrainingExperiments/AugLossAlpha100/Conformer-SSL/2023-01-24_00-42-05/checkpoints/Epoch68.ckpt"
ssl_model_ckpt_path = "/data/shehzeen/SSLTTS/PretrainingExperiments/MultiLing256/Conformer-SSL/2023-01-29_21-22-11/checkpoints/Epoch39.ckpt"
# hifi_ckpt_path = "/data/shehzeen/SSLTTS/HiFiLibriEpoch334.ckpt"
# fastpitch_ckpt_path = "/data/shehzeen/SSLTTS/TextlessFastPitchExperiments/AugmentedTraining/2023-01-26_14-00-48/checkpoints/Epoch89.ckpt"
# target_audio_paths = ["/data/shehzeen/SSLTTS/EVALDATA/source_2.wav"]

with open('/home/shehzeen/SimSwap/faceswap_status.txt', 'w') as f:
    f.write("not ready")

target_audio_paths = {
    'obama': [
        "/data/shehzeen/SSLTTS/CelebrityData/YoutubeChunkedAudio/obama_2_335.wav",
        "/data/shehzeen/SSLTTS/CelebrityData/YoutubeChunkedAudio/obama_1_335.wav",
        "/data/shehzeen/SSLTTS/CelebrityData/YoutubeChunkedAudio/obama_1_336.wav",
        "/data/shehzeen/SSLTTS/CelebrityData/YoutubeChunkedAudio/obama_1_337.wav",
    ],
    'modi': [
        "/data/shehzeen/SSLTTS/CelebrityData/YoutubeChunkedAudio/modi_1_334.wav",
        "/data/shehzeen/SSLTTS/CelebrityData/YoutubeChunkedAudio/modi_1_335.wav",
        "/data/shehzeen/SSLTTS/CelebrityData/YoutubeChunkedAudio/modi_1_336.wav",
        "/data/shehzeen/SSLTTS/CelebrityData/YoutubeChunkedAudio/modi_1_337.wav",
    ],
    'ravish': [
        "/data/shehzeen/SSLTTS/CelebrityData/YoutubeChunkedAudio/ravish_2_141.wav",
        "/data/shehzeen/SSLTTS/CelebrityData/YoutubeChunkedAudio/ravish_2_142.wav",
        "/data/shehzeen/SSLTTS/CelebrityData/YoutubeChunkedAudio/ravish_2_143.wav",
        "/data/shehzeen/SSLTTS/CelebrityData/YoutubeChunkedAudio/ravish_2_144.wav",
    ],
    'lex': [
        "/data/shehzeen/SSLTTS/CelebrityData/YoutubeChunkedAudio/lex_1_290.wav",
        "/data/shehzeen/SSLTTS/CelebrityData/YoutubeChunkedAudio/lex_1_291.wav",
        "/data/shehzeen/SSLTTS/CelebrityData/YoutubeChunkedAudio/lex_1_292.wav",
        "/data/shehzeen/SSLTTS/CelebrityData/YoutubeChunkedAudio/lex_1_293.wav",
    ],
    'oprah': [
        "/data/shehzeen/SSLTTS/CelebrityData/YoutubeChunkedAudio/oprah_2_151.wav",
        "/data/shehzeen/SSLTTS/CelebrityData/YoutubeChunkedAudio/oprah_2_152.wav",
        "/data/shehzeen/SSLTTS/CelebrityData/YoutubeChunkedAudio/oprah_2_153.wav",
        "/data/shehzeen/SSLTTS/CelebrityData/YoutubeChunkedAudio/oprah_2_154.wav",
    ],
    'emma': [
        "/data/shehzeen/SSLTTS/CelebrityData/YoutubeChunkedAudio/emma_1_2.wav",
        "/data/shehzeen/SSLTTS/CelebrityData/YoutubeChunkedAudio/emma_1_5.wav",
        "/data/shehzeen/SSLTTS/CelebrityData/YoutubeChunkedAudio/emma_1_8.wav",
    ],
    'priyanka': [
        "/data/shehzeen/SSLTTS/CelebrityData/YoutubeChunkedAudio/priyanka_1_2.wav",
        "/data/shehzeen/SSLTTS/CelebrityData/YoutubeChunkedAudio/priyanka_1_8.wav",
        "/data/shehzeen/SSLTTS/CelebrityData/YoutubeChunkedAudio/priyanka_1_11.wav",
    ],
    'miley': [
        "/data/shehzeen/SSLTTS/CelebrityData/YoutubeChunkedAudio/miley_1_2.wav",
        "/data/shehzeen/SSLTTS/CelebrityData/YoutubeChunkedAudio/miley_1_5.wav",
        "/data/shehzeen/SSLTTS/CelebrityData/YoutubeChunkedAudio/miley_1_11.wav",
    ],
    'aubrey': [
        "/data/shehzeen/SSLTTS/CelebrityData/YoutubeChunkedAudio/aubrey_1_5.wav",
        "/data/shehzeen/SSLTTS/CelebrityData/YoutubeChunkedAudio/aubrey_1_8.wav",
        "/data/shehzeen/SSLTTS/CelebrityData/YoutubeChunkedAudio/aubrey_1_11.wav",
    ],
    'sundar': [
        "/data/shehzeen/SSLTTS/CelebrityData/YoutubeChunkedAudio/sundar_1_5.wav",
        "/data/shehzeen/SSLTTS/CelebrityData/YoutubeChunkedAudio/sundar_1_8.wav",
        "/data/shehzeen/SSLTTS/CelebrityData/YoutubeChunkedAudio/sundar_1_11.wav",
    ],
    'ahmadCorrect': [
        "/data/shehzeen/SSLTTS/CelebrityData/YoutubeChunkedAudio/ahmadCorrect_1_5.wav",
        "/data/shehzeen/SSLTTS/CelebrityData/YoutubeChunkedAudio/ahmadCorrect_1_8.wav",
        "/data/shehzeen/SSLTTS/CelebrityData/YoutubeChunkedAudio/ahmadCorrect_1_11.wav",
    ],
}

# fastpitch_ckpt_path = "/data/shehzeen/SSLTTS/CelebrityFastPitch/CelebrityLexOprah/2023-01-29_16-31-54/checkpoints/Epoch167.ckpt"
# fastpitch_ckpt_path = "/data/shehzeen/SSLTTS/CelebrityFastPitch/CelebrityFemailSpeakers/2023-01-30_14-41-11/checkpoints/Epoch300.ckpt"
fastpitch_ckpt_path = (
    "/data/shehzeen/SSLTTS/CelebrityFastPitch/CelebrityAhmad/2023-02-05_17-13-10/checkpoints/Epoch31.ckpt"
)
# hifi_ckpt_path = "/data/shehzeen/SSLTTS/HifiGANObama/HifiGan/2023-01-28_19-02-46/checkpoints/Epoch909.ckpt"
# hifi_ckpt_path = "/data/shehzeen/SSLTTS/HifiGANCelebrity/HifiGan/2023-01-29_16-23-01/checkpoints/Epoch69.ckpt"
# hifi_ckpt_path = "/data/shehzeen/SSLTTS/HifiGANOnSynth/HifiGan/2023-01-29_21-35-32/checkpoints/Epoch799.ckpt"
# hifi_ckpt_path = "/data/shehzeen/SSLTTS/HifiGANOnSynthNewCelebs/HifiGan/2023-01-30_18-23-17/checkpoints/Epoch1019.ckpt"
hifi_ckpt_path = "/data/shehzeen/SSLTTS/HifiGANOnCelebAhmad/HifiGan/2023-02-05_17-57-10/checkpoints/Epoch479.ckpt"
# hifi_ckpt_path = "/data/shehzeen/SSLTTS/HiFiLibriEpoch334.ckpt"
device = "cuda:0" if torch.cuda.is_available() else "cpu"

ssl_model = nemo_asr.models.ssl_models.SpeechEncDecSelfSupervisedModel.load_from_checkpoint(ssl_model_ckpt_path)
ssl_model.eval()
ssl_model.to(device)

nemo_sv_model = label_models.EncDecSpeakerLabelModel.from_pretrained("titanet_large")
nemo_sv_model = nemo_sv_model.to(device)
nemo_sv_model.eval()
sv_sample_rate = nemo_sv_model._cfg.preprocessor.sample_rate

vocoder = hifigan.HifiGanModel.load_from_checkpoint(hifi_ckpt_path).to(device)
vocoder.eval()

fastpitch_model = fastpitch_ssl.FastPitchModel_SSL.load_from_checkpoint(fastpitch_ckpt_path, strict=False)
fastpitch_model = fastpitch_model.to(device)
fastpitch_model.eval()
fastpitch_model.non_trainable_models = {'vocoder': vocoder}
fpssl_sample_rate = fastpitch_model._cfg.sample_rate

wav_featurizer_sv = WaveformFeaturizer(sample_rate=sv_sample_rate, int_values=False, augmentor=None)
wav_featurizer_fp = WaveformFeaturizer(sample_rate=fpssl_sample_rate, int_values=False, augmentor=None)


vad_model, vad_utils = torch.hub.load(
    repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=True, onnx=False
)

(vad_get_speech_timestamps, vad_save_audio, vad_read_audio, vad_VADIterator, vad_collect_chunks) = vad_utils

vad_model = vad_model.to("cpu")

facial_image_paths = {
    'obama': "/home/shehzeen/first-order-model/obama.jpg",
    'ahmadCorrect': "/home/shehzeen/ahmad.png",
    'sundar': "/home/shehzeen/sundar.png",
    'modi': "/home/shehzeen/modi.png",
    'emma': "/home/shehzeen/emma.png",
    'priyanka': "/home/shehzeen/priyanka.png",
}


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

    _, speaker_embeddings = nemo_sv_model(input_signal=signal_batch, input_signal_length=signal_length_batch)
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


def get_output_video_fromsimswap():
    status = "not ready"
    while status != "output ready":
        time.sleep(0.1)
        with open("/home/shehzeen/SimSwap/faceswap_status.txt") as f:
            status = f.read()
        print("waiting for output video")

    output_video_path = "/home/shehzeen/SimSwap/demo_file/result.mp4"
    with open(output_video_path, 'rb') as f:
        video_bytes = f.read()
        video_base64 = base64.b64encode(video_bytes).decode('utf-8')
        return video_base64


# Write a POST view to accept audio base64 and process it through the vc model
@app.route('/convert_voice', methods=['POST'])
def convert_voice():
    # get base64 audio from request
    # convert base64 to wav
    # get speaker embedding
    # run vc model

    audio_base64 = request.values.get('audio')
    inputvideo_base64 = request.values.get('video')
    speaker = request.values.get('speaker')

    input_video_decoded = base64.b64decode(inputvideo_base64)

    video_path = "/home/shehzeen/SimSwap/demo_file/source.mp4"
    with open(video_path, 'wb') as f:
        f.write(input_video_decoded)

    if speaker in facial_image_paths:
        with open("/home/shehzeen/SimSwap/demo_files/source_image_fp.txt", 'w') as f:
            f.write(facial_image_paths[speaker])

    with open('/home/shehzeen/SimSwap/faceswap_status.txt', 'w') as f:
        f.write("input ready")

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
        # no voiced frames, return the same audio
        silence = audio_np * 0
        # video_path = wav2lip_infer.get_lipsynced_video(audio_np_16000, speaker, silence=True)
        # video_base64 = base64.b64encode(open(video_path, 'rb').read()).decode('utf-8')
        # video_base64 = inputvideo_base64
        video_base64 = get_output_video_fromsimswap()
        return json.dumps({'audio_converted': base64.b64encode(silence).decode('utf-8'), 'video': video_base64})

    st = time.time()
    with torch.no_grad():
        audio_np = audio_np[:-1]
        audio_signal = torch.from_numpy(audio_np).to(device)[None]

        audio_signal_length = torch.tensor([audio_signal.shape[1]]).to(device)

        processed_signal, processed_signal_length = ssl_model.preprocessor(
            input_signal=audio_signal, length=audio_signal_length,
        )
        batch_content_embedding, batch_encoded_len = ssl_model.encoder(
            audio_signal=processed_signal, length=processed_signal_length
        )
        final_content_embedding = batch_content_embedding[0, :, : batch_encoded_len[0]]
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

    video_start_time = time.time()
    # video_path = wav2lip_infer.get_lipsynced_video(audio_np_16000, speaker)
    video_time = time.time() - video_start_time
    print("video time ", video_time)
    # load video path and convert to base64

    # video_base64 = base64.b64encode(open(video_path, 'rb').read()).decode('utf-8')
    # video_base64 = inputvideo_base64
    video_base64 = get_output_video_fromsimswap()

    return json.dumps({'audio_converted': base64.b64encode(wav_generated).decode('utf-8'), 'video': video_base64,})


if __name__ == '__main__':
    app.run()
