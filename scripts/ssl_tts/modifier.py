import base64
import json
import os
import shutil
import time

import librosa
import numpy as np
import pyaudio
import requests
import soundfile as sf
from pygame import mixer  # Playin
from pygame._sdl2 import get_audio_device_name, get_num_audio_devices  # Get playback device names

current_file_number = 0
session_dir = 'session/'
outdir = 'out/'
SEGSIZE_SECS = 3

mixer.init(devicename='VB-Cable')


if os.path.exists(outdir):
    shutil.rmtree(outdir)
os.makedirs(outdir)

wav_files = os.listdir(session_dir)
wav_files = [fname for fname in wav_files if fname.endswith('.wav')]
wav_numbers = [int(fname.split('.')[0]) for fname in wav_files]
wav_numbers.sort()

if len(wav_numbers) == 0:
    last_wavnum = 0
else:
    last_wavnum = wav_numbers[-1]

current_file_number = last_wavnum
print("last_wavname", last_wavnum)

for i in range(1000000):
    st = time.time()
    last_wavname = "{}.wav".format(last_wavnum)
    input_wav_path = "{}/{}".format(session_dir, last_wavname)
    if not os.path.exists(input_wav_path):
        print("no file, waiting")
        time.sleep(0.1)
        continue
    wav, sr = sf.read(input_wav_path, dtype='float32')

    if len(wav) != int(2048 * SEGSIZE_SECS * 10):
        continue

    assert sr == 22050
    print(wav.dtype)
    wav_base64 = base64.b64encode(wav)

    with open('speaker.txt') as f:
        speaker_name = f.read().strip()

    response = requests.post(
        "http://localhost:5000/convert_voice", data={"audio": wav_base64, "speaker": speaker_name}
    )
    response_data = json.loads(response.text)
    response_wav = base64.b64decode(response_data['audio_converted'])
    response_wav = np.frombuffer(response_wav, dtype=np.float32)

    shifted_path = "{}/shifted{}.wav".format(outdir, last_wavnum)
    sf.write(shifted_path, response_wav, 22050)

    print("before play", time.time() - st)
    print("Writing audio", shifted_path, response_wav.dtype)
    # stream.write(response_wav.tostring())

    wait_ctr = 0
    while mixer.music.get_busy():
        # current_audio_pos = mixer.music.get_pos()
        # if wait_ctr % 1000 == 0:
        #     print("waiting", current_audio_pos)
        # if current_audio_pos >= 1500:
        #     break
        time.sleep(0.0001)
        wait_ctr += 1
        continue

    if i == 0:
        print("Playing", shifted_path)
        mixer.music.load(shifted_path)
        mixer.music.play()
    else:
        # mixer.music.queue(shifted_path)
        print("Playing", shifted_path)
        mixer.music.unload()
        mixer.music.load(shifted_path)
        mixer.music.play()

    # mixer.music.load(shifted_path)
    # mixer.music.play()
    # mixer.music.set_endevent()
    # else:
    # mixer.music.queue(shifted_path)
    # print(i)
    print("final", time.time() - st)
    last_wavnum += 1

# def dummy_callback(in_data, frame_count, time_info, status):
#     global current_file_number
#     outfile_path = os.path.join(outdir, "shifted{}.wav".format(current_file_number))
#     while not os.path.exists(outfile_path):
#         # print("Waiting for", outfile_path)
#         time.sleep(0.01)
#     outdata,_ = sf.read(outfile_path, dtype='float32')
#     print(outdata.shape, outdata.dtype)
#     outdatabytes = outdata.tostring()
#     current_file_number += 1
#     return (outdatabytes, pyaudio.paContinue)

# p = pyaudio.PyAudio()

# device_index = 0
# for i in range(p.get_device_count()):
#     dev = p.get_device_info_by_index(i)
#     print(i, dev['name'], dev['maxInputChannels'], dev['maxOutputChannels'])
#     if dev['name'] == 'VB-Cable':
#         device_index = i
#         break

# stream = p.open(format=pyaudio.paFloat32,
#                 channels=1,
#                 rate=22050,
#                 output_device_index=device_index,
#                 output=True,
#                 stream_callback=dummy_callback
#                 )
