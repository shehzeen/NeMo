import os
import asyncio
import base64
import json
import websockets
import pyaudio
import numpy as np
import librosa
from pygame._sdl2 import get_num_audio_devices, get_audio_device_name #Get playback device names
from pygame import mixer #Playin
import soundfile as sf
import time
import shutil
import librosa

SEGSIZE_SECS = 3

FRAMES_PER_BUFFER = int(2048*SEGSIZE_SECS)
FORMAT = pyaudio.paFloat32
CHANNELS = 1
RATE = 22050
p = pyaudio.PyAudio()

# starts recording

# mixer.init(devicename='VB-Cable')

stream = p.open(
    format=FORMAT,
    channels=CHANNELS,
    rate=RATE,
    input=True,
    frames_per_buffer=FRAMES_PER_BUFFER,
)

# check size of stream


session_dir = 'session/'
if os.path.exists(session_dir):
    shutil.rmtree(session_dir)

os.makedirs(session_dir)
print(session_dir)

wav_total = np.array([])
wav_ctr = 0
for i in range(10000000):
    # print number of bytes in stream
    print("i", i)
    data = stream.read(FRAMES_PER_BUFFER)
    # save data in wav file
    # continue

    wav = np.fromstring(data, dtype=np.float32)
    wav_total = np.concatenate((wav_total, wav))
    if len(wav_total) == FRAMES_PER_BUFFER*10:
        wav_path = "{}/{}.wav".format(session_dir, wav_ctr)
        sf.write(wav_path, wav_total, RATE)
        wav_ctr += 1
        wav_total = np.array([])
    # wav_pitch_shifted = librosa.effects.pitch_shift(wav, RATE, n_steps=2)
    # sav wav_pitch_shifted
    # librosa.output.write_wav('test.wav', wav_pitch_shifted, RATE)
    
    # mixer.music.load('test.wav')
    # mixer.music.play()
    # pitch shift audio


    # flush stream
    # convert into numpy array
    # data = np.fromstring(data, dtype=np.int16)
    
