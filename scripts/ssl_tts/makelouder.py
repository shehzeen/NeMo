import librosa
import soundfile as sf
import numpy as np

audio_file = '/data/shehzeen/SingingVoices/tiken_spokenvoice.wav'
y, sr = librosa.load(audio_file, sr=22050)

normalized_y = librosa.util.normalize(y)
max_peak = np.amax(normalized_y)
print("max_peak: ", max_peak)
target_peak_value = 0.9  # Adjust this value as needed
amplification_factor = target_peak_value / max_peak
print("amplification_factor: ", amplification_factor)

amplified_y = y * amplification_factor


amplified_file = '/data/shehzeen/SingingVoices/tiken_spoken_voice_22050_louder.wav'
sf.write(amplified_file, normalized_y, sr)