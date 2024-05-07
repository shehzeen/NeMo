from nemo.collections.tts.models import AudioCodecModel
import torch
import os
import torchaudio
import json

def read_manifest(manifest_path):
    records = []
    with open(manifest_path, 'r') as f:
        all_lines = f.readlines()
        for line in all_lines:
            line = line.strip()
            records.append(json.loads(line))
    return records

manifest_path = "/datap/misc/speechllm_codecdatasets/manifests/5k_LRHM__train_nemo_codec_bw_6.0_phoneme_plus_sentencepiece_tts_low_similarity2.json"
records = read_manifest(manifest_path)

codec_model = AudioCodecModel.restore_from("/Data/Checkpoints/rlang_codec/SpeechCodec.nemo")
codec_model.to('cuda')
codec_model.eval()
codec_model_sample_rate = 22050
codec_model_downsampling_factor = 256.0


# codec_paths = [
#     "/datap/misc/speechllm_codecdatasets/codecs/combined_contexts/en_HiFiTTS_6097_clean_0.pt"
# ]

with torch.no_grad():
    for ridx, record in enumerate(records):
        codec_paths = [ record["context"], record["answer"] ]
        for cidx in range(len(codec_paths)):
            codec_path = codec_paths[cidx]
            codec = torch.load(codec_path)
            codec = codec.to('cuda')
            codec = codec.unsqueeze(0)

            codec_lens = torch.Tensor([codec.shape[2]]).long().cuda()
            codec_decoded_audios, _ = codec_model.decode(tokens=codec.long(), tokens_len=codec_lens)

            codec_decoded_audio = codec_decoded_audios[0]

            codec_decoded_audio_path = os.path.join("/Data/CodecListen/listening_{}={}.wav".format(ridx, cidx))
            torchaudio.save(codec_decoded_audio_path, codec_decoded_audio[None].cpu(), codec_model_sample_rate)
            print("Saved to", codec_decoded_audio_path)
            print("Similarity", record["similarity"])
        
        if ridx > 100:
            break