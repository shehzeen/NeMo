import json
import os
import torch
from nemo.collections.tts.models import AudioCodecModel
import torchaudio
from torchaudio.transforms import Resample
import nemo.collections.asr as nemo_asr
import matplotlib.pyplot as plt

# manifest_path = "/datap/misc/speechllm_codecdatasets/manifests/LRHM_train_nemo_codec_bw_6.0_phoneme_tts.json"
manifest_path = "/datap/misc/speechllm_codecdatasets/manifests/5k_LRHM__train_nemo_codec_bw_6.0_phoneme_plus_sentencepiece_tts.json"
codec_model_path = "/Data/Checkpoints/rlang_codec/SpeechCodec.nemo"

resampler = Resample(orig_freq=22050, new_freq=16000).cuda()


def plot_similarity_historgram(records):
    similarities = [r['similarity'] for r in records]
    plt.hist(similarities, bins=20)
    plt.xlabel('Speaker Similarity')
    plt.ylabel('Count')
    plt.title('Speaker Similarity Histogram')
    # Save the figure
    plt.savefig('speaker_similarity_histogram.png')
    # Clear the plot
    plt.clf()
    plt.close()
    
def read_manifest(manifest_path):
    records = []
    with open(manifest_path, 'r') as f:
        all_lines = f.readlines()
        for line in all_lines:
            line = line.strip()
            records.append(json.loads(line))
    return records

def write_manifest(manifest_path, records):
    with open(manifest_path, 'w') as f:
        file_str = ""
        for record in records:
            file_str += json.dumps(record) + "\n"
        file_str = file_str.strip()
        f.write(file_str)
        print("Wrote {} records to: {}".format(len(records), manifest_path))



def decode_codes_and_find_similarity(codec_model, codec_fps):
    """
    Decodes the codec_fps (codec filepaths, typically context and answer)
    and finds the similarity between the embeddings
    """
    with torch.no_grad():
        max_codec_len = 0
        codec_list = []
        codec_lens = []
        for codec_fp in codec_fps:
            codec = torch.load(codec_fp)
            codec_len = torch.Tensor([codec.shape[1]]).long().cuda()
            codec_lens.append(codec_len)
            if codec.shape[1] > max_codec_len:
                max_codec_len = codec.shape[1]
            codec_list.append(codec)
        
        codec_lens = torch.stack(codec_lens).long().cuda()
        codecs = torch.zeros(len(codec_list), codec_list[0].shape[0], max_codec_len).cuda()
        for i, codec in enumerate(codec_list):
            codecs[i, :, :codec.shape[1]] = codec
        codecs = codecs.long()
        codec_decoded_audios, _ = codec_model.decode(tokens=codecs, tokens_len=codec_lens[:,0])

        audios_16 = []
        audio_16_lens = []
        max_16_len = 0
        for idx in range(len(codec_decoded_audios)):
            codec_decoded_audio = codec_decoded_audios[idx]
            codec_decoded_audio = codec_decoded_audio[:codec_lens[idx][0].item() * int(codec_model_downsampling_factor)]
            
            # Resample from 22050 to 16000
            codec_decoded_audio_16 = resampler(codec_decoded_audio)
            audios_16.append(codec_decoded_audio_16)
            audio_16_lens.append(codec_decoded_audio_16.shape[0])
            if codec_decoded_audio_16.shape[0] > max_16_len:
                max_16_len = codec_decoded_audio_16.shape[0]

            # codec_decoded_audio_path = os.path.join("/datap/misc/decodingtesting/testing_{}.wav".format(idx))
            # torchaudio.save(codec_decoded_audio_path, codec_decoded_audio[None].cpu(), codec_model_sample_rate)

            # codec_decoded_audio_path = os.path.join("/datap/misc/decodingtesting/testing_{}_16k.wav".format(idx))
            # torchaudio.save(codec_decoded_audio_path, codec_decoded_audio_16[None].cpu(), 16000)

        audio_16_batch = torch.zeros(len(audios_16), max_16_len).cuda()
        for idx in range(len(audios_16)):
            audio_16_batch[idx, :audio_16_lens[idx]] = audios_16[idx]
        
        audio_signal_len_16 = torch.Tensor(audio_16_lens).long().cuda()
        # import ipdb; ipdb.set_trace()
        nemo_sv_model.eval()
        # nemo_sv_model.freeze()
        _, embs = nemo_sv_model.forward(input_signal=audio_16_batch, input_signal_length=audio_signal_len_16)
        # Find cosine similarity between embs
        similarities = []
        for i in range(embs.shape[0]-1):
            emb1 = embs[i]
            emb2 = embs[-1]
            similarity = torch.nn.functional.cosine_similarity(emb1, emb2, dim=0).item()
            similarities.append(similarity)

        return sum(similarities) / len(similarities)


records = read_manifest(manifest_path)

codec_model = AudioCodecModel.restore_from(codec_model_path)
codec_model.to('cuda')
codec_model.eval()
codec_model_sample_rate = 22050
codec_model_downsampling_factor = 256.0

nemo_sv_model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(model_name='titanet_large')

with_similarity_records = []
low_similarity_records = []
high_similarity_records = []
for ridx, record in enumerate(records):
    print("{} out of {}".format(ridx, len(records)))
    context_fps = record["context"].split(";")
    answer_fp = record["answer"]
    try:
        similarity = decode_codes_and_find_similarity(codec_model, context_fps + [answer_fp])
    except:
        print("Error in record: ", record)
        continue

    record["similarity"] = round(similarity, 4)
    if similarity < 0.35:
        low_similarity_records.append(record)
    elif similarity > 0.7:
        high_similarity_records.append(record)
    
    with_similarity_records.append(record)
    print(ridx, record["similarity"])
    
    if (ridx+1) % 100 == 0 or ridx == len(records) - 1:
        updated_manifest_path = manifest_path.replace(".json", "_withsimilarity_inprogress2.json")
        write_manifest(updated_manifest_path, with_similarity_records)
        plot_similarity_historgram(with_similarity_records)

        low_similarity_manifest_path = manifest_path.replace(".json", "_low_similarity2.json")
        write_manifest(low_similarity_manifest_path, low_similarity_records)

        high_similarity_manifest_path = manifest_path.replace(".json", "_high_similarity2.json")
        write_manifest(high_similarity_manifest_path, high_similarity_records)