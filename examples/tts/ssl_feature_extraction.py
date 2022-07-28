import nemo.collections.asr as nemo_asr
import argparse
from nemo.collections.tts.torch import data as tts_data
from torch.utils.data import DataLoader
import torch
import json
import os
import time
from nemo.collections.tts.torch.helpers import get_base_dir
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest_path', type=str, default="/home/pneekhara/Datasets/78419/Hi_Fi_TTS_v_0_backup/92_manifest_clean_train.json")
    parser.add_argument('--out_dir', type=str, default='/home/pneekhara/Datasets/temp/ssl_features')
    args = parser.parse_args()

    # ssl_model = nemo_asr.models.ssl_models.SpeechEncDecSelfSupervisedModel.from_pretrained(model_name='ssl_en_conformer_large')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = tts_data.VocoderDataset(manifest_filepath=args.manifest_path, sample_rate=16000)
    # dataset = tts_data.TTSDataset(manifest_filepath=args.manifest_path, sample_rate=16000, text_tokenizer=EnglishCharsTokenizer())
    dataloader = DataLoader(dataset, collate_fn=dataset.collate_fn, shuffle=False, batch_size=64)

    ssl_model = nemo_asr.models.ssl_models.SpeechEncDecSelfSupervisedModel.from_pretrained(model_name='ssl_en_conformer_large')
    ssl_model.to(device)
    ssl_model.eval()

    manifest_data = []
    audio_paths = []
    with open(args.manifest_path) as f:
        for record in f.readlines():
            if len(record) > 0:
                record_data = json.loads(record)
                audio_paths.append(record_data['audio_filepath'])
                manifest_data.append(record_data)
    
    audio_base_dir = get_base_dir(audio_paths)

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    
    manifest_idx = 0
    start_time = time.time()
    for bidx, batch in enumerate(dataloader):
        audio_signal, audio_signal_length = batch
        audio_signal = audio_signal.to(device)
        audio_signal_length = audio_signal_length.to(device)
        
        with torch.no_grad():
            processed_signal, processed_signal_length = ssl_model.preprocessor(
                input_signal=audio_signal, length=audio_signal_length,
            )

            encoded, encoded_len = ssl_model.encoder(audio_signal=processed_signal, length=processed_signal_length)
        
        for _i in range(len(audio_signal_length)):
            assert manifest_data[manifest_idx]['duration'] == audio_signal_length[_i].item()/16000.
            
            item_encoding = encoded[_i]
            item_encoding_length = encoded_len[_i].item()
            
            item_encoding = item_encoding[:,:item_encoding_length]
            
            item_encoding = item_encoding.cpu().float()

            rel_audio_path_as_text_id = Path(manifest_data[manifest_idx]['audio_filepath']).relative_to(audio_base_dir).with_suffix("")
            rel_audio_path_as_text_id = str(rel_audio_path_as_text_id).replace("/", "_")
            out_path = os.path.join(args.out_dir, "{}.pt".format(rel_audio_path_as_text_id))
            out_path_parent_dir = Path(out_path).parent.absolute()
            if not os.path.exists(out_path_parent_dir):
                os.makedirs(out_path_parent_dir)
            torch.save(item_encoding, out_path)
            manifest_idx += 1

        print("Completed {} of {} in {:.2f} seconds".format(bidx+1, len(dataloader), time.time() - start_time))

if __name__ == '__main__':
    main()