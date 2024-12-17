from nemo.collections.tts.models import T5TTS_Model
from nemo.collections.tts.data.text_to_speech_dataset import T5TTSDataset
from omegaconf.omegaconf import OmegaConf, open_dict
import os
import glob
import torch
import soundfile as sf
import evaluate_generated_audio
import evaluate_multilingual_generated_audio
import json
import argparse

dataset_meta_info = {
    'vctk': {
        'manifest_path' : '/home/shehzeenh/Code/NewT5TTS/manifests/smallvctk__phoneme__nemo_audio_21fps_8codebooks_2kcodes_v2bWithWavLM_simplet5_withcontextaudiopaths.json',
        'audio_dir' : '/Data/VCTK-Corpus',
        'feature_dir' : '/Data/VCTK-Corpus',
    },
    'riva_challenging': {
        'manifest_path' : '/home/shehzeenh/Code/NewT5TTS/manifests/challengingLindyRodney__phoneme__nemo_audio_21fps_8codebooks_2kcodes_v2bWithWavLM_simplet5_withContextAudioPaths.json',
        'audio_dir' : '/Data/RivaData/riva',
        'feature_dir' : '/Data/RivaData/riva',
    },
    'libri_val': {
        'manifest_path' : '/home/shehzeenh/Code/NewT5TTS/manifests/libri360_val.json',
        'audio_dir' : '/Data/LibriTTS',
        'feature_dir' : '/Data/LibriTTS',
    },
    'cissy_audiocontext': {
        'manifest_path' : '/home/shehzeenh/Code/NewT5TTS/cissymanifests/combined_val.json',
        'audio_dir' : '/Data/Ethovox/CissyJones',
        'feature_dir' : '/Data/Ethovox/CissyJones',
    },
    'cissy_textcontext': {
        'manifest_path' : '/home/shehzeenh/Code/NewT5TTS/cissymanifests/combined_val_textcontext.json',
        'audio_dir' : '/Data/Ethovox/CissyJones',
        'feature_dir' : '/Data/Ethovox/CissyJones',
    },
    'spanish_cml': {
        'manifest_path' : '/Data/CML/manifests_with_codecs/cml_tts_dataset_spanish_v0.1/dev_subset_withAudioCodes_codec21Khz_no_eliz_filtered.json',
        'audio_dir': '/Data/CML/cml_tts_dataset_spanish_v0.1',
        'feature_dir': '/Data/CML/cml_tts_dataset_spanish_v0.1',
        'tokenizer_names': ['spanish_phoneme'],
        'whisper_language': 'es'
    },
    'german_cml': {
        'manifest_path' : '/Data/CML/manifests_with_codecs/cml_tts_dataset_german_v0.1/dev_subset_withAudioCodes_codec21Khz_no_eliz_filtered.json',
        'audio_dir': '/Data/CML/cml_tts_dataset_german_v0.1',
        'feature_dir': '/Data/CML/cml_tts_dataset_german_v0.1',
        'tokenizer_names': ['german_phoneme'],
        'whisper_language': 'de'
    },
    'french_cml': {
        'manifest_path' : '/Data/CML/manifests_with_codecs/cml_tts_dataset_french_v0.1/dev_subset_withAudioCodes_codec21Khz_no_eliz_filtered.json',
        'audio_dir': '/Data/CML/cml_tts_dataset_french_v0.1',
        'feature_dir': '/Data/CML/cml_tts_dataset_french_v0.1',
        'tokenizer_names': ['multilingual_sentencepiece'],
        'whisper_language': 'fr'
    },
    'italian_cml': {
        'manifest_path' : '/Data/CML/manifests_with_codecs/cml_tts_dataset_italian_v0.1/dev_subset_withAudioCodes_codec21Khz_no_eliz_filtered.json',
        'audio_dir': '/Data/CML/cml_tts_dataset_italian_v0.1',
        'feature_dir': '/Data/CML/cml_tts_dataset_italian_v0.1',
        'tokenizer_names': ['multilingual_sentencepiece'],
        'whisper_language': 'it'
    },
    'portuguese_cml': {
        'manifest_path' : '/Data/CML/manifests_with_codecs/cml_tts_dataset_portuguese_v0.1/dev_subset_withAudioCodes_codec21Khz_no_eliz_filtered.json',
        'audio_dir': '/Data/CML/cml_tts_dataset_portuguese_v0.1',
        'feature_dir': '/Data/CML/cml_tts_dataset_portuguese_v0.1',
        'tokenizer_names': ['multilingual_sentencepiece'],
        'whisper_language': 'pt'
    },
    'polish_cml': {
        'manifest_path' : '/Data/CML/manifests_with_codecs/cml_tts_dataset_polish_v0.1/dev_subset_withAudioCodes_codec21Khz_no_eliz_filtered.json',
        'audio_dir': '/Data/CML/cml_tts_dataset_polish_v0.1',
        'feature_dir': '/Data/CML/cml_tts_dataset_polish_v0.1',
        'tokenizer_names': ['multilingual_sentencepiece'],
        'whisper_language': 'pl'
    },
    'dutch_cml': {
        'manifest_path' : '/Data/CML/manifests_with_codecs/cml_tts_dataset_dutch_v0.1/dev_subset_withAudioCodes_codec21Khz_no_eliz_filtered.json',
        'audio_dir': '/Data/CML/cml_tts_dataset_dutch_v0.1',
        'feature_dir': '/Data/CML/cml_tts_dataset_dutch_v0.1',
        'tokenizer_names': ['multilingual_sentencepiece'],
        'whisper_language': 'nl'
    }

}


def run_inference(hparams_file, checkpoint_file, datasets, out_dir, temperature, topk, codecmodel_path, use_cfg, cfg_scale, batch_size):
    # import ipdb; ipdb.set_trace()
    model_cfg = OmegaConf.load(hparams_file).cfg

    with open_dict(model_cfg):
        model_cfg.codecmodel_path = codecmodel_path
        if hasattr(model_cfg, 'text_tokenizer'):
            # Backward compatibility for models trained with absolute paths in text_tokenizer
            model_cfg.text_tokenizer.g2p.phoneme_dict = "scripts/tts_dataset_files/ipa_cmudict-0.7b_nv23.01.txt"
            model_cfg.text_tokenizer.g2p.heteronyms = "scripts/tts_dataset_files/heteronyms-052722"
            model_cfg.text_tokenizer.g2p.phoneme_probability = 1.0
        model_cfg.train_ds = None
        model_cfg.validation_ds = None


    model = T5TTS_Model(cfg=model_cfg)
    if model_cfg.t5_decoder.pos_emb == "learnable":
        if (model_cfg.t5_decoder.use_flash_self_attention) is False and (model_cfg.t5_decoder.use_flash_self_attention is False):
            model.use_kv_cache_for_inference = True

    # Load weights from checkpoint file
    print("Loading weights from checkpoint")
    ckpt = torch.load(checkpoint_file)
    model.load_state_dict(ckpt['state_dict'])
    print("Loaded weights.")
    model.cuda()
    model.eval()
    # import ipdb; ipdb.set_trace()

    checkpoint_name = checkpoint_file.split("/")[-1].split(".ckpt")[0]
    checkpoint_name = "{}_Temp{}_Topk{}_Cfg_{}_{}".format(checkpoint_name, temperature, topk, use_cfg, cfg_scale)
    
    for dataset in datasets:
        eval_dir = os.path.join(out_dir, "{}_{}".format(checkpoint_name, dataset))
        audio_dir = os.path.join(eval_dir, "audio")
        os.makedirs(audio_dir, exist_ok=True) 
        dataset_meta = {dataset: dataset_meta_info[dataset]}
        if 'whisper_language' in dataset_meta[dataset]:
            del dataset_meta[dataset]['whisper_language']
        test_dataset = T5TTSDataset(
            dataset_meta=dataset_meta,
            sample_rate=model_cfg.sample_rate,
            min_duration=0.5,
            max_duration=20,
            codec_model_downsample_factor=model_cfg.codec_model_downsample_factor,
            bos_id=model.bos_id,
            eos_id=model.eos_id,
            context_audio_bos_id=model.context_audio_bos_id,
            context_audio_eos_id=model.context_audio_eos_id,
            audio_bos_id=model.audio_bos_id,
            audio_eos_id=model.audio_eos_id,
            num_audio_codebooks=model_cfg.num_audio_codebooks,
            prior_scaling_factor=None,
            load_cached_codes_if_available=False,
            dataset_type='test',
            tokenizer_config=None,
            load_16khz_audio=model.model_type == 'single_encoder_sv_tts',
            use_text_conditioning_tokenizer=model.use_text_conditioning_encoder,
            pad_context_text_to_max_duration=model.pad_context_text_to_max_duration,
            context_duration_min=model.cfg.get('context_duration_min', 5.0),
            context_duration_max=model.cfg.get('context_duration_max', 5.0),
        )
        test_dataset.text_tokenizer, test_dataset.text_conditioning_tokenizer = model._setup_tokenizers(model.cfg, mode='test')

        test_data_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=batch_size,
            collate_fn=test_dataset.collate_fn,
            num_workers=2,
            shuffle=False
        )

        item_idx = 0
        for bidx, batch in enumerate(test_data_loader):
            print("Processing batch {} out of {} of dataset {}".format(bidx, len(test_data_loader), dataset))
            batch_cuda ={}
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch_cuda[key] = batch[key].cuda()
                else:
                    batch_cuda[key] = batch[key]
            
            predicted_audio, predicted_audio_lens, _, _ = model.infer_batch(batch_cuda, max_decoder_steps=500, temperature=temperature, topk=topk, use_cfg=use_cfg, cfg_scale=cfg_scale)
            for idx in range(predicted_audio.size(0)):
                predicted_audio_np = predicted_audio[idx].float().detach().cpu().numpy()
                predicted_audio_np = predicted_audio_np[:predicted_audio_lens[idx]]
                audio_path = os.path.join(audio_dir, f"predicted_audio_{item_idx}.wav")
                sf.write(audio_path, predicted_audio_np, model.cfg.sample_rate)
                item_idx += 1
        
        metrics = evaluate_multilingual_generated_audio.evaluate(
            dataset_meta_info[dataset]['manifest_path'],
            dataset_meta_info[dataset]['audio_dir'],
            audio_dir,
            language=dataset_meta_info[dataset].get('whisper_language', 'en')
        )

        with open(os.path.join(eval_dir, f"{dataset}_metrics.json"), "w") as f:
            json.dump(metrics, f)

        all_experiment_csv = os.path.join(out_dir, "all_experiment_metrics.csv")
        if not os.path.exists(all_experiment_csv):
            with open(all_experiment_csv, "w") as f:
                f.write("checkpoint_name,dataset,cer_filewise_avg,wer_filewise_avg,cer_cumulative,wer_cumulative,ssim_pred_gt_avg,ssim_pred_context_avg,ssim_gt_context_avg,ssim_pred_gt_avg_alternate,ssim_pred_context_avg_alternate,ssim_gt_context_avg_alternate\n")
        with open(all_experiment_csv, "a") as f:
            f.write(f"{checkpoint_name},{dataset},{metrics['cer_filewise_avg']},{metrics['wer_filewise_avg']},{metrics['cer_cumulative']},{metrics['wer_cumulative']},{metrics['ssim_pred_gt_avg']},{metrics['ssim_pred_context_avg']},{metrics['ssim_gt_context_avg']},{metrics['ssim_pred_gt_avg_alternate']},{metrics['ssim_pred_context_avg_alternate']},{metrics['ssim_gt_context_avg_alternate']}\n")
            print(f"Wrote metrics for {checkpoint_name} and {dataset} to {all_experiment_csv}")



def main():
    parser = argparse.ArgumentParser(description='Experiment Evaluation')
    parser.add_argument('--hparams_file', type=str, default="/Data/Experiments/CML_DecoderContext_FilteredData_WithRivaSpanish/T5TTS/0/hparams.yaml")
    parser.add_argument('--checkpoint_file', type=str, default="/Data/Experiments/CML_DecoderContext_FilteredData_WithRivaSpanish/T5TTS/0/checkpoints/multilingual_cml_epoch37.ckpt")
    parser.add_argument('--codecmodel_path', type=str, default="/Data/Checkpoints/AudioCodec_21Hz_no_eliz.nemo")
    parser.add_argument('--datasets', type=str, default="german_cml,french_cml,italian_cml,portuguese_cml,polish_cml,dutch_cml")
    parser.add_argument('--base_exp_dir', type=str, default=None)
    parser.add_argument('--draco_exp_dir', type=str, default=None)
    parser.add_argument('--server_address', type=str, default=None)
    parser.add_argument('--exp_names', type=str, default=None)
    parser.add_argument('--local_ckpt_dir', type=str, default=None)
    parser.add_argument('--out_dir', type=str, default="/Data/Experiments/CML_Inference2")
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--use_cfg', action='store_true')
    parser.add_argument('--cfg_scale', type=float, default=1.0)
    parser.add_argument('--topk', type=int, default=80)
    parser.add_argument('--batch_size', type=int, default=16)
    args = parser.parse_args()

    if (args.hparams_file is not None) and (args.checkpoint_file is not None) and (args.hparams_file != "null"):
        run_inference(
            args.hparams_file, 
            args.checkpoint_file, 
            args.datasets.split(","), 
            args.out_dir, 
            args.temperature, 
            args.topk,
            args.codecmodel_path,
            args.use_cfg,
            args.cfg_scale,
            args.batch_size
        )
        return
    else:
        BASE_EXP_DIR = args.base_exp_dir
        DRACO_EXP_DIR = args.draco_exp_dir
        # Mount DRACO_EXP_DIR to BASE_EXP_DIR as follows:
        # sshfs -o allow_other pneekhara@draco-oci-dc-02.draco-oci-iad.nvidia.com:/lustre/fsw/portfolios/llmservice/users/pneekhara/gitrepos/experiments/NewT5AllFixedFresh /datap/misc/dracomount/
        if args.exp_names is None:
            exp_names = os.listdir(BASE_EXP_DIR)
        else:
            exp_names = args.exp_names.split(",")

        for exp_name in exp_names:
            exp_dir = os.path.join(BASE_EXP_DIR, exp_name)
            # recurisvely look for hparams.yaml
            try:
                hparams_file = glob.glob(f"{exp_dir}/**/hparams.yaml", recursive=True)[0]
                checkpoints_dir = glob.glob(f"{exp_dir}/**/checkpoints", recursive=True)[0]
                last_checkpoint = (glob.glob(f"{checkpoints_dir}/*last.ckpt"))[0]
            except:
                print(f"Skipping experiment {exp_name} as hparams or last checkpoint not found.")
                continue
            last_checkpoint_path_draco = last_checkpoint.replace(BASE_EXP_DIR, DRACO_EXP_DIR) 
            epoch_num = last_checkpoint.split("epoch=")[1].split("-")[0]

            checkpoint_copy_path = os.path.join(args.local_ckpt_dir, f"{exp_name}_epoch_{epoch_num}.ckpt")
            hparams_copy_path = os.path.join(args.local_ckpt_dir, f"{exp_name}_hparams.yaml")
            
            scp_command = f"scp {args.server_address}:{last_checkpoint_path_draco} {checkpoint_copy_path}"
            print(f"Running command: {scp_command}")
            os.system(scp_command)
            print("Copied checkpoint.")
            hparams_path_draco = hparams_file.replace(BASE_EXP_DIR, DRACO_EXP_DIR)
            scp_command_hparams = f"scp {args.server_address}:{hparams_path_draco} {hparams_copy_path}"
            print(f"Running command: {scp_command_hparams}")
            os.system(scp_command_hparams)
            print("Copied hparams file.")
            # import ipdb; ipdb.set_trace()
            print("Hparams file path: ", hparams_copy_path)
            print("Checkpoint file path: ", checkpoint_copy_path)
            run_inference(
                hparams_copy_path, 
                checkpoint_copy_path, 
                args.datasets.split(","), 
                args.out_dir, 
                args.temperature, 
                args.topk, 
                args.codecmodel_path, 
                args.use_cfg,
                args.cfg_scale,
                args.batch_size
            )
            

if __name__ == '__main__':
    main()