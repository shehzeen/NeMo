To perform DPO/RPO/PPO on T5-TTS models, follow the below steps.

## Step 1: Generate a list of text and context pairs for which we want to generate preference data
We pair challenging texts with random audio context from Libri/Riva speakers. 
We also create a similar number of text-context pairs with regular text from our training dataset (but pair those texts with a random audio context different from the one used in training)

```
python scripts/speechllm_dpo/create_text_context_pairs.py \
--challenging_texts /Data/challenging_texts_nemollm.txt \
--riva_manifest /Data/CodecDatasets/updatedcodecs/manifests/RivattsEnglish_train_nemo_codec_bw_6.0_phoneme_tts_highsimilarity3.json \
--libri_manifest /Data/CodecDatasets/updatedcodecs/manifests/LibriTTSCorrectContext_train_nemo_codec_bw_6.0_phoneme_tts_highsimilarity2.json \
--output_manifest /Data/CodecDatasets/updatedcodecs/manifests/challenging_nemo21Hz_textcontextpairs.json ;
````

## Step 2: Generate multiple audios for each text-context pair 
We generate around 4 to 6 audios from a pre-trained T5-TTS checkpoint for each text-context pair from which we'll select the best and worst generation in the next step.
Text-context pairs passed as `rlhf_train_ds` in the argument. The below command generates output tokens and saves the codecs and a manifest in the experiment directory.
`rlhf_num_samples_per_example` is the number of audios we want to generate for each text-audio pair. 
`rlhf_num_generations_per_iteration` specifies how many total audios we want to generate. 
To generate all audios for all pairs set `rlhf_num_generations_per_iteration = rlhf_num_samples_per_example * SIZE OF textcontextpairs.json`

```
python examples/tts/speechllm/megatron_t5_speechllm_dpo.py \
--config-name=megatron_t5_speechllm_multiencoder.yaml \
+init_from_ptl_ckpt="/Data/Checkpoints/multiencoder_step135k.ckpt" \
+mode="generate" \
exp_manager.exp_dir=/Data/Experiments/DPO_GenerationsDebug \
name="DPO21Hz_90kGenerations" \
model.english_only_model=true \
model.global_batch_size=128 \
model.micro_batch_size=128 \
trainer.devices=1 \
trainer.precision=bf16 \
model.data.train_task=tts \
model.data.train_ds='["/Data/CodecDatasets/updatedcodecs/manifests/empty_file.json"]' \
model.data.validation_ds='["/Data/CodecDatasets/updatedcodecs/manifests/empty_file.json"]' \
+model.data.test_ds='["/Data/CodecDatasets/updatedcodecs/manifests/empty_file.json"]' \
+model.rlhf_train_ds='["/Data/CodecDatasets/updatedcodecs/manifests/challenging_nemo21Hz_textcontextpairs.json"]' \
+model.data.sup_data_path="/datap/misc/speechllm_codecdatasets/" \
model.data.g2p.english.phoneme_dict="/home/shehzeenh/Code/NeMo/scripts/tts_dataset_files/ipa_cmudict-0.7b_nv23.01.txt" \
model.data.g2p.english.heteronyms="/home/shehzeenh/Code/NeMo/scripts/tts_dataset_files/heteronyms-052722" \
model.data.g2p.spanish.phoneme_dict="/home/shehzeenh/Code/NeMo/scripts/tts_dataset_files/es_ES/es_ES_nv230301.dict" \
model.data.g2p.mandarin.phoneme_dict="/home/shehzeenh/Code/NeMo/scripts/tts_dataset_files/zh/36finals/ipa_dict_nv23.05.txt" \
model.data.g2p.german.phoneme_dict="/home/shehzeenh/Code/NeMo/scripts/tts_dataset_files/de/de_nv240125.dict" \
model.data.g2p.german.heteronyms="/home/shehzeenh/Code/NeMo/scripts/tts_dataset_files/de/de_nv230119.heteronym" \
model.data.max_seq_length=512 \
+model.max_inference_timesteps=510 \
+model.data.speech_codebook_size=2048 \
model.data.codebook_fps=21 \
+model.data.add_special_tokens_to_only_first_codebook=true \
+model.data.min_seq_length=5 \
+model.temperature=0.9 \
model.data.context_duration_min=3.0 \
model.data.context_duration_max=20.0 \
model.data.use_attention_prior=false \
model.use_alignment_loss=true \
model.codecmodel_path="/Data/Checkpoints/AudioCodec_21Hz-2k-codes_updated.nemo" \
model.data.speech_offset=30128 \
model.lm_vocab_size=30000 \
model.frozen_model.encoder.hidden_size=768 \
model.frozen_model.encoder.num_layers=6 \
model.frozen_model.decoder.num_layers=12 \
model.frozen_model.decoder.hidden_size=768 \
model.frozen_model.decoder.ffn_hidden_size=2048 \
model.override_tokenizer_vocab_file="/Data/Checkpoints/9a77f10c2793465e8e8a3fa5fcbef8b0_vocab.txt" \
+model.asr_model_name="nvidia/parakeet-tdt-1.1b" \
'+model.frozen_model.decoder.layer_type=[1,1,1,2,2,2,2,2,2,2,1,1]' \
'+model.alignment_decoder_layerids=[0,1,2,3,4]' \
'model.enc_output_to_layers=[[8,9],[3,4,5,6,7]]' \
model.data.num_workers=2 \
+model.rlhf_num_samples_per_example=6 \
+model.rlhf_num_generations_per_iteration=288

```

## Step 3: Evaluate the CER/SSIM of generated audio and create preference dataset

The below commands caclulates CER/SSIM of all generated audios. For each group in the generated audio it finds the best and worst record.
Best record has lowest CER and incase of a tie, a higher SSIM is prefered. Can explore more strategies. 
The final train and validation manifests are saved in the same experiment directory. 
In each manifest, we have 2 chosen records, followed by 2 rejected rewards and so on. 
We assume the micro-batch size is 4 during training so first half of the micro-batch is chosen and the 2nd half is rejected examples. 
(Train/val sets are not shuffled during training to preserve this order)

```
python scripts/speechllm_dpo/create_preference_data_from_generations.py \
--codec_model_path /Data/Checkpoints/AudioCodec_21Hz-2k-codes_updated.nemo \
--generated_manifest /Data/Experiments/DPO_GenerationsDebug/DPO21Hz_90kGenerations/rlhf_generations/generated_outputs_manifest.json \
--batch_size 8 \
--sample_rate 22050 \
--group_size 6 \
--val_size 256
```
The above command should generate train and val manifests as `generated_outputs_manifest_final_records_train.json` and `generated_outputs_manifest_final_records_val.json` in the same directory as the input `generated_manifest`.
These should be suitable for training locally with a micro-batch size of 4 on a single device.

If we are training on cluster, we need to reorder the preference dataset so that micro-batch on each GPU in a node gets 2 chosen and 2 rejected examples (for a micro-batch size of 4).
For this, we have a script that handles the re-ordering for 2 nodes assuming micro-batch size of 4. 

```
python scripts/speechllm_dpo/reshuffle_manifest.py --manifest /Data/Experiments/DPO_GenerationsDebug/DPO21Hz_90kGenerations/rlhf_generations/generated_outputs_manifest_final_records_train.json
python scripts/speechllm_dpo/reshuffle_manifest.py --manifest /Data/Experiments/DPO_GenerationsDebug/DPO21Hz_90kGenerations/rlhf_generations/generated_outputs_manifest_final_records_val.json
```

The above commands save `final_records_train_scalar_rewards_2nodes.json` and `final_records_val_scalar_rewards_2nodes.json` which can be used for training on cluster with 2 nodes.

## Step 4: Train DPO/RPO on the preference dataset

To train locally, run the following command. 

```
HYDRA_FULL_ERROR=1 NEMO_ENABLE_COLORING=1 python examples/tts/speechllm/megatron_t5_speechllm_dpo.py \
--config-name=megatron_t5_speechllm_multiencoder.yaml \
+init_from_ptl_ckpt="/Data/Checkpoints/multiencoder_step135k.ckpt" \
+mode="train" \
exp_manager.exp_dir=/Data/Experiments/DPO21HzDebug \
name="DPODebug" \
+model.dpo_loss_type=dpo \
+model.dpo_beta=0.01 \
+model.dpo_sft_loss_weight=0.0 \
+model.dpo_pref_loss_weight=1.0 \
model.english_only_model=true \
model.global_batch_size=4 \
model.micro_batch_size=4 \
model.validation_drop_last=True \
trainer.max_steps=1000000 \
trainer.devices=1 \
trainer.precision=bf16 \
model.optim.lr=1e-6 \
~model.optim.sched \
model.data.train_task=tts \
model.data.train_ds='["/Data/Experiments/DPO_GenerationsDebug/DPO21Hz_90kGenerations/rlhf_generations/generated_outputs_manifest_final_records_train.json"]' \
model.data.validation_ds='["/Data/Experiments/DPO_GenerationsDebug/DPO21Hz_90kGenerations/rlhf_generations/generated_outputs_manifest_final_records_val.json"]' \
+model.data.test_ds='["/Data/CodecDatasets/updatedcodecs/manifests/dummy_test.json"]' \
+model.data.sup_data_path="/datap/misc/speechllm_codecdatasets/" \
model.data.g2p.english.phoneme_dict="/home/shehzeenh/Code/NeMo/scripts/tts_dataset_files/ipa_cmudict-0.7b_nv23.01.txt" \
model.data.g2p.english.heteronyms="/home/shehzeenh/Code/NeMo/scripts/tts_dataset_files/heteronyms-052722" \
model.data.g2p.spanish.phoneme_dict="/home/shehzeenh/Code/NeMo/scripts/tts_dataset_files/es_ES/es_ES_nv230301.dict" \
model.data.g2p.mandarin.phoneme_dict="/home/shehzeenh/Code/NeMo/scripts/tts_dataset_files/zh/36finals/ipa_dict_nv23.05.txt" \
model.data.g2p.german.phoneme_dict="/home/shehzeenh/Code/NeMo/scripts/tts_dataset_files/de/de_nv240125.dict" \
model.data.g2p.german.heteronyms="/home/shehzeenh/Code/NeMo/scripts/tts_dataset_files/de/de_nv230119.heteronym" \
+model.data.add_special_tokens_to_only_first_codebook=true \
+model.data.min_seq_length=5 \
model.data.context_duration_min=3.0 \
model.data.context_duration_max=20.0 \
model.data.max_seq_length=512 \
+model.max_inference_timesteps=510 \
+model.data.speech_codebook_size=2048 \
model.data.codebook_fps=21 \
model.data.use_attention_prior=false \
model.use_alignment_loss=true \
+model.alignment_loss_end_step=0 \
model.codecmodel_path="/Data/Checkpoints/AudioCodec_21Hz-2k-codes_updated.nemo" \
model.data.speech_offset=30128 \
model.lm_vocab_size=30000 \
model.frozen_model.encoder.hidden_size=768 \
model.frozen_model.encoder.num_layers=6 \
model.frozen_model.decoder.num_layers=12 \
model.frozen_model.decoder.hidden_size=768 \
model.frozen_model.decoder.ffn_hidden_size=2048 \
+model.alignment_loss_scale=0.1 \
+model.alignment_text_end_offset=1 \
exp_manager.resume_if_exists=True \
exp_manager.resume_ignore_no_checkpoint=True \
exp_manager.create_early_stopping_callback=False \
~exp_manager.early_stopping_callback_params \
exp_manager.checkpoint_callback_params.save_top_k=3 \
~trainer.check_val_every_n_epoch \
trainer.val_check_interval=10 \
model.override_tokenizer_vocab_file="/Data/Checkpoints/9a77f10c2793465e8e8a3fa5fcbef8b0_vocab.txt" \
+model.asr_model_name="nvidia/parakeet-tdt-1.1b" \
'+model.frozen_model.decoder.layer_type=[1,1,1,2,2,2,2,2,2,2,1,1]' \
'+model.alignment_decoder_layerids=[0,1,2,3,4]' \
'model.enc_output_to_layers=[[8,9],[3,4,5,6,7]]' 
```

Some important arguments in the above command are
```
+model.dpo_loss_type=dpo \
+model.dpo_beta=0.01 \
+model.dpo_sft_loss_weight=0.0 \
+model.dpo_pref_loss_weight=1.0 \
```

Switch these out with rpo or other preference loss functions.

A sample training sub file on EOS: `/lustre/fsw/llmservice_nemo_speechlm/users/shehzeenh/launchscripts/dpo_21Hz.sub`

Evaluate the checkpoints the same way as we would evaluate the base checkpoint `/Data/Checkpoints/multiencoder_step135k.ckpt`