# Repeat the below script continuously to evaluate the model on the test set
EXP_DIRS=(
    "/lustre/fs8/portfolios/llmservice/users/shehzeenh/mountdir/experiments/LRHM_experiments_Feb24/"
    # "/lustre/fs8/portfolios/llmservice/users/shehzeenh/mountdir/experiments/LRHM_Mar26/"
    # "/lustre/fs8/portfolios/llmservice/users/shehzeenh/mountdir/experiments/LRHM_experiments_Feb24/"
    # /lustre/fs8/portfolios/llmservice/users/pneekhara/gitrepos/experiments/t5_feb24/LossMaskFixed/
)

SERVER_ADDRESSES=(
    "draco-oci-login-01.draco-oci-iad.nvidia.com"
    # "draco-oci-login-01.draco-oci-iad.nvidia.com"
)

EXP_NAMES=(
    # "lrhm_mixed_answercontext_decodercontext"
    "lrhm_5s_decodercontext_allheads"
    # "newcode_LRHM_decodercontext1e-4"
    # "lrhm_5K_combined_decodercontext_allheads_corrected"
    # "lrhm_5K_decodercontext_allheads"
    # "CTC_0.05_All_Data_PS_lr5e-5_Parallel_8k_to_15k"
    # "LRHM_noPriornoCTC"
    # "newcode_LRHM_encodercontext1e-4_onlyPrior"
)

TEST_DSS=(
    "/datap/misc/speechllm_codecdatasets/manifests/smallvctk5scontext_nemo_codec_bw_6.0_phoneme_plus_sentencepiece_tts.json"
    # "/Data/CodecDatasets/updatedcodecs/manifests/testINTEVAL.json"
    # "/datap/misc/speechllm_codecdatasets/manifests/LibriValOrig_nemo_codec_bw_6.0_phoneme_plus_sentencepiece_tts_5s.json"
)

CODEC_MODEL_TYPES=(
    "nemo_codec"
    # "nemo_codec"
)

SEQ_PATTERNS=(
    "parallel"
    # "parallel"
)

FIXED_CHECKPOINTS=(
    "none"
    # "lrhm_5K_combined_decodercontext_allheads_corrected_step234582"
    # "newcode_LRHM_encodercontext1e-4_onlyPrior_step246772"
    # "Lrhm_nopriornotctc249k"
    # "lrhm_5K_decodercontext_allheads_step302660"
    # "newcode_LRHM_decodercontext1e-4_step294116"
    # "lrhm_mixed_answercontext_decodercontext_step122970"
    # "lrhm_5s_decodercontext_allheads_step94801"
    # "newcode_LRHM_decodercontext1e-4_step265792"
    # "lrhm_5K_decodercontext_allheads_step572980"
    # "CTC_0.05_All_Data_PS_lr5e-5_Parallel_8k_to_15k_step238256"
)

ENGLISH_ONLY_MODEL=(
    # "false"
    "true"
    # "true"
)

CONTEXT_CONDITIONING_TYPES=(
    "decoder"
    # "encoder"
    # "encoder"
)

ADD_SPECIAL_TOKENS_TO_FIRST=(
    "false"
    # "false"
)

MAX_SEQ_LEN=(
    "2048"
    # "1536"
)

MAX_INF_STEPS=(
    "2000"
    # "1500"
)

CONTEXT_SECONDS=(
    # "2.9"
    "5"
    # "2.9"
)

# Repeat whole thing 10 times
for ((j=0; j<1; j++)); do

for ((i=0; i<${#EXP_NAMES[@]}; i++)); do

EXP_DIR=${EXP_DIRS[i]}
EXP_NAME=${EXP_NAMES[i]}
TEST_DS=${TEST_DSS[i]}
CODEC_MODEL_TYPE=${CODEC_MODEL_TYPES[i]}
SEQ_PATTERN=${SEQ_PATTERNS[i]}
FIXED_CHECKPOINT=${FIXED_CHECKPOINTS[i]}
SERVER_ADDRESS=${SERVER_ADDRESSES[i]}

# if english only model is true, then set the language model path to the english only model
if [ "${ENGLISH_ONLY_MODEL[i]}" = "true" ]; then
    if [ "${MAX_SEQ_LEN[i]}" = "1536" ]; then
        LANGUAGE_MODEL_PATH="/Data/Checkpoints/megatron_t5_expanded_vocab_posemb1536.nemo"
    else
        LANGUAGE_MODEL_PATH="/Data/Checkpoints/megatron_t5_expanded_vocab_posemb_2048.nemo"
    fi
    OVERRIDE_TOKEN_MODEL="null"
    SPEECH_OFFSET=30128
    LM_VCOAB_SIZE=30000
    NUM_SENTINEL_TOKENS=10128
else
    LANGUAGE_MODEL_PATH="None"
    OVERRIDE_TOKEN_MODEL="None"
    SPEECH_OFFSET=250265
    LM_VCOAB_SIZE=250265
    NUM_SENTINEL_TOKENS=9832
fi

# if codec model type is dac, then set the codec fps to 100
if [ "$CODEC_MODEL_TYPE" = "nemo_codec" ]; then
    CODEC_FPS=86
    CODEC_MODEL_CODEBOOKS=8
else
    CODEC_FPS=75
    CODEC_MODEL_CODEBOOKS=8
fi


LOCAL_CKPT_DIR="/Data/FebTTSCheckpoints"

# If fixed_checkpoint is none, then copy the checkpoint from selene-login

if [ "$FIXED_CHECKPOINT" = "none" ]; then
    echo "Copying checkpoint from selene-login"
    CHECKPOINT_PATH=$EXP_DIR/$EXP_NAME/p_tuning_squad_t5/checkpoints/*last.ckpt
    
    scp shehzeenh@$SERVER_ADDRESS:$CHECKPOINT_PATH $LOCAL_CKPT_DIR

    # Read name of the checkpoint file in LOCAL_CKPT_DIR
    CHECKPOINT_FILE=$(ls $LOCAL_CKPT_DIR | grep "last.ckpt")
    # Take 1st line of the above output and first word of that line
    CHECKPOINT_FILE=$(echo $CHECKPOINT_FILE | head -n 1 | awk '{print $1;}')

    # Get the iter number after "step=" in the checkpoint file name
    CHECKPOINT_ITER=$(echo $CHECKPOINT_FILE | sed -e 's/.*step=\([0-9]*\).*/\1/')

    echo "Checkpoint file: $CHECKPOINT_FILE"
    echo "Checkpoint iter: $CHECKPOINT_ITER"

    # New checkpoint filename is EXP_NAME + CHECKPOINT_ITER .ckpt
    NEW_CHECKPOINT_FILE=$EXP_NAME"_step"$CHECKPOINT_ITER".ckpt"
    NEW_EXP_NAME=$EXP_NAME"_step"$CHECKPOINT_ITER

    echo "New checkpoint file: $NEW_CHECKPOINT_FILE"

    # rename the checkpoint file
    echo "mv $LOCAL_CKPT_DIR/$CHECKPOINT_FILE $LOCAL_CKPT_DIR/$NEW_CHECKPOINT_FILE"

    mv $LOCAL_CKPT_DIR/$CHECKPOINT_FILE $LOCAL_CKPT_DIR/$NEW_CHECKPOINT_FILE
else
    echo "Using Given Checkpoint: $FIXED_CHECKPOINT"
    NEW_CHECKPOINT_FILE=$FIXED_CHECKPOINT".ckpt"
    NEW_EXP_NAME=$FIXED_CHECKPOINT
fi

export HYDRA_FULL_ERROR=1 ;

read -r -d '' cmd <<EOF
python examples/nlp/language_modeling/megatron_t5_speechlm_sft_inference.py \
--config-name=megatron_t5_speechlm_inference.yaml \
name=$NEW_EXP_NAME \
model.data.test_ds='["$TEST_DS"]' \
model.data.g2p.english.phoneme_dict="/home/shehzeenh/Code/NeMo/scripts/tts_dataset_files/ipa_cmudict-0.7b_nv23.01.txt" \
model.data.g2p.english.heteronyms="/home/shehzeenh/Code/NeMo/scripts/tts_dataset_files/heteronyms-052722" \
model.data.g2p.spanish.phoneme_dict="/home/shehzeenh/Code/NeMo/scripts/tts_dataset_files/es_ES/es_ES_nv230301.dict" \
model.data.g2p.mandarin.phoneme_dict="/home/shehzeenh/Code/NeMo/scripts/tts_dataset_files/zh/36finals/ipa_dict_nv23.05.txt" \
model.data.g2p.german.phoneme_dict="/home/shehzeenh/Code/NeMo/scripts/tts_dataset_files/de/de_nv240125.dict" \
model.data.g2p.german.heteronyms="/home/shehzeenh/Code/NeMo/scripts/tts_dataset_files/de/de_nv230119.heteronym" \
+model.data.add_special_tokens_to_only_first_codebook=${ADD_SPECIAL_TOKENS_TO_FIRST[i]} \
model.data.train_task=all \
+model.freeze_model=False \
model.data.max_seq_length=${MAX_SEQ_LEN[i]} \
model.max_inference_timesteps=${MAX_INF_STEPS[i]} \
+model.data.context_duration_min=${CONTEXT_SECONDS[i]} \
+model.data.context_duration_max=${CONTEXT_SECONDS[i]} \
+model.data.context_pattern=parallel \
model.top_k=80 \
model.temperature=0.85 \
exp_manager.exp_dir=/Data/Experiments/MarchInterspeechEVAL/WAVLM_VCTK \
model.data.sup_data_path=/Data/SupDir/ \
model.global_batch_size=32 \
model.micro_batch_size=32 \
+model.num_sentinel_tokens=$NUM_SENTINEL_TOKENS \
model.data.speech_offset=$SPEECH_OFFSET \
+model.lm_vocab_size=$LM_VCOAB_SIZE \
+model.data.num_speech_codebooks=$CODEC_MODEL_CODEBOOKS \
+model.data.codebook_fps=$CODEC_FPS \
+model.codecmodel_type=$CODEC_MODEL_TYPE \
+model.codecmodel_path=/Data/Checkpoints/rlang_codec/SpeechCodec.nemo \
trainer.devices=1 \
trainer.precision=bf16 \
model.language_model_path=$LANGUAGE_MODEL_PATH \
+model.override_token_model=$OVERRIDE_TOKEN_MODEL \
model.seq_pattern=$SEQ_PATTERN \
checkpoint_path="$LOCAL_CKPT_DIR/$NEW_CHECKPOINT_FILE" \
+model.english_only_model=${ENGLISH_ONLY_MODEL[i]} \
+model.context_conditioning="${CONTEXT_CONDITIONING_TYPES[i]}" \
+model.asr_model_name="stt_en_conformer_transducer_large" \
model.speech_head_type=linear
EOF

echo "Running "command": $cmd"

eval $cmd

# read -r -d '' cmdd <<EOF
# python hallucination_eval.py \
# --exp_base_dir /Data/Experiments/MarchInterspeechEVAL/WAVLM_VCTK \
# --exp_name $NEW_EXP_NAME \
# --manifest_path $TEST_DS
# EOF

# echo "Running "command": $cmdd"

# eval $cmdd

# Remove the checkpoint file
echo "rm $LOCAL_CKPT_DIR/$NEW_CHECKPOINT_FILE"

# rm $LOCAL_CKPT_DIR/$NEW_CHECKPOINT_FILE

# sleep 1m # sleep for 2 minutes

done

done
