#!/bin/bash

set -xe

EPOCHS=$1
BATCH_SIZE=$2
D_MODEL=$3
SEED=$4
CHUNKS=$5
TYPE_TOKEN_EMBEDDING=$6
TYPE_POSITION_EMBEDDING=$7
ATTENTION_HEADS=$8
MAX_LENGTH=$9
DROPOUT=$10
ACTIVATION=$11
TRAIN_STEP_SIZE=$12
TRANSFORMER_FF_SIZE=$13
GRADIENT_ACCUMULATION_STEPS=$14
PATIENCE=$15
LANG=$16
USE_LABEL_SMOOTHING=$17
SMOOTHING_PARAM=$18
USE_LOCAL_TRANSFORMER=$19

DATA_DIR=../data/alignments

TRAIN_FILE=${DATA_DIR}/train_${LANG}_.csv
VAL_FILE=${DATA_DIR}/valid_${LANG}_.csv

DATA_CONFIG="--train_file ${TRAIN_FILE} --val_file ${VAL_FILE}"

if  [ ${USE_LOCAL_TARNSFORMER} == use_local_transformer ] && [ ${USE_LABEL_SMOOTHING} == use_label_smoothing ]; then    
    TRAIN_CONFIG="--epochs ${EPOCHS} --batch_size ${BATCH_SIZE} --d_model ${D_MODEL} --seed ${SEED} \
                  --chunks ${CHUNKS} \
                  --token_embedding_type ${TYPE_TOKEN_EMBEDDING} --position_embedding_type ${TYPE_POSITION_EMBEDDING} \
                  --attn_heads ${ATTENTION_HEADS} --max_len ${MAX_LENGTH} --dropout ${DROPOUT} \
                  --activation_fn ${ACTIVATION} --train_step_size ${TRAIN_STEP_SIZE} --transformer_ff_size ${TRANSFORMER_FF_SIZE} \
                  --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} --patience ${PATIENCE}  --lang ${LANG} --smoothing_param ${SMOOTHING_PARAM} --use_label_smoothing --use_local_transformer"
elif [ ${USE_LOCAL_TARNSFORMER} == use_local_transformer ] && [ ${USE_LABEL_SMOOTHING} != use_label_smoothing ]; then
       TRAIN_CONFIG="--epochs ${EPOCHS} --batch_size ${BATCH_SIZE} --d_model ${D_MODEL} --seed ${SEED} \
                      --chunks ${CHUNKS} \
                      --token_embedding_type ${TYPE_TOKEN_EMBEDDING} --position_embedding_type ${TYPE_POSITION_EMBEDDING} \
                      --attn_heads ${ATTENTION_HEADS} --max_len ${MAX_LENGTH} --dropout ${DROPOUT} \
                      --activation_fn ${ACTIVATION} --train_step_size ${TRAIN_STEP_SIZE} --transformer_ff_size ${TRANSFORMER_FF_SIZE} \
                      --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} --patience ${PATIENCE}  --lang ${LANG} --use_local_transformer"
elif [ ${USE_LOCAL_TARNSFORMER} != use_local_transformer ] && [ ${USE_LABEL_SMOOTHING} == use_label_smoothing ]; then
    TRAIN_CONFIG="--epochs ${EPOCHS} --batch_size ${BATCH_SIZE} --d_model ${D_MODEL} --seed ${SEED} \
                  --chunks ${CHUNKS} \
                  --token_embedding_type ${TYPE_TOKEN_EMBEDDING} --position_embedding_type ${TYPE_POSITION_EMBEDDING} \
                  --attn_heads ${ATTENTION_HEADS} --max_len ${MAX_LENGTH} --dropout ${DROPOUT} \
                  --activation_fn ${ACTIVATION} --train_step_size ${TRAIN_STEP_SIZE} --transformer_ff_size ${TRANSFORMER_FF_SIZE} \
                  --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} --patience ${PATIENCE} --lang ${LANG} --smoothing_param ${SMOOTHING_PARAM} --use_label_smoothing"
else
    TRAIN_CONFIG="--epochs ${EPOCHS} --batch_size ${BATCH_SIZE} --d_model ${D_MODEL} --seed ${SEED} \
                  --chunks ${CHUNKS} \
                  --token_embedding_type ${TYPE_TOKEN_EMBEDDING} --position_embedding_type ${TYPE_POSITION_EMBEDDING} \
                  --attn_heads ${ATTENTION_HEADS} --max_len ${MAX_LENGTH} --dropout ${DROPOUT} \
                  --activation_fn ${ACTIVATION} --train_step_size ${TRAIN_STEP_SIZE} --transformer_ff_size ${TRANSFORMER_FF_SIZE} \
                  --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} --patience ${PATIENCE} --lang ${LANG}"    
fi
echo "Start training..."
python train.py ${DATA_CONFIG} ${TRAIN_CONFIG}