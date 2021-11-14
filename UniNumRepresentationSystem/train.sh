#!/bin/bash

set -xe

EPOCHS=$1
BATCH_SIZE=$2
D_MODEL=$3
SEED=$4
TRAIN_CSV=$5
TYPE_TOKEN_EMBEDDING=$6
TYPE_POSITION_EMBEDDING=$7
ATTENTION_HEADS=$8
MAX_LENGTH=$9
DROPOUT=$10
ACTIVATION=$11
TRANSFORMER_FF_SIZE=$12
GRADIENT_ACCUMULATION_STEPS=$13
PATIENCE=$14
SMOOTHING_PARAM=$15
AUX_CSV=$16
K_PARAM=$17
VAL_SPLIT=$18
LR=$19
USE_LABEL_SMOOTHING=$20
USE_LOCAL_TRANSFORMER=$21


DATA_DIR=../data

TRAIN_FILE=${DATA_DIR}/${TRAIN_CSV}
AUX_TRAIN_FILE=${DATA_DIR}/${AUX_CSV}

DATA_CONFIG="--train_csv ${TRAIN_FILE} --auxillary_data_csv ${AUX_TRAIN_FILE}"

if [ ${USE_LOCAL_TARNSFORMER} == use_local_transformer ] && [ ${USE_LABEL_SMOOTHING} == use_label_smoothing ]; then
    TRAIN_CONFIG="--epochs ${EPOCHS} --batch_size ${BATCH_SIZE} --d_model ${D_MODEL} --seed ${SEED} \
                  --token_embedder_type ${TYPE_TOKEN_EMBEDDING} --pos_embedder_type ${TYPE_POSITION_EMBEDDING} \
                  --n_heads ${ATTENTION_HEADS} --max_length ${MAX_LENGTH} --dropout ${DROPOUT} --learning_rate ${LR}\
                  --activation ${ACTIVATION} --feedforward_dim ${TRANSFORMER_FF_SIZE} --k ${K_PARAM} --val_split ${VAL_SPLIT}\
                  --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} --patience ${PATIENCE} --smoothing_param ${SMOOTHING_PARAM}  --embedder_slice_count 8 --embedder_bucket_count 16000 --hidden_dim 768 --n_classes 9 --n_layers 1 --use_label_smoothing --use_local_transformer"
elif [ ${USE_LOCAL_TARNSFORMER} == use_local_transformer ]; then    
    TRAIN_CONFIG="--epochs ${EPOCHS} --batch_size ${BATCH_SIZE} --d_model ${D_MODEL} --seed ${SEED} \
                  --token_embedder_type ${TYPE_TOKEN_EMBEDDING} --pos_embedder_type ${TYPE_POSITION_EMBEDDING} \
                  --n_heads ${ATTENTION_HEADS} --max_length ${MAX_LENGTH} --dropout ${DROPOUT} --learning_rate ${LR}\
                  --activation ${ACTIVATION} --feedforward_dim ${TRANSFORMER_FF_SIZE} --k ${K_PARAM} --val_split ${VAL_SPLIT}\
                  --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} --patience ${PATIENCE} --smoothing_param ${SMOOTHING_PARAM} --embedder_slice_count 8 --embedder_bucket_count 16000 --hidden_dim 768 --n_classes 9 --n_layers 1 --use_local_transformer"
elif [ ${USE_LABEL_SMOOTHING} == use_label_smoothing ]; then
    TRAIN_CONFIG="--epochs ${EPOCHS} --batch_size ${BATCH_SIZE} --d_model ${D_MODEL} --seed ${SEED} \
                  --token_embedder_type ${TYPE_TOKEN_EMBEDDING} --pos_embedder_type ${TYPE_POSITION_EMBEDDING} \
                  --n_heads ${ATTENTION_HEADS} --max_length ${MAX_LENGTH} --dropout ${DROPOUT} --learning_rate ${LR}\
                  --activation ${ACTIVATION} --feedforward_dim ${TRANSFORMER_FF_SIZE} --k ${K_PARAM} --val_split ${VAL_SPLIT}\
                  --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} --patience ${PATIENCE} --smoothing_param ${SMOOTHING_PARAM} --embedder_slice_count 8 --embedder_bucket_count 16000 --hidden_dim 768 --n_classes 9 --n_layers 1 --use_label_smoothing"
else
    TRAIN_CONFIG="--epochs ${EPOCHS} --batch_size ${BATCH_SIZE} --d_model ${D_MODEL} --seed ${SEED} \
                  --token_embedder_type ${TYPE_TOKEN_EMBEDDING} --pos_embedder_type ${TYPE_POSITION_EMBEDDING} \
                  --n_heads ${ATTENTION_HEADS} --max_length ${MAX_LENGTH} --dropout ${DROPOUT} --learning_rate ${LR}\
                  --activation ${ACTIVATION} --feedforward_dim ${TRANSFORMER_FF_SIZE} --k ${K_PARAM} --val_split ${VAL_SPLIT}\
                  --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} --patience ${PATIENCE} --n_layers 1 --smoothing_param ${SMOOTHING_PARAM} --embedder_slice_count 8 --embedder_bucket_count 16000 --hidden_dim 768 --n_classes 9"
fi

echo "Start training..."
python train.py ${DATA_CONFIG} ${TRAIN_CONFIG}