#!/bin/bash

set -xe

INPUT_FILE=$1
BATCH_SIZE=$2
OUTPUT_DIR=$3
LANG=$4
PRETRAINED_DIR=$5

BASE_DIR=../data/alignments

DATA_DIR=${BASE_DIR}/${INPUT_FILE}_${LANG}_.csv

TEST_CONFIG="--input_file ${DATA_DIR} --seed 42 --batch_size ${BATCH_SIZE} --output_dir ${OUTPUT_DIR} --lang ${LANG} --pretrained ${PRETRAINED_DIR}"

echo "Start evaluating..."
python transliterate_infer.py ${TEST_CONFIG}