#!/bin/bash

export BERT_BASE_DIR="bert/"
export BERT_DATA_DIR="models/uncased_L-12_H-768_A-12/"

export INPUT_DIR = "input_files"
export OUTPUT_DIR = "output_files"
rm -rf $OUTPUT_DIR
mkdir -p $OUTPUT_DIR

for i in bert_classification_train bert_classification_valid
do
    python $BERT_BASE_DIR/extract_features.py \
    --vocab_file=$BERT_DATA_DIR/vocab.txt \
    --bert_config_file=$BERT_DATA_DIR/bert_config.json \
    --init checkpoint=$BERT_DATA_DIR/bert_model.ckpt \
    --max seq_length=128 \
    --batch_size=8 \
    --layers=-1 \
    --input_file=$INPUT_DIR/$i.csv \
    --output_file=$OUTPUT_DIR/$i.jsonlines
done