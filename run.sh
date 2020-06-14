#!/bin/bash

export MAX_LENGTH=70
export BERT_MODEL=bert-base-multilingual-cased
export BATCH_SIZE=32
export NUM_EPOCHS=3
export SEED=1

export DATA_DIR=data/partut
export OUTPUT_DIR_NAME=models/partut-model
export CURRENT_DIR=${PWD}
export OUTPUT_DIR=${CURRENT_DIR}/${OUTPUT_DIR_NAME}
mkdir -p $OUTPUT_DIR

python tagger.py --data_dir $DATA_DIR \
--model_name_or_path $BERT_MODEL \
--output_dir $OUTPUT_DIR \
--max_seq_length $MAX_LENGTH \
--num_train_epochs $NUM_EPOCHS \
--train_batch_size $BATCH_SIZE \
--seed $SEED \
--do_train
