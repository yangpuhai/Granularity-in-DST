#!/bin/bash

# Custom parameters
cuda=4,5,7
seed=42
train_size_window=1
train_scale=1.0
max_seq_length=64

output_dir=models-multiwoz2.1/exp_gran[${train_size_window}]_scale[${train_scale}]_seed[${seed}]
target_slot='all'
bert_dir='pytorch_pretrained_bert'

CUDA_VISIBLE_DEVICES=$cuda python3 code/main-multislot.py \
--do_train \
--do_eval \
--num_train_epochs 300 \
--data_dir data/multiwoz2.1 \
--bert_model bert-base-uncased \
--do_lower_case \
--bert_dir $bert_dir \
--task_name bert-gru-sumbt \
--nbt rnn \
--output_dir $output_dir \
--target_slot all \
--warmup_proportion 0.1 \
--learning_rate 5e-5 \
--train_batch_size 4 \
--eval_batch_size 16 \
--distance_metric euclidean \
--patience 15 \
--tf_dir tensorboard-multiwoz2.1 \
--hidden_dim 300 \
--max_label_length 32 \
--max_seq_length $max_seq_length \
--max_turn_length 22 \
--train_size_window $train_size_window \
--train_scale $train_scale \
--seed $seed 

