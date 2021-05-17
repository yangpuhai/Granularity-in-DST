# dataset can be "DSTC2", "WOZ2.0", "sim-M", "sim-R" or "MultiWOZ2.1"
dataset="MultiWOZ2.1"

random_seed=42
train_size_window=1
test_size_window=1
train_scale=1.0
max_seq_length=320

nohup python -u SOMDST_train_SG.py \
  --dataset=${dataset} \
  --max_seq_length=${max_seq_length} \
  --batch_size=16 \
  --enc_lr=4e-5 \
  --dec_lr=1e-4 \
  --n_epochs=200 \
  --patience=15 \
  --patience_start_epoch=30 \
  --dropout=0.1 \
  --word_dropout=0.1 \
  --random_seed=${random_seed} \
  --train_size_window=${train_size_window} \
  --train_scale=${train_scale} \
  --test_size_window=${test_size_window} \
  > SOMDST_${dataset}_TG[${train_size_window}]_IG[${test_size_window}]_scale[${train_scale}]_seed[${random_seed}]_train.log 2>&1 &