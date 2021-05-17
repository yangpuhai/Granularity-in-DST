# dataset can be "DSTC2", "WOZ2.0", "sim-M", "sim-R" or "MultiWOZ2.1"
dataset="MultiWOZ2.1"

random_seed=42
train_size_window=0
train_scale=1.0

nohup python -u SpanPtr_train.py \
  --dataset=${dataset} \
  --batch_size=32 \
  --lr=1e-4 \
  --n_epochs=100 \
  --patience=15 \
  --dropout=0.1 \
  --word_dropout=0.1 \
  --random_seed=${random_seed} \
  --hidden_size=400 \
  --train_size_window=${train_size_window} \
  --train_scale=${train_scale} \
  --test_size_window=0 \
  > SpanPtr_${dataset}_gran[${train_size_window}]_scale[${train_scale}]_seed[${random_seed}]_train.log 2>&1 &