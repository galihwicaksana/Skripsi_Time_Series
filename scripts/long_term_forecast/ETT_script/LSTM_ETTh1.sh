#!/bin/bash
source /mnt/extended-home/galih/miniconda3/bin/activate timeMixer

cd /mnt/extended-home/galih/Time-Series-Library
export PYTHONPATH="/mnt/extended-home/galih/Time-Series-Library:$PYTHONPATH"

export CUDA_VISIBLE_DEVICES=1

model_name=LSTM

e_layers=2
d_layers=1
d_model=16
d_ff=32
learning_rate=0.01
train_epochs=10
patience=10
batch_size=128

log_file="long_term_forecast_LSTM_ETTh1_results.log"
echo "Training Results - LSTM Baseline" > $log_file
echo "Sequence Lengths: [48, 96, 168, 336]" >> $log_file
echo "Prediction Lengths: [96, 192, 336, 720]" >> $log_file
echo "========================================" >> $log_file
echo "" >> $log_file

for seq_len in 48 96 168 336; do
  # label_len mengikuti pola Transformer/TimesNet
  if [ $seq_len -eq 48 ]; then
    label_len=24
  elif [ $seq_len -eq 96 ]; then
    label_len=48
  elif [ $seq_len -eq 168 ]; then
    label_len=84
  else
    label_len=168
  fi

  for pred_len in 96 192 336 720; do
    model_id="ETTh1_${seq_len}_${pred_len}"

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Running: seq_len=$seq_len, label_len=$label_len, pred_len=$pred_len" | tee -a $log_file

    python -u run.py \
      --task_name long_term_forecast \
      --is_training 1 \
      --root_path ./dataset/ETT-small/ \
      --data_path ETTh1.csv \
      --model_id $model_id \
      --model $model_name \
      --data ETTh1 \
      --features M \
      --seq_len $seq_len \
      --label_len $label_len \
      --pred_len $pred_len \
      --e_layers $e_layers \
      --d_layers $d_layers \
      --enc_in 7 \
      --dec_in 7 \
      --c_out 7 \
      --des 'Exp' \
      --itr 1 \
      --d_model $d_model \
      --d_ff $d_ff \
      --dropout 0.2 \
      --learning_rate $learning_rate \
      --train_epochs $train_epochs \
      --patience $patience \
      --batch_size $batch_size | tee -a $log_file

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Completed: seq_len=$seq_len, label_len=$label_len, pred_len=$pred_len" | tee -a $log_file
    echo "========================================" >> $log_file
    echo "" >> $log_file
  done
done

echo "" | tee -a $log_file
echo "[$(date '+%Y-%m-%d %H:%M:%S')] All training completed for LSTM on ETTh1!" | tee -a $log_file
