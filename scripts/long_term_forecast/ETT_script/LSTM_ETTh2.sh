#!/bin/bash
source /mnt/extended-home/galih/miniconda3/bin/activate timeMixer

# Set working directory and Python path
cd /mnt/extended-home/galih/Time-Series-Library
export PYTHONPATH="/mnt/extended-home/galih/Time-Series-Library:$PYTHONPATH"

export CUDA_VISIBLE_DEVICES=0

model_name=LSTM
root_path=./dataset/ETT-small/
data_path=ETTh2.csv
data=ETTh2
features=M
e_layers=3
d_layers=1
enc_in=7
dec_in=7
c_out=7
d_model=100
d_ff=32
des='Exp'
itr=1
batch_size=128
learning_rate=0.01
train_epochs=10
patience=10

log_file="long_term_forecast_LSTM_ETTh2_results.log"
echo "Training Results - LSTM Baseline" > $log_file
echo "Sequence Lengths: [48, 96, 168, 336]" >> $log_file
echo "Prediction Lengths: [96, 192, 336, 720]" >> $log_file
echo "========================================" >> $log_file
echo "" >> $log_file

for seq_len in 48 96 168 336; do
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
    model_id="ETTh2_${seq_len}_${pred_len}"

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Running: seq_len=$seq_len, label_len=$label_len, pred_len=$pred_len" | tee -a $log_file

    python -u run.py \
      --task_name long_term_forecast \
      --is_training 1 \
      --root_path $root_path \
      --data_path $data_path \
      --model_id $model_id \
      --model $model_name \
      --gpu 0 \
      --data $data \
      --features $features \
      --seq_len $seq_len \
      --label_len $label_len \
      --pred_len $pred_len \
      --e_layers $e_layers \
      --d_layers $d_layers \
      --enc_in $enc_in \
      --dec_in $dec_in \
      --c_out $c_out \
      --d_model $d_model \
      --d_ff $d_ff \
      --dropout 0.2 \
      --batch_size $batch_size \
      --learning_rate $learning_rate \
      --train_epochs $train_epochs \
      --patience $patience \
      --des $des \
      --itr $itr | tee -a $log_file

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Completed: seq_len=$seq_len, label_len=$label_len, pred_len=$pred_len" | tee -a $log_file
    echo "========================================" >> $log_file
    echo "" >> $log_file
  done
done

echo "" | tee -a $log_file
echo "[$(date '+%Y-%m-%d %H:%M:%S')] All training completed for LSTM on ETTh2!" | tee -a $log_file
