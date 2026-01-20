#!/bin/bash
source /mnt/extended-home/galih/miniconda3/bin/activate timeMixer

# Set working directory and Python path
cd /mnt/extended-home/galih/Time-Series-Library
export PYTHONPATH="/mnt/extended-home/galih/Time-Series-Library:$PYTHONPATH"

export CUDA_VISIBLE_DEVICES=1

model_name=TimeMixer
root_path=./dataset/ETT-small/
data_path=ETTh2.csv
data=ETTh2
features=M
e_layers=2
down_sampling_layers=3
down_sampling_window=2
learning_rate=0.01
d_model=16
d_ff=32
train_epochs=10
patience=10
batch_size=128
enc_in=7
c_out=7
des='Exp'
itr=1

log_file="long_term_forecast_timeMixer_ETTh2_results.log"
echo "Training Results - Comprehensive Experiment" > $log_file
echo "Sequence Lengths: [48, 96, 168, 336]" >> $log_file
echo "Prediction Lengths: [96, 192, 336, 720]" >> $log_file
echo "========================================" >> $log_file
echo "" >> $log_file

for seq_len in 48 96 168 336; do
  for pred_len in 96 192 336 720; do
    model_id="ETTh2_${seq_len}_${pred_len}"
    
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Running training: seq_len=$seq_len, pred_len=$pred_len" | tee -a $log_file

    python -u run.py \
      --task_name long_term_forecast \
      --is_training 1 \
      --root_path $root_path \
      --data_path $data_path \
      --model_id $model_id \
      --model $model_name \
      --data $data \
      --features $features \
      --seq_len $seq_len \
      --label_len 0 \
      --pred_len $pred_len \
      --e_layers $e_layers \
      --enc_in $enc_in \
      --c_out $c_out \
      --des $des \
      --itr $itr \
      --d_model $d_model \
      --d_ff $d_ff \
      --learning_rate $learning_rate \
      --train_epochs $train_epochs \
      --patience $patience \
      --batch_size $batch_size \
      --down_sampling_layers $down_sampling_layers \
      --down_sampling_method avg \
      --down_sampling_window $down_sampling_window | tee -a $log_file

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Completed: seq_len=$seq_len, pred_len=$pred_len" | tee -a $log_file
    echo "========================================" >> $log_file
    echo "" >> $log_file

  done
done

echo "" | tee -a $log_file
echo "[$(date '+%Y-%m-%d %H:%M:%S')] All training completed for TimeMixer on ETTh2!" | tee -a $log_file
