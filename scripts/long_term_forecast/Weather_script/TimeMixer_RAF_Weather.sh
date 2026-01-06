#!/bin/bash
source /mnt/extended-home/galih/miniconda3/bin/activate timeMixer

# Set working directory and Python path
cd /mnt/extended-home/galih/Time-Series-Library
export PYTHONPATH="/mnt/extended-home/galih/Time-Series-Library:$PYTHONPATH"

export CUDA_VISIBLE_DEVICES=0

model_name=TimeMixer_RAF

# Weather dataset: 21 features
e_layers=2
down_sampling_layers=3
down_sampling_window=2
learning_rate=0.01
d_model=16
d_ff=32
train_epochs=10
patience=10
batch_size=128

# RAF-specific hyperparameters (OPTIMIZED)
use_raf=1
raf_d_embed=16  # Match d_model
raf_fusion_method='gating'
raf_similarity_metric='cosine'
kb_path='./checkpoints/knowledge_bases/weather_kb.pkl'

# Log file
log_file="long_term_forecast_RAF_Weather_results.log"
echo "RAF-TimeMixer Training Results - Weather Dataset" > $log_file
echo "Hyperparameters:" >> $log_file
echo "  Learning Rate: $learning_rate" >> $log_file
echo "  Epochs: $train_epochs" >> $log_file
echo "  Patience: $patience" >> $log_file
echo "  Batch Size: $batch_size" >> $log_file
echo "  d_model: $d_model" >> $log_file
echo "  d_ff: $d_ff" >> $log_file
echo "RAF parameters:" >> $log_file
echo "  use_raf: $use_raf" >> $log_file
echo "  raf_top_k: ADAPTIVE (3/5/7/10)" >> $log_file
echo "  raf_d_embed: $raf_d_embed" >> $log_file
echo "  raf_fusion_method: $raf_fusion_method" >> $log_file
echo "  KB size: 5000 patterns" >> $log_file
echo "========================================" >> $log_file
echo "" >> $log_file

# Build knowledge base
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Building Knowledge Base (5000 patterns)..." | tee -a $log_file
python -u scripts/build_kb_weather.py \
  --model $model_name \
  --max_kb_size 5000 \
  --data custom \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --seq_len 96 \
  --pred_len 96 \
  --enc_in 21 \
  --c_out 21 \
  --e_layers $e_layers \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size $batch_size \
  --down_sampling_layers $down_sampling_layers \
  --down_sampling_method avg \
  --down_sampling_window $down_sampling_window \
  --use_raf $use_raf \
  --raf_d_embed $raf_d_embed \
  --raf_fusion_method $raf_fusion_method \
  --raf_similarity_metric $raf_similarity_metric \
  --kb_path $kb_path | tee -a $log_file

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Knowledge Base built successfully!" | tee -a $log_file
echo "" >> $log_file

# Run experiments - 4 seq_len × 4 pred_len = 16 experiments
for seq_len in 48 96 168 336; do
  for pred_len in 96 192 336 720; do
    
    # Adaptive top_k
    if [ $pred_len -eq 96 ]; then
      raf_top_k=3
    elif [ $pred_len -eq 192 ]; then
      raf_top_k=5
    elif [ $pred_len -eq 336 ]; then
      raf_top_k=7
    else  # 720
      raf_top_k=10
    fi
    
    model_id="Weather_${seq_len}_${pred_len}_RAF"

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Running training: seq_len=$seq_len, pred_len=$pred_len, top_k=$raf_top_k" | tee -a $log_file

    python -u run.py \
      --task_name long_term_forecast \
      --is_training 1 \
      --root_path ./dataset/weather/ \
      --data_path weather.csv \
      --model_id $model_id \
      --model $model_name \
      --data custom \
      --features M \
      --seq_len $seq_len \
      --label_len 0 \
      --pred_len $pred_len \
      --e_layers $e_layers \
      --enc_in 21 \
      --dec_in 21 \
      --c_out 21 \
      --des Exp \
      --itr 1 \
      --d_model $d_model \
      --d_ff $d_ff \
      --learning_rate $learning_rate \
      --train_epochs $train_epochs \
      --patience $patience \
      --batch_size $batch_size \
      --down_sampling_layers $down_sampling_layers \
      --down_sampling_method avg \
      --down_sampling_window $down_sampling_window \
      --use_raf $use_raf \
      --raf_top_k $raf_top_k \
      --raf_d_embed $raf_d_embed \
      --raf_fusion_method $raf_fusion_method \
      --raf_similarity_metric $raf_similarity_metric \
      --kb_path $kb_path | tee -a $log_file

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Completed: seq_len=$seq_len, pred_len=$pred_len" | tee -a $log_file
    echo "" >> $log_file
  done
done

echo "[$(date '+%Y-%m-%d %H:%M:%S')] All RAF-TimeMixer training completed for Weather!" | tee -a $log_file
