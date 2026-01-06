#!/bin/bash
source /mnt/extended-home/galih/miniconda3/bin/activate timeMixer

# Set working directory and Python path
cd /mnt/extended-home/galih/Time-Series-Library
export PYTHONPATH="/mnt/extended-home/galih/Time-Series-Library:$PYTHONPATH"

export CUDA_VISIBLE_DEVICES=2

model_name=TimeMixer_RAF

# Original TimeMixer hyperparameters (EXACTLY matching baseline for fair comparison)
e_layers=2
down_sampling_layers=3
down_sampling_window=2
learning_rate=0.01
d_model=16
d_ff=32
train_epochs=10
patience=10
batch_size=128

# RAF-specific hyperparameters (OPTIMIZED - Phase 1)
use_raf=1
# Adaptive top_k will be set per pred_len below
raf_d_embed=16  # CHANGED: Match d_model for efficiency
raf_fusion_method='gating'  # CHANGED: More lightweight than attention
raf_similarity_metric='cosine'  # options: cosine, euclidean
kb_path='./checkpoints/knowledge_bases/etth1_kb.pkl'

# Log file untuk hasil training
log_file="long_term_forecast_timeMixer_RAF_ETTh1_optimized.log"
echo "RAF-TimeMixer Training Results - OPTIMIZED (Phase 1)" > $log_file
echo "Hyperparameters (matched with baseline):" >> $log_file
echo "  Learning Rate: $learning_rate" >> $log_file
echo "  Epochs: $train_epochs" >> $log_file
echo "  Patience: $patience" >> $log_file
echo "  Batch Size: $batch_size" >> $log_file
echo "RAF-specific parameters (OPTIMIZED):" >> $log_file
echo "  use_raf: $use_raf" >> $log_file
echo "  raf_top_k: ADAPTIVE (3/5/7/10 based on pred_len)" >> $log_file
echo "  raf_d_embed: $raf_d_embed (CHANGED: 128->16)" >> $log_file
echo "  raf_fusion_method: $raf_fusion_method (CHANGED: attention->gating)" >> $log_file
echo "  raf_similarity_metric: $raf_similarity_metric" >> $log_file
echo "  KB size: 5000 patterns" >> $log_file
echo "========================================" >> $log_file
echo "" >> $log_file

# First, build knowledge base from training data
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Building Knowledge Base (5000 patterns)..." | tee -a $log_file
python -u scripts/build_kb_etth1.py \
  --model $model_name \
  --max_kb_size 5000 \
  --data ETTh1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --seq_len 96 \
  --pred_len 96 \
  --e_layers $e_layers \
  --enc_in 7 \
  --c_out 7 \
  --d_model $d_model \
  --d_ff $d_ff \
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

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Knowledge Base built successfully!" | tee -a $log_file
echo "" >> $log_file

# Run experiments with same configurations as baseline
for seq_len in 96; do
  for pred_len in 96 192 336 720; do
    
    # Adaptive top_k based on pred_len (OPTIMIZATION)
    if [ $pred_len -eq 96 ]; then
      raf_top_k=3
    elif [ $pred_len -eq 192 ]; then
      raf_top_k=5
    elif [ $pred_len -eq 336 ]; then
      raf_top_k=7
    else  # 720
      raf_top_k=10
    fi
    
    model_id="ETTh1_${seq_len}_${pred_len}_RAF"

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Running training: seq_len=$seq_len, pred_len=$pred_len, top_k=$raf_top_k" | tee -a $log_file

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
      --label_len 0 \
      --pred_len $pred_len \
      --e_layers $e_layers \
      --enc_in 7 \
      --c_out 7 \
      --des 'Exp' \
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
    echo "========================================" >> $log_file
    echo "" >> $log_file

  done
done

echo "" | tee -a $log_file
echo "[$(date '+%Y-%m-%d %H:%M:%S')] All RAF-TimeMixer training completed for ETTh1!" | tee -a $log_file

# Print comparison summary
echo "" | tee -a $log_file
echo "========================================" | tee -a $log_file
echo "COMPARISON INSTRUCTIONS:" | tee -a $log_file
echo "Compare this log with: long_term_forecast_timeMixer_ETTh1_results.log" | tee -a $log_file
echo "Focus on seq_len=96 results for fair comparison" | tee -a $log_file
echo "Baseline results (from previous log):" | tee -a $log_file
echo "  96->96:  MSE=0.3780, MAE=0.3974" | tee -a $log_file
echo "  96->192: MSE=0.4395, MAE=0.4295" | tee -a $log_file
echo "  96->336: MSE=0.5000, MAE=0.4597" | tee -a $log_file
echo "  96->720: MSE=0.4750, MAE=0.4700" | tee -a $log_file
echo "========================================" | tee -a $log_file
