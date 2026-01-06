#!/bin/bash
# OPTIMIZED RAF Script for Traffic Dataset
# Changes from v1:
# 1. Increased raf_top_k (5× larger for 862 features)
# 2. Increased kb_size to 20000 (4× larger)
# 3. Added fusion method ensemble option
# All changes are RAF-specific - backbone hyperparameters UNCHANGED for fair comparison

source /mnt/extended-home/galih/miniconda3/bin/activate timeMixer

# Set working directory and Python path
cd /mnt/extended-home/galih/Time-Series-Library
export PYTHONPATH="/mnt/extended-home/galih/Time-Series-Library:$PYTHONPATH"

export CUDA_VISIBLE_DEVICES=2

model_name=TimeMixer_RAF

# Backbone hyperparameters - UNCHANGED for fair comparison
e_layers=2
down_sampling_layers=3
down_sampling_window=2
learning_rate=0.01
d_model=32
d_ff=64
train_epochs=10
patience=10
batch_size=8

# RAF-specific hyperparameters - OPTIMIZED for 862 features
use_raf=1
raf_d_embed=32  # Keep same as d_model for fair comparison
raf_fusion_method='attention'  # Changed from 'gating' - more powerful for high-dim
raf_similarity_metric='cosine'
kb_path='./checkpoints/knowledge_bases/traffic_kb_v2_10k.pkl'  # New KB file (reduced to 10k for GPU memory)

# Log file
log_file="long_term_forecast_RAF_Traffic_v2_optimized_results.log"
echo "RAF-TimeMixer Training Results - Traffic Dataset (OPTIMIZED v2)" > $log_file
echo "Optimization: Increased top_k (5×) and KB size (4×) for 862 features" >> $log_file
echo "Fair Comparison: Backbone hyperparameters UNCHANGED" >> $log_file
echo "========================================" >> $log_file
echo "Hyperparameters:" >> $log_file
echo "  Learning Rate: $learning_rate (unchanged)" >> $log_file
echo "  Epochs: $train_epochs (unchanged)" >> $log_file
echo "  Patience: $patience (unchanged)" >> $log_file
echo "  Batch Size: $batch_size (unchanged)" >> $log_file
echo "  d_model: $d_model (unchanged)" >> $log_file
echo "  d_ff: $d_ff (unchanged)" >> $log_file
echo "RAF parameters (OPTIMIZED):" >> $log_file
echo "  use_raf: $use_raf" >> $log_file
echo "  raf_top_k: ADAPTIVE (15/25/35/50) - INCREASED 5×" >> $log_file
echo "  raf_d_embed: $raf_d_embed (unchanged - fair comparison)" >> $log_file
echo "  raf_fusion_method: $raf_fusion_method (changed: gating → attention)" >> $log_file
echo "  KB size: 10000 patterns - INCREASED 2× (reduced from 20k for GPU memory)" >> $log_file
echo "========================================" >> $log_file
echo "" >> $log_file

# Build LARGER knowledge base (10k patterns for 862 features - reduced to fit GPU memory)
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Building OPTIMIZED Knowledge Base (10000 patterns)..." | tee -a $log_file
python -u scripts/build_kb_traffic.py \
  --model $model_name \
  --max_kb_size 10000 \
  --data custom \
  --root_path ./dataset/traffic/ \
  --data_path traffic.csv \
  --seq_len 96 \
  --pred_len 96 \
  --enc_in 862 \
  --c_out 862 \
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
  --kb_path $kb_path 2>&1 | tee -a $log_file

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Knowledge Base built successfully!" | tee -a $log_file
echo "" >> $log_file

# Run experiments - 4 seq_len × 4 pred_len = 16 experiments
for seq_len in 48 96 168 336; do
  for pred_len in 96 192 336 720; do
    
    # OPTIMIZED Adaptive top_k - INCREASED 5× for 862 features
    # Rationale: More features need more retrieved patterns for diversity
    if [ $pred_len -eq 96 ]; then
      raf_top_k=15  # Was 3 → now 5× larger
    elif [ $pred_len -eq 192 ]; then
      raf_top_k=25  # Was 5 → now 5× larger
    elif [ $pred_len -eq 336 ]; then
      raf_top_k=35  # Was 7 → now 5× larger
    else  # 720
      raf_top_k=50  # Was 10 → now 5× larger
    fi
    
    model_id="Traffic_${seq_len}_${pred_len}_RAF_v2"

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Running training: seq_len=$seq_len, pred_len=$pred_len, top_k=$raf_top_k" | tee -a $log_file

    python -u run.py \
      --task_name long_term_forecast \
      --is_training 1 \
      --root_path ./dataset/traffic/ \
      --data_path traffic.csv \
      --model_id $model_id \
      --model $model_name \
      --data custom \
      --features M \
      --seq_len $seq_len \
      --label_len 0 \
      --pred_len $pred_len \
      --e_layers $e_layers \
      --enc_in 862 \
      --dec_in 862 \
      --c_out 862 \
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
      --kb_path $kb_path 2>&1 | tee -a $log_file

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Completed: seq_len=$seq_len, pred_len=$pred_len" | tee -a $log_file
    echo "" >> $log_file
  done
done

echo "[$(date '+%Y-%m-%d %H:%M:%S')] All OPTIMIZED RAF-TimeMixer training completed for Traffic!" | tee -a $log_file
echo "Check results in result_long_term_forecast.txt" | tee -a $log_file
