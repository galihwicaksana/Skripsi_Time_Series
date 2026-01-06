#!/bin/bash
# Test RAF using existing baseline checkpoint (no retraining needed!)
# This is more efficient since late fusion only affects inference

source /mnt/extended-home/galih/miniconda3/bin/activate timeMixer
cd /mnt/extended-home/galih/Time-Series-Library
export PYTHONPATH="/mnt/extended-home/galih/Time-Series-Library:$PYTHONPATH"

export CUDA_VISIBLE_DEVICES=0

# RAF parameters
use_raf=1
raf_fusion_method='gating'
raf_similarity_metric='cosine'

# Dataset selection
if [ "$1" == "weather" ]; then
    dataset="custom"
    data_path="weather.csv"
    root_path="./dataset/weather/"
    enc_in=21
    c_out=21
    d_model=16
    d_ff=32
    batch_size=128
    kb_path="./checkpoints/knowledge_bases/weather_kb.pkl"
    raf_d_embed=16
    baseline_checkpoint_pattern="long_term_forecast_Weather_*_TimeMixer_custom"
elif [ "$1" == "traffic" ]; then
    dataset="custom"
    data_path="traffic.csv"
    root_path="./dataset/traffic/"
    enc_in=862
    c_out=862
    d_model=32
    d_ff=64
    batch_size=8
    kb_path="./checkpoints/knowledge_bases/traffic_kb.pkl"
    raf_d_embed=32
    baseline_checkpoint_pattern="long_term_forecast_Traffic_*_TimeMixer_custom"
else
    echo "Usage: bash test_raf_with_baseline.sh [weather|traffic]"
    exit 1
fi

echo "=========================================="
echo "  RAF Testing with Baseline Checkpoint"
echo "  Dataset: $1"
echo "  Strategy: Load baseline → Apply RAF → Test"
echo "=========================================="
echo ""

# Find baseline checkpoints
echo "Looking for baseline checkpoints..."
baseline_dirs=$(find checkpoints/ -maxdepth 1 -type d -name "${baseline_checkpoint_pattern}*" | grep -v RAF)

if [ -z "$baseline_dirs" ]; then
    echo "❌ No baseline checkpoints found!"
    echo "Please train baseline first or current approach will train new model."
    exit 1
fi

echo "✓ Found baseline checkpoints:"
echo "$baseline_dirs" | head -5
echo ""

# For testing, we just need to:
# 1. Load baseline checkpoint
# 2. Enable RAF
# 3. Run testing only (--is_training 0)

log_file="RAF_testing_with_baseline_${1}.log"
echo "RAF Testing Results - Using Baseline Checkpoint" > $log_file
echo "Dataset: $1" >> $log_file
echo "Approach: Load trained baseline + Apply RAF at inference" >> $log_file
echo "========================================" >> $log_file
echo "" >> $log_file

# Test each configuration
for seq_len in 48 96 168 336; do
  for pred_len in 96 192 336 720; do
    
    # Adaptive top_k
    if [ $pred_len -eq 96 ]; then
      raf_top_k=3
    elif [ $pred_len -eq 192 ]; then
      raf_top_k=5
    elif [ $pred_len -eq 336 ]; then
      raf_top_k=7
    else
      raf_top_k=10
    fi
    
    # Find corresponding baseline checkpoint
    baseline_setting="${1}_${seq_len}_${pred_len}_TimeMixer"
    checkpoint_dir=$(find checkpoints/ -maxdepth 1 -type d -name "*${baseline_setting}*" | grep -v RAF | head -1)
    
    if [ -z "$checkpoint_dir" ]; then
      echo "[$(date)] ⚠️ No baseline checkpoint for seq=${seq_len} pred=${pred_len}, skipping..." | tee -a $log_file
      continue
    fi
    
    checkpoint_path="${checkpoint_dir}/checkpoint.pth"
    
    if [ ! -f "$checkpoint_path" ]; then
      echo "[$(date)] ⚠️ Checkpoint not found: $checkpoint_path" | tee -a $log_file
      continue
    fi
    
    echo "[$(date)] Testing: seq_len=$seq_len, pred_len=$pred_len, top_k=$raf_top_k" | tee -a $log_file
    echo "  Using baseline: $checkpoint_dir" | tee -a $log_file
    
    model_id="${1}_${seq_len}_${pred_len}_RAF_from_baseline"
    
    # Run testing with RAF enabled
    python -u run.py \
      --task_name long_term_forecast \
      --is_training 0 \
      --root_path $root_path \
      --data_path $data_path \
      --model_id $model_id \
      --model TimeMixer_RAF \
      --data $dataset \
      --features M \
      --seq_len $seq_len \
      --label_len 0 \
      --pred_len $pred_len \
      --e_layers 2 \
      --enc_in $enc_in \
      --dec_in $enc_in \
      --c_out $c_out \
      --des Exp \
      --itr 1 \
      --d_model $d_model \
      --d_ff $d_ff \
      --batch_size $batch_size \
      --down_sampling_layers 3 \
      --down_sampling_method avg \
      --down_sampling_window 2 \
      --use_raf $use_raf \
      --raf_top_k $raf_top_k \
      --raf_d_embed $raf_d_embed \
      --raf_fusion_method $raf_fusion_method \
      --raf_similarity_metric $raf_similarity_metric \
      --kb_path $kb_path \
      --checkpoint_path $checkpoint_path | tee -a $log_file
    
    echo "[$(date)] ✓ Completed: seq_len=$seq_len, pred_len=$pred_len" | tee -a $log_file
    echo "" >> $log_file
  done
done

echo "[$(date)] All RAF testing completed!" | tee -a $log_file
echo "Results saved to: $log_file" | tee -a $log_file
