"""
Script to build Knowledge Base from Weather training data for RAF-TimeMixer
"""

import argparse
import os
import sys
import torch
import numpy as np
from tqdm import tqdm

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from data_provider.data_factory import data_provider
from utils.knowledge_base import KnowledgeBase

def build_knowledge_base(args):
    """Build knowledge base from training data"""
    print("="*50)
    print("Building Knowledge Base for RAF-TimeMixer")
    print("="*50)
    print(f"Dataset: {args.data}")
    print(f"Data path: {args.data_path}")
    print(f"Sequence length: {args.seq_len}")
    print(f"Prediction length: {args.pred_len}")
    print(f"Max KB size: {args.max_kb_size}")
    print(f"KB save path: {args.kb_path}")
    print("="*50)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load training data
    print("\nLoading training data...")
    train_data, train_loader = data_provider(args, 'train')
    print(f"Training data loaded: {len(train_data)} samples")
    
    # Initialize knowledge base
    kb = KnowledgeBase(max_size=args.max_kb_size, device=device)
    
    # Initialize model for feature extraction
    print("\nInitializing model for feature extraction...")
    from models.TimeMixer_RAF import Model
    
    model = Model(args).to(device)
    model.eval()
    
    # Extract features from training data
    print("\nExtracting features from training data...")
    total_samples = 0
    max_samples = min(args.max_kb_size, len(train_data))
    
    with torch.no_grad():
        for batch_idx, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(tqdm(train_loader)):
            if total_samples >= max_samples:
                break
            
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            
            # Extract features and embeddings
            try:
                embeddings, features = model.extract_features_for_kb(batch_x)
                
                # Get target sequences
                targets = batch_y[:, -args.pred_len:, :]  # [B, pred_len, C]
                
                # Add to KB
                kb.add_batch(embeddings, features, targets)
                
                total_samples += batch_x.size(0)
                
            except Exception as e:
                print(f"\nWarning: Error processing batch {batch_idx}: {e}")
                continue
    
    print(f"\nExtracted {total_samples} samples")
    
    # Build KB index
    print("\nBuilding KB index...")
    kb.build()
    print(f"Knowledge Base built with {len(kb)} patterns")
    
    # Save KB
    print(f"\nSaving KB to {args.kb_path}...")
    kb.save(args.kb_path)
    print(f"Knowledge Base saved to {args.kb_path}")
    
    print("\n" + "="*50)
    print("Knowledge Base built successfully!")
    print(f"Total patterns stored: {len(kb)}")
    print(f"Saved to: {args.kb_path}")
    print("="*50)
    
    return kb


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Build Knowledge Base for RAF-TimeMixer - Weather')
    
    # Basic config
    parser.add_argument('--model', type=str, default='TimeMixer_RAF')
    parser.add_argument('--task_name', type=str, default='long_term_forecast')
    parser.add_argument('--is_training', type=int, default=1)
    parser.add_argument('--data', type=str, default='custom')
    parser.add_argument('--root_path', type=str, default='./dataset/weather/')
    parser.add_argument('--data_path', type=str, default='weather.csv')
    parser.add_argument('--features', type=str, default='M')
    parser.add_argument('--target', type=str, default='OT')
    parser.add_argument('--freq', type=str, default='h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/')
    parser.add_argument('--batch_size', type=int, default=128)
    
    # Forecasting task
    parser.add_argument('--seq_len', type=int, default=96)
    parser.add_argument('--label_len', type=int, default=0)
    parser.add_argument('--pred_len', type=int, default=96)
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly')
    parser.add_argument('--inverse', action='store_true', default=False)
    parser.add_argument('--augmentation_ratio', type=int, default=0)
    parser.add_argument('--jitter', action='store_true', default=False)
    parser.add_argument('--scaling', action='store_true', default=False)
    parser.add_argument('--permutation', action='store_true', default=False)
    parser.add_argument('--randompermutation', action='store_true', default=False)
    parser.add_argument('--magwarp', action='store_true', default=False)
    parser.add_argument('--timewarp', action='store_true', default=False)
    parser.add_argument('--windowslice', action='store_true', default=False)
    parser.add_argument('--windowwarp', action='store_true', default=False)
    parser.add_argument('--rotation', action='store_true', default=False)
    parser.add_argument('--spawner', action='store_true', default=False)
    parser.add_argument('--dtwwarp', action='store_true', default=False)
    parser.add_argument('--shapedtwwarp', action='store_true', default=False)
    parser.add_argument('--wdba', action='store_true', default=False)
    parser.add_argument('--discdtw', action='store_true', default=False)
    parser.add_argument('--discsdtw', action='store_true', default=False)
    parser.add_argument('--extra_tag', type=str, default='')
    
    # Model define - Weather has 21 features
    parser.add_argument('--enc_in', type=int, default=21)
    parser.add_argument('--dec_in', type=int, default=21)
    parser.add_argument('--c_out', type=int, default=21)
    parser.add_argument('--d_model', type=int, default=16)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--e_layers', type=int, default=2)
    parser.add_argument('--d_layers', type=int, default=1)
    parser.add_argument('--d_ff', type=int, default=32)
    parser.add_argument('--moving_avg', type=int, default=25)
    parser.add_argument('--factor', type=int, default=1)
    parser.add_argument('--distil', default=True, action='store_false')
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--embed', type=str, default='timeF')
    parser.add_argument('--activation', type=str, default='gelu')
    parser.add_argument('--output_attention', action='store_true')
    
    # TimeMixer specific
    parser.add_argument('--down_sampling_layers', type=int, default=3)
    parser.add_argument('--down_sampling_method', type=str, default='avg')
    parser.add_argument('--down_sampling_window', type=int, default=2)
    parser.add_argument('--channel_independence', type=int, default=0)
    parser.add_argument('--decomp_method', type=str, default='moving_avg')
    parser.add_argument('--use_norm', type=int, default=1)
    parser.add_argument('--top_k', type=int, default=5)
    
    # RAF specific (optimized parameters)
    parser.add_argument('--use_raf', type=int, default=1)
    parser.add_argument('--raf_top_k', type=int, default=5)
    parser.add_argument('--raf_d_embed', type=int, default=16)
    parser.add_argument('--raf_fusion_method', type=str, default='gating')
    parser.add_argument('--raf_similarity_metric', type=str, default='cosine')
    
    # Data loader
    parser.add_argument('--num_workers', type=int, default=10)
    
    # Knowledge Base
    parser.add_argument('--kb_path', type=str, default='./checkpoints/knowledge_bases/weather_kb.pkl')
    parser.add_argument('--max_kb_size', type=int, default=5000)
    
    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True)
    parser.add_argument('--gpu', type=int, default=0)
    
    args = parser.parse_args()
    args.use_gpu = True if torch.cuda.is_available() else False
    
    # Build knowledge base
    build_knowledge_base(args)
