"""
Retrieval Module for RAF (Retrieval Augmented Forecasting)
Implements query encoding and similarity-based retrieval
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class QueryEncoder(nn.Module):
    """
    Lightweight encoder to generate embeddings for query sequences
    """
    def __init__(self, seq_len, d_model, d_embed=128, dropout=0.1):
        super(QueryEncoder, self).__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.d_embed = d_embed
        
        # Simple CNN-based encoder
        self.conv_layers = nn.Sequential(
            nn.Conv1d(d_model, d_embed * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(d_embed * 2, d_embed, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        
        # Alternative: Attention pooling
        self.attention_pool = nn.Sequential(
            nn.Linear(d_model, 1),
            nn.Softmax(dim=1)
        )
        
        self.use_attention = True
        
    def forward(self, x):
        """
        Args:
            x: [B, T, D] - encoded features from TimeMixer
        Returns:
            embedding: [B, d_embed] - query embedding for retrieval
        """
        if self.use_attention:
            # Attention-based pooling
            attn_weights = self.attention_pool(x)  # [B, T, 1]
            weighted = (x * attn_weights).sum(dim=1)  # [B, D]
            # Project to embedding space
            embedding = F.linear(weighted, 
                                 torch.randn(self.d_embed, self.d_model).to(x.device))
            embedding = F.normalize(embedding, p=2, dim=-1)
        else:
            # CNN-based encoding
            x = x.permute(0, 2, 1)  # [B, D, T]
            embedding = self.conv_layers(x).squeeze(-1)  # [B, d_embed]
            embedding = F.normalize(embedding, p=2, dim=-1)
        
        return embedding


class RetrievalModule(nn.Module):
    """
    Main retrieval module that searches for similar patterns
    """
    def __init__(self, seq_len, d_model, d_embed=128, top_k=5, 
                 similarity_metric='cosine', dropout=0.1):
        super(RetrievalModule, self).__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.d_embed = d_embed
        self.top_k = top_k
        self.similarity_metric = similarity_metric
        
        # Query encoder
        self.query_encoder = QueryEncoder(seq_len, d_model, d_embed, dropout)
        
        # Knowledge base will be set externally
        self.kb_embeddings = None  # [N, d_embed]
        self.kb_features = None    # [N, T, D]
        self.kb_targets = None     # [N, pred_len, C]
        
    def set_knowledge_base(self, kb_embeddings, kb_features, kb_targets):
        """
        Set the knowledge base for retrieval
        
        Args:
            kb_embeddings: [N, d_embed] - stored embeddings
            kb_features: [N, T, D] - stored encoder features
            kb_targets: [N, pred_len, C] - stored target sequences
        """
        self.kb_embeddings = kb_embeddings
        self.kb_features = kb_features
        self.kb_targets = kb_targets
        
    def compute_similarity(self, query_embed, kb_embed):
        """
        Compute similarity between query and knowledge base
        
        Args:
            query_embed: [B, d_embed]
            kb_embed: [N, d_embed]
        Returns:
            similarity: [B, N]
        """
        if self.similarity_metric == 'cosine':
            # Cosine similarity
            query_norm = F.normalize(query_embed, p=2, dim=-1)
            kb_norm = F.normalize(kb_embed, p=2, dim=-1)
            similarity = torch.matmul(query_norm, kb_norm.T)  # [B, N]
        elif self.similarity_metric == 'euclidean':
            # Negative euclidean distance (higher = more similar)
            query_expand = query_embed.unsqueeze(1)  # [B, 1, d_embed]
            kb_expand = kb_embed.unsqueeze(0)  # [1, N, d_embed]
            dist = torch.norm(query_expand - kb_expand, p=2, dim=-1)  # [B, N]
            similarity = -dist
        else:
            raise ValueError(f"Unknown similarity metric: {self.similarity_metric}")
        
        return similarity
    
    def retrieve(self, query_features):
        """
        Retrieve top-k similar patterns from knowledge base
        
        Args:
            query_features: [B, T, D] - encoded features
        Returns:
            retrieved_features: [B, k, T, D]
            retrieved_targets: [B, k, pred_len, C]
            similarity_scores: [B, k]
        """
        B = query_features.size(0)
        
        # Encode query
        query_embed = self.query_encoder(query_features)  # [B, d_embed]
        
        # Check if KB is available
        if self.kb_embeddings is None:
            # Return dummy values if KB not initialized
            return None, None, None
        
        # Compute similarity
        similarity = self.compute_similarity(query_embed, self.kb_embeddings)  # [B, N]
        
        # Get top-k
        top_k_scores, top_k_indices = torch.topk(similarity, k=self.top_k, dim=-1)  # [B, k]
        
        # Retrieve features and targets
        # Optimize: vectorized indexing to reduce CPU->GPU transfers
        kb_on_gpu = self.kb_features.device.type == 'cuda'
        
        if not kb_on_gpu:
            # KB is on CPU - retrieve and transfer in chunks to avoid OOM
            # For large KB (20k patterns × 862 features), transferring all to GPU at once causes OOM
            # Solution: Process per-batch sample instead of flattening
            
            retrieved_features_list = []
            retrieved_targets_list = []
            
            # Disable gradient for retrieval to save memory
            with torch.no_grad():
                for b in range(B):
                    # Get indices for this batch sample
                    sample_indices = top_k_indices[b].cpu()  # [k]
                    
                    # Retrieve from CPU KB
                    sample_features = self.kb_features[sample_indices]  # [k, T, D]
                    sample_targets = self.kb_targets[sample_indices]    # [k, pred_len, C]
                    
                    # Move to GPU (only k patterns, manageable)
                    # Use non_blocking for async transfer
                    sample_features = sample_features.to(query_features.device, non_blocking=True)
                    sample_targets = sample_targets.to(query_features.device, non_blocking=True)
                    
                    retrieved_features_list.append(sample_features)
                    retrieved_targets_list.append(sample_targets)
                
                # Stack to [B, k, T, D] and [B, k, pred_len, C]
                retrieved_features = torch.stack(retrieved_features_list, dim=0)
                retrieved_targets = torch.stack(retrieved_targets_list, dim=0)
        else:
            # KB already on GPU - direct vectorized indexing
            # Use advanced indexing for batch retrieval
            batch_indices = torch.arange(B, device=query_features.device).unsqueeze(1).expand(-1, self.top_k)
            retrieved_features = self.kb_features[top_k_indices]  # [B, k, T, D]
            retrieved_targets = self.kb_targets[top_k_indices]    # [B, k, pred_len, C]
        
        return retrieved_features, retrieved_targets, top_k_scores
    
    def forward(self, query_features):
        """
        Forward pass: encode query and retrieve similar patterns
        
        Args:
            query_features: [B, T, D]
        Returns:
            retrieved_features: [B, k, T, D]
            retrieved_targets: [B, k, pred_len, C]
            similarity_scores: [B, k]
        """
        return self.retrieve(query_features)
