"""
Late Fusion Module for RAF (Retrieval Augmented Forecasting)
Combines retrieved patterns with query features
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LateFusionModule(nn.Module):
    """
    Late fusion mechanism to combine retrieved patterns with encoded features
    Implements multiple fusion strategies: attention, gating, and weighted averaging
    """
    def __init__(self, d_model, top_k=5, fusion_method='attention', dropout=0.1):
        super(LateFusionModule, self).__init__()
        self.d_model = d_model
        self.top_k = top_k
        self.fusion_method = fusion_method
        
        if fusion_method == 'attention':
            # Cross-attention between query and retrieved patterns
            self.query_proj = nn.Linear(d_model, d_model)
            self.key_proj = nn.Linear(d_model, d_model)
            self.value_proj = nn.Linear(d_model, d_model)
            self.out_proj = nn.Linear(d_model, d_model)
            self.dropout = nn.Dropout(dropout)
            
        elif fusion_method == 'gating':
            # Gating mechanism
            self.gate = nn.Sequential(
                nn.Linear(d_model * 2, d_model),
                nn.Sigmoid()
            )
            self.transform = nn.Linear(d_model, d_model)
            
        elif fusion_method == 'weighted':
            # Simple weighted averaging
            self.weight_net = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.ReLU(),
                nn.Linear(d_model // 2, 1),
                nn.Softmax(dim=1)
            )
        else:
            raise ValueError(f"Unknown fusion method: {fusion_method}")
    
    def attention_fusion(self, query_features, retrieved_features, similarity_scores):
        """
        Attention-based fusion
        
        Args:
            query_features: [B, T, D]
            retrieved_features: [B, k, T, D]
            similarity_scores: [B, k]
        Returns:
            fused_features: [B, T, D]
        """
        B, T, D = query_features.shape
        k = retrieved_features.size(1)
        T_kb = retrieved_features.size(2)
        
        # Handle seq_len mismatch first: resize retrieved features
        if T_kb != T:
            # Reshape and interpolate
            retrieved_features = retrieved_features.reshape(B * k, T_kb, D)  # [B*k, T_kb, D]
            retrieved_features = retrieved_features.permute(0, 2, 1)  # [B*k, D, T_kb]
            retrieved_features = F.interpolate(retrieved_features, size=T, mode='linear', align_corners=False)
            retrieved_features = retrieved_features.permute(0, 2, 1)  # [B*k, T, D]
            retrieved_features = retrieved_features.view(B, k, T, D)  # [B, k, T, D]
        
        # Project query, key, value
        Q = self.query_proj(query_features)  # [B, T, D]
        K = self.key_proj(retrieved_features.view(B * k, T, D))  # [B*k, T, D]
        V = self.value_proj(retrieved_features.view(B * k, T, D))  # [B*k, T, D]
        
        # Reshape for attention
        K = K.view(B, k, T, D)  # [B, k, T, D]
        V = V.view(B, k, T, D)  # [B, k, T, D]
        
        # Compute attention weights
        # Average over time dimension for simplicity
        Q_avg = Q.mean(dim=1, keepdim=True)  # [B, 1, D]
        K_avg = K.mean(dim=2)  # [B, k, D]
        
        # Attention scores
        attn_scores = torch.matmul(Q_avg, K_avg.transpose(-2, -1)) / (D ** 0.5)  # [B, 1, k]
        
        # Combine with similarity scores
        sim_scores = similarity_scores.unsqueeze(1)  # [B, 1, k]
        attn_scores = attn_scores + sim_scores * 0.5  # Blend
        
        attn_weights = F.softmax(attn_scores, dim=-1)  # [B, 1, k]
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to retrieve features
        V_avg = V.mean(dim=2)  # [B, k, D]
        attended = torch.matmul(attn_weights, V_avg)  # [B, 1, D]
        attended = attended.squeeze(1).unsqueeze(1).expand(-1, T, -1)  # [B, T, D]
        
        # Combine with query
        fused = self.out_proj(query_features + attended)
        
        return fused
    
    def gating_fusion(self, query_features, retrieved_features, similarity_scores):
        """
        Gating-based fusion
        
        Args:
            query_features: [B, T, D]
            retrieved_features: [B, k, T, D]
            similarity_scores: [B, k]
        Returns:
            fused_features: [B, T, D]
        """
        B, T, D = query_features.shape
        
        # Weighted average of retrieved patterns
        weights = F.softmax(similarity_scores, dim=-1)  # [B, k]
        weights = weights.unsqueeze(-1).unsqueeze(-1)  # [B, k, 1, 1]
        
        retrieved_avg = (retrieved_features * weights).sum(dim=1)  # [B, T_kb, D]
        
        # Handle seq_len mismatch: resize retrieved features to match query T
        if retrieved_avg.size(1) != T:
            # Interpolate along time dimension to match query seq_len
            retrieved_avg = retrieved_avg.permute(0, 2, 1)  # [B, D, T_kb]
            retrieved_avg = F.interpolate(retrieved_avg, size=T, mode='linear', align_corners=False)
            retrieved_avg = retrieved_avg.permute(0, 2, 1)  # [B, T, D]
        
        # Transform retrieved features
        retrieved_transformed = self.transform(retrieved_avg)  # [B, T, D]
        
        # Compute gate
        concat = torch.cat([query_features, retrieved_transformed], dim=-1)  # [B, T, 2D]
        gate_values = self.gate(concat)  # [B, T, D]
        
        # Gated fusion
        fused = gate_values * query_features + (1 - gate_values) * retrieved_transformed
        
        return fused
    
    def weighted_fusion(self, query_features, retrieved_features, similarity_scores):
        """
        Simple weighted averaging fusion
        
        Args:
            query_features: [B, T, D]
            retrieved_features: [B, k, T, D]
            similarity_scores: [B, k]
        Returns:
            fused_features: [B, T, D]
        """
        B, T, D = query_features.shape
        
        # Use similarity scores as weights
        weights = F.softmax(similarity_scores, dim=-1)  # [B, k]
        weights = weights.unsqueeze(-1).unsqueeze(-1)  # [B, k, 1, 1]
        
        # Weighted sum of retrieved features
        fused = (retrieved_features * weights).sum(dim=1)  # [B, T_kb, D]
        
        # Handle seq_len mismatch: resize to match query T
        if fused.size(1) != T:
            fused = fused.permute(0, 2, 1)  # [B, D, T_kb]
            fused = F.interpolate(fused, size=T, mode='linear', align_corners=False)
            fused = fused.permute(0, 2, 1)  # [B, T, D]
        
        return fused
        
        # Weighted average
        retrieved_avg = (retrieved_features * weights).sum(dim=1)  # [B, T, D]
        
        # Simple average with query
        fused = (query_features + retrieved_avg) / 2.0
        
        return fused
    
    def forward(self, query_features, retrieved_features, similarity_scores):
        """
        Forward pass: fuse query with retrieved patterns
        
        Args:
            query_features: [B, T, D] - encoded query features
            retrieved_features: [B, k, T, D] - retrieved similar patterns
            similarity_scores: [B, k] - similarity scores
        Returns:
            fused_features: [B, T, D] - augmented features
        """
        # Handle case when no retrieval is available
        if retrieved_features is None or similarity_scores is None:
            return query_features
        
        if self.fusion_method == 'attention':
            return self.attention_fusion(query_features, retrieved_features, similarity_scores)
        elif self.fusion_method == 'gating':
            return self.gating_fusion(query_features, retrieved_features, similarity_scores)
        elif self.fusion_method == 'weighted':
            return self.weighted_fusion(query_features, retrieved_features, similarity_scores)
        else:
            return query_features


class MultiScaleLateFusion(nn.Module):
    """
    Late fusion for multi-scale features (compatible with TimeMixer's multi-scale architecture)
    """
    def __init__(self, d_model, num_scales=3, top_k=5, fusion_method='attention', dropout=0.1):
        super(MultiScaleLateFusion, self).__init__()
        self.num_scales = num_scales
        
        # Create fusion module for each scale
        self.fusion_modules = nn.ModuleList([
            LateFusionModule(d_model, top_k, fusion_method, dropout)
            for _ in range(num_scales)
        ])
    
    def forward(self, query_features_list, retrieved_features_list, similarity_scores):
        """
        Apply fusion to each scale
        
        Args:
            query_features_list: List of [B, T_i, D] - multi-scale query features
            retrieved_features_list: List of [B, k, T_i, D] - multi-scale retrieved features
            similarity_scores: [B, k] - similarity scores (shared across scales)
        Returns:
            fused_features_list: List of [B, T_i, D] - augmented multi-scale features
        """
        fused_features_list = []
        
        for i, (query_feat, fusion_module) in enumerate(zip(query_features_list, self.fusion_modules)):
            if retrieved_features_list is not None and i < len(retrieved_features_list):
                retrieved_feat = retrieved_features_list[i]
                fused_feat = fusion_module(query_feat, retrieved_feat, similarity_scores)
            else:
                fused_feat = query_feat
            
            fused_features_list.append(fused_feat)
        
        return fused_features_list
