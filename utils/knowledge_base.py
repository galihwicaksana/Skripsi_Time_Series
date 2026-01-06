"""
Knowledge Base Manager for RAF (Retrieval Augmented Forecasting)
Handles storage, indexing, and efficient retrieval of historical patterns
"""

import torch
import numpy as np
import pickle
import os
from pathlib import Path


class KnowledgeBase:
    """
    Knowledge Base for storing and retrieving historical time series patterns
    """
    def __init__(self, max_size=1000, device='cuda'):
        """
        Args:
            max_size: Maximum number of patterns to store
            device: Device for tensor operations
        """
        self.max_size = max_size
        self.device = device
        
        # Storage
        self.embeddings = []      # Query embeddings
        self.features = []        # Encoder features
        self.targets = []         # Target sequences
        self.metadata = []        # Additional metadata (optional)
        
        self.is_built = False
    
    def add(self, embedding, feature, target, metadata=None):
        """
        Add a single pattern to the knowledge base
        
        Args:
            embedding: [d_embed] - query embedding
            feature: [T, D] - encoder features
            target: [pred_len, C] - target sequence
            metadata: dict - optional metadata
        """
        # Convert to CPU for storage
        if isinstance(embedding, torch.Tensor):
            embedding = embedding.detach().cpu()
        if isinstance(feature, torch.Tensor):
            feature = feature.detach().cpu()
        if isinstance(target, torch.Tensor):
            target = target.detach().cpu()
        
        self.embeddings.append(embedding)
        self.features.append(feature)
        self.targets.append(target)
        self.metadata.append(metadata if metadata is not None else {})
        
        # Maintain max size
        if len(self.embeddings) > self.max_size:
            self.embeddings.pop(0)
            self.features.pop(0)
            self.targets.pop(0)
            self.metadata.pop(0)
    
    def add_batch(self, embeddings, features, targets, metadata_list=None):
        """
        Add a batch of patterns to the knowledge base
        
        Args:
            embeddings: [B, d_embed]
            features: [B, T, D]
            targets: [B, pred_len, C]
            metadata_list: List of metadata dicts
        """
        B = embeddings.size(0)
        
        for i in range(B):
            meta = metadata_list[i] if metadata_list is not None else None
            self.add(embeddings[i], features[i], targets[i], meta)
    
    def build(self):
        """
        Build the knowledge base index for efficient retrieval
        Converts lists to tensors
        """
        if len(self.embeddings) == 0:
            print("Warning: Knowledge base is empty!")
            return
        
        # Stack all embeddings, features, targets
        self.embeddings_tensor = torch.stack(self.embeddings).to(self.device)  # [N, d_embed]
        self.features_tensor = torch.stack(self.features).to(self.device)  # [N, T, D]
        self.targets_tensor = torch.stack(self.targets).to(self.device)  # [N, pred_len, C]
        
        self.is_built = True
        print(f"Knowledge Base built with {len(self.embeddings)} patterns")
    
    def get_tensors(self):
        """
        Get tensor representations for retrieval module
        
        Returns:
            embeddings_tensor: [N, d_embed]
            features_tensor: [N, T, D]
            targets_tensor: [N, pred_len, C]
        """
        if not self.is_built:
            self.build()
        
        return self.embeddings_tensor, self.features_tensor, self.targets_tensor
    
    def save(self, save_path):
        """
        Save knowledge base to disk
        
        Args:
            save_path: Path to save file
        """
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        
        kb_data = {
            'embeddings': self.embeddings,
            'features': self.features,
            'targets': self.targets,
            'metadata': self.metadata,
            'max_size': self.max_size,
        }
        
        with open(save_path, 'wb') as f:
            pickle.dump(kb_data, f)
        
        print(f"Knowledge Base saved to {save_path}")
    
    def load(self, load_path):
        """
        Load knowledge base from disk
        
        Args:
            load_path: Path to load file
        """
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"Knowledge base file not found: {load_path}")
        
        with open(load_path, 'rb') as f:
            kb_data = pickle.load(f)
        
        self.embeddings = kb_data['embeddings']
        self.features = kb_data['features']
        self.targets = kb_data['targets']
        self.metadata = kb_data['metadata']
        self.max_size = kb_data['max_size']
        
        self.build()
        
        print(f"Knowledge Base loaded from {load_path} with {len(self.embeddings)} patterns")
    
    def clear(self):
        """Clear all stored patterns"""
        self.embeddings = []
        self.features = []
        self.targets = []
        self.metadata = []
        self.is_built = False
    
    def __len__(self):
        """Return number of patterns in KB"""
        return len(self.embeddings)
    
    def __repr__(self):
        return f"KnowledgeBase(size={len(self)}, max_size={self.max_size}, built={self.is_built})"


class KnowledgeBaseBuilder:
    """
    Helper class to build knowledge base from a dataset
    """
    def __init__(self, model, data_loader, device='cuda', max_size=1000):
        """
        Args:
            model: Model with query encoder (RAF-enabled model)
            data_loader: DataLoader for the dataset
            device: Device for computation
            max_size: Maximum KB size
        """
        self.model = model
        self.data_loader = data_loader
        self.device = device
        self.max_size = max_size
        self.kb = KnowledgeBase(max_size=max_size, device=device)
    
    def build_from_data(self, num_samples=None):
        """
        Build knowledge base from data loader
        
        Args:
            num_samples: Maximum number of samples to use (None = use all)
        Returns:
            kb: Built KnowledgeBase object
        """
        self.model.eval()
        
        total_samples = 0
        with torch.no_grad():
            for batch_idx, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(self.data_loader):
                if num_samples is not None and total_samples >= num_samples:
                    break
                
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                
                # Extract features and embeddings from model
                # This requires the model to have a method to extract intermediate features
                if hasattr(self.model, 'extract_features_for_kb'):
                    embeddings, features = self.model.extract_features_for_kb(batch_x)
                    
                    # Add to KB
                    self.kb.add_batch(embeddings, features, batch_y[:, -self.model.pred_len:, :])
                    
                    total_samples += batch_x.size(0)
                else:
                    print("Warning: Model does not have 'extract_features_for_kb' method!")
                    break
                
                if (batch_idx + 1) % 10 == 0:
                    print(f"Processed {total_samples} samples...")
        
        # Build index
        self.kb.build()
        
        return self.kb
    
    def save_kb(self, save_path):
        """Save the built knowledge base"""
        self.kb.save(save_path)
    
    def load_kb(self, load_path):
        """Load a knowledge base"""
        self.kb.load(load_path)
        return self.kb
