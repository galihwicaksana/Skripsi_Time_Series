"""
TimeMixer with RAF (Retrieval Augmented Forecasting) - Late Fusion Implementation
Extends original TimeMixer with retrieval-augmented capabilities
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Autoformer_EncDec import series_decomp
from layers.Embed import DataEmbedding_wo_pos
from layers.StandardNorm import Normalize
from layers.Retrieval import RetrievalModule
from layers.RAFFusion import LateFusionModule

# Import original TimeMixer components
from models.TimeMixer import (
    DFT_series_decomp,
    MultiScaleSeasonMixing,
    MultiScaleTrendMixing,
    PastDecomposableMixing
)


class Model(nn.Module):
    """
    RAF-enhanced TimeMixer with Late Fusion
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.down_sampling_window = configs.down_sampling_window
        self.channel_independence = configs.channel_independence
        
        # RAF-specific parameters
        self.use_raf = getattr(configs, 'use_raf', True)
        self.raf_top_k = getattr(configs, 'raf_top_k', 5)
        self.raf_d_embed = getattr(configs, 'raf_d_embed', 128)
        self.raf_fusion_method = getattr(configs, 'raf_fusion_method', 'attention')
        self.raf_similarity_metric = getattr(configs, 'raf_similarity_metric', 'cosine')
        
        # Original TimeMixer components
        self.pdm_blocks = nn.ModuleList([PastDecomposableMixing(configs)
                                         for _ in range(configs.e_layers)])

        self.preprocess = series_decomp(configs.moving_avg)
        self.enc_in = configs.enc_in

        if self.channel_independence:
            self.enc_embedding = DataEmbedding_wo_pos(1, configs.d_model, configs.embed, configs.freq,
                                                      configs.dropout)
        else:
            self.enc_embedding = DataEmbedding_wo_pos(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                                      configs.dropout)

        self.layer = configs.e_layers

        self.normalize_layers = torch.nn.ModuleList(
            [
                Normalize(self.configs.enc_in, affine=True, non_norm=True if configs.use_norm == 0 else False)
                for i in range(configs.down_sampling_layers + 1)
            ]
        )

        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.predict_layers = torch.nn.ModuleList(
                [
                    torch.nn.Linear(
                        configs.seq_len // (configs.down_sampling_window ** i),
                        configs.pred_len,
                    )
                    for i in range(configs.down_sampling_layers + 1)
                ]
            )

            if self.channel_independence:
                self.projection_layer = nn.Linear(
                    configs.d_model, 1, bias=True)
            else:
                self.projection_layer = nn.Linear(
                    configs.d_model, configs.c_out, bias=True)

                self.out_res_layers = torch.nn.ModuleList([
                    torch.nn.Linear(
                        configs.seq_len // (configs.down_sampling_window ** i),
                        configs.seq_len // (configs.down_sampling_window ** i),
                    )
                    for i in range(configs.down_sampling_layers + 1)
                ])

                self.regression_layers = torch.nn.ModuleList(
                    [
                        torch.nn.Linear(
                            configs.seq_len // (configs.down_sampling_window ** i),
                            configs.pred_len,
                        )
                        for i in range(configs.down_sampling_layers + 1)
                    ]
                )

        # RAF components
        if self.use_raf:
            # Retrieval module
            self.retrieval_module = RetrievalModule(
                seq_len=configs.seq_len,
                d_model=configs.d_model,
                d_embed=self.raf_d_embed,
                top_k=self.raf_top_k,
                similarity_metric=self.raf_similarity_metric,
                dropout=configs.dropout
            )
            
            # Late fusion module - applied after PDM blocks
            self.fusion_module = LateFusionModule(
                d_model=configs.d_model,
                top_k=self.raf_top_k,
                fusion_method=self.raf_fusion_method,
                dropout=configs.dropout
            )
        
        if self.task_name == 'imputation' or self.task_name == 'anomaly_detection':
            if self.channel_independence:
                self.projection_layer = nn.Linear(
                    configs.d_model, 1, bias=True)
            else:
                self.projection_layer = nn.Linear(
                    configs.d_model, configs.c_out, bias=True)
        if self.task_name == 'classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(
                configs.d_model * configs.seq_len, configs.num_class)

    def set_knowledge_base(self, kb_embeddings, kb_features, kb_targets):
        """
        Set knowledge base for retrieval
        
        Args:
            kb_embeddings: [N, d_embed]
            kb_features: [N, T, D]
            kb_targets: [N, pred_len, C]
        """
        if self.use_raf:
            self.retrieval_module.set_knowledge_base(kb_embeddings, kb_features, kb_targets)
    
    def extract_features_for_kb(self, x_enc):
        """
        Extract features and embeddings for building knowledge base
        Used during KB construction phase
        
        Args:
            x_enc: [B, T, N]
        Returns:
            embeddings: [B, d_embed]
            features: [B, T, D]
        """
        B, T, N = x_enc.size()
        
        # Multi-scale processing
        x_enc, _ = self.__multi_scale_process_inputs(x_enc, None)
        
        x_list = []
        for i, x in enumerate(x_enc):
            B, T, N = x.size()
            x = self.normalize_layers[i](x, 'norm')
            if self.channel_independence:
                x = x.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
            x_list.append(x)
        
        # Embedding
        enc_out_list = []
        x_list = self.pre_enc(x_list)
        for i, x in enumerate(x_list[0] if not self.channel_independence else x_list):
            enc_out = self.enc_embedding(x, None)
            enc_out_list.append(enc_out)
        
        # Pass through PDM blocks
        for i in range(self.layer):
            enc_out_list = self.pdm_blocks[i](enc_out_list)
        
        # Use first scale for embedding
        features = enc_out_list[0]  # [B, T, D]
        
        # Generate embeddings
        if self.use_raf:
            embeddings = self.retrieval_module.query_encoder(features)  # [B, d_embed]
        else:
            # Simple average pooling if RAF not enabled
            embeddings = features.mean(dim=1)  # [B, D]
        
        return embeddings, features

    def out_projection(self, dec_out, i, out_res):
        dec_out = self.projection_layer(dec_out)
        out_res = out_res.permute(0, 2, 1)
        out_res = self.out_res_layers[i](out_res)
        out_res = self.regression_layers[i](out_res).permute(0, 2, 1)
        dec_out = dec_out + out_res
        return dec_out

    def pre_enc(self, x_list):
        if self.channel_independence:
            return (x_list, None)
        else:
            out1_list = []
            out2_list = []
            for x in x_list:
                x_1, x_2 = self.preprocess(x)
                out1_list.append(x_1)
                out2_list.append(x_2)
            return (out1_list, out2_list)

    def __multi_scale_process_inputs(self, x_enc, x_mark_enc):
        if self.configs.down_sampling_method == 'max':
            down_pool = torch.nn.MaxPool1d(self.configs.down_sampling_window, return_indices=False)
        elif self.configs.down_sampling_method == 'avg':
            down_pool = torch.nn.AvgPool1d(self.configs.down_sampling_window)
        elif self.configs.down_sampling_method == 'conv':
            padding = 1 if torch.__version__ >= '1.5.0' else 2
            down_pool = nn.Conv1d(in_channels=self.configs.enc_in, out_channels=self.configs.enc_in,
                                  kernel_size=3, padding=padding,
                                  stride=self.configs.down_sampling_window,
                                  padding_mode='circular',
                                  bias=False)
        else:
            return x_enc, x_mark_enc
        # B,T,C -> B,C,T
        x_enc = x_enc.permute(0, 2, 1)

        x_enc_ori = x_enc
        x_mark_enc_mark_ori = x_mark_enc

        x_enc_sampling_list = []
        x_mark_sampling_list = []
        x_enc_sampling_list.append(x_enc.permute(0, 2, 1))
        x_mark_sampling_list.append(x_mark_enc)

        for i in range(self.configs.down_sampling_layers):
            x_enc_sampling = down_pool(x_enc_ori)

            x_enc_sampling_list.append(x_enc_sampling.permute(0, 2, 1))
            x_enc_ori = x_enc_sampling

            if x_mark_enc_mark_ori is not None:
                x_mark_sampling_list.append(x_mark_enc_mark_ori[:, ::self.configs.down_sampling_window, :])
                x_mark_enc_mark_ori = x_mark_enc_mark_ori[:, ::self.configs.down_sampling_window, :]

        return x_enc_sampling_list, x_mark_sampling_list

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        """
        Forecasting with RAF late fusion
        """
        B, T, N = x_enc.size()
        
        # Multi-scale processing
        x_enc, x_mark_enc = self.__multi_scale_process_inputs(x_enc, x_mark_enc)

        x_list = []
        x_mark_list = []
        if x_mark_enc is not None:
            for i, x, x_mark in zip(range(len(x_enc)), x_enc, x_mark_enc):
                B, T, N = x.size()
                x = self.normalize_layers[i](x, 'norm')
                if self.channel_independence:
                    x = x.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
                    x_list.append(x)
                    x_mark = x_mark.repeat(N, 1, 1)
                    x_mark_list.append(x_mark)
                else:
                    x_list.append(x)
                    x_mark_list.append(x_mark)
        else:
            for i, x in zip(range(len(x_enc)), x_enc, ):
                B, T, N = x.size()
                x = self.normalize_layers[i](x, 'norm')
                if self.channel_independence:
                    x = x.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
                x_list.append(x)

        # Embedding
        enc_out_list = []
        x_list = self.pre_enc(x_list)
        if x_mark_enc is not None:
            for i, x, x_mark in zip(range(len(x_list[0])), x_list[0], x_mark_list):
                enc_out = self.enc_embedding(x, x_mark)
                enc_out_list.append(enc_out)
        else:
            for i, x in zip(range(len(x_list[0])), x_list[0]):
                enc_out = self.enc_embedding(x, None)
                enc_out_list.append(enc_out)

        # Past Decomposable Mixing as encoder for past
        for i in range(self.layer):
            enc_out_list = self.pdm_blocks[i](enc_out_list)

        # 🔥 RAF LATE FUSION: Apply retrieval and fusion after PDM blocks
        if self.use_raf and self.training == False:  # Only during inference
            # Clear GPU cache before retrieval to free memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Use first scale features for retrieval
            query_features = enc_out_list[0]  # [B, T, D]
            
            # Retrieve similar patterns
            retrieved_features, retrieved_targets, similarity_scores = \
                self.retrieval_module(query_features)
            
            # Apply late fusion
            if retrieved_features is not None:
                enhanced_features = self.fusion_module(
                    query_features, 
                    retrieved_features, 
                    similarity_scores
                )
                enc_out_list[0] = enhanced_features
                
                # Clear intermediate results to save memory
                del retrieved_features, retrieved_targets, similarity_scores
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # Future Multipredictor Mixing as decoder for future
        dec_out_list = self.future_multi_mixing(B, enc_out_list, x_list)

        dec_out = torch.stack(dec_out_list, dim=-1).sum(-1)
        dec_out = self.normalize_layers[0](dec_out, 'denorm')
        return dec_out

    def future_multi_mixing(self, B, enc_out_list, x_list):
        dec_out_list = []
        if self.channel_independence:
            x_list = x_list[0]
            for i, enc_out in zip(range(len(x_list)), enc_out_list):
                dec_out = self.predict_layers[i](enc_out.permute(0, 2, 1)).permute(
                    0, 2, 1)  # align temporal dimension
                dec_out = self.projection_layer(dec_out)
                dec_out = dec_out.reshape(B, self.configs.c_out, self.pred_len).permute(0, 2, 1).contiguous()
                dec_out_list.append(dec_out)

        else:
            for i, enc_out, out_res in zip(range(len(x_list[0])), enc_out_list, x_list[1]):
                dec_out = self.predict_layers[i](enc_out.permute(0, 2, 1)).permute(
                    0, 2, 1)  # align temporal dimension
                dec_out = self.out_projection(dec_out, i, out_res)
                dec_out_list.append(dec_out)

        return dec_out_list

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]
        return None
