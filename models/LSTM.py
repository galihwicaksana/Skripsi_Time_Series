import torch
import torch.nn as nn
from layers.Embed import DataEmbedding


class Model(nn.Module):
    """Simple LSTM baseline integrated with existing experiment pipeline.

    Supports tasks: long_term_forecast, short_term_forecast, imputation,
    anomaly_detection, classification.
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.configs = configs

        d_model = configs.d_model
        self.enc_embedding = DataEmbedding(configs.enc_in, d_model, configs.embed, configs.freq, configs.dropout)
        # For decoder input embedding (used only in forecasting tasks)
        self.dec_embedding = DataEmbedding(configs.dec_in, d_model, configs.embed, configs.freq, configs.dropout)

        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=configs.e_layers,
            batch_first=True,
            dropout=configs.dropout if configs.e_layers > 1 else 0.0
        )

        # Projection heads per task
        if self.task_name in ['long_term_forecast', 'short_term_forecast']:
            self.projection = nn.Linear(d_model, configs.c_out, bias=True)
        elif self.task_name in ['imputation', 'anomaly_detection']:
            self.projection = nn.Linear(d_model, configs.c_out, bias=True)
        elif self.task_name == 'classification':
            self.projection = nn.Linear(d_model * configs.seq_len, configs.num_class)
            self.act = nn.GELU()
            self.dropout = nn.Dropout(configs.dropout)
        else:
            self.projection = nn.Linear(d_model, configs.c_out, bias=True)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        # Embed encoder sequence
        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # [B, seq_len, d_model]

        if self.task_name in ['long_term_forecast', 'short_term_forecast']:
            # Embed decoder (label tokens + zeros) and concatenate temporally
            dec_out = self.dec_embedding(x_dec, x_mark_dec)  # [B, label_len+pred_len, d_model]
            lstm_in = torch.cat([enc_out, dec_out], dim=1)  # [B, seq_len+label_len+pred_len, d_model]
            lstm_out, _ = self.lstm(lstm_in)
            # Take last pred_len time steps
            forecast_part = lstm_out[:, -self.pred_len:, :]  # [B, pred_len, d_model]
            out = self.projection(forecast_part)  # [B, pred_len, c_out]
            return out

        if self.task_name == 'imputation':
            lstm_out, _ = self.lstm(enc_out)
            return self.projection(lstm_out)  # [B, L, D]

        if self.task_name == 'anomaly_detection':
            lstm_out, _ = self.lstm(enc_out)
            return self.projection(lstm_out)  # [B, L, D]

        if self.task_name == 'classification':
            lstm_out, _ = self.lstm(enc_out)
            x = self.act(lstm_out)
            x = self.dropout(x)
            x = x.reshape(x.shape[0], -1)
            return self.projection(x)  # [B, num_class]

        # Default: behave like forecasting returning last pred_len
        lstm_out, _ = self.lstm(enc_out)
        forecast_part = lstm_out[:, -self.pred_len:, :]
        return self.projection(forecast_part)
