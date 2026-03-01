# utils/model.py
# Shared EMG_CNN model definition for AlterEgo Pipeline
# Used by: 7-train-model.py, 8-realtime-predict.py, 9-confusion-matrix.py
#
# KEY FIX: Uses AdaptiveAvgPool1d instead of a hardcoded flatten size,
# so the model works with any number of channels or timesteps.

import torch
import torch.nn as nn
import math

class EMG_CNN(nn.Module):
    """1D CNN for EMG classification. N-channel adaptive. (Standard/Edge Model)"""
    def __init__(self, input_channels, num_classes):
        super(EMG_CNN, self).__init__()
        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.relu = nn.ReLU()
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # x shape: (Batch, Time, Features) -> Permute to (Batch, Features, Time)
        x = x.permute(0, 2, 1)
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.adaptive_pool(x).squeeze(-1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

class EMG_LSTM(nn.Module):
    """LSTM (Long Short-Term Memory) for temporal sequence modeling. (Stronger Model)"""
    def __init__(self, input_channels, num_classes, hidden_size=128, num_layers=2):
        super(EMG_LSTM, self).__init__()
        # LSTM takes (Batch, Time, Features) directly
        self.lstm = nn.LSTM(input_channels, hidden_size, num_layers, batch_first=True, dropout=0.3)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x shape: (Batch, Time, Features)
        # LSTM output: (Batch, Time, Hidden)
        out, _ = self.lstm(x)
        # Take the last time step's output
        out = out[:, -1, :]
        out = self.fc(out)
        return out

class EMG_Transformer(nn.Module):
    """Transformer Encoder for EMG. (State-of-the-Art/Experimental)"""
    def __init__(self, input_channels, num_classes, d_model=64, nhead=4, num_layers=2):
        super(EMG_Transformer, self).__init__()
        self.embedding = nn.Linear(input_channels, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=128, dropout=0.3, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # x: (Batch, Time, Features)
        x = self.embedding(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        # Global Average Pooling across time
        x = x.mean(dim=1)
        x = self.fc(x)
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (Batch, Time, d_model)
        x = x + self.pe[:x.size(1), :]
        return x
