import torch
from torch import nn


class SimpleHARCNN(nn.Module):
    def __init__(self, in_channels=3, num_classes=5, mid_channels=64, kernel_size=5, stride=1, dropout=0.1, final_out_channels=128, features_len=16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels, mid_channels, kernel_size=kernel_size, stride=stride, bias=False, padding=kernel_size // 2),
            nn.BatchNorm1d(mid_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(dropout),
            nn.Conv1d(mid_channels, mid_channels * 2, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(mid_channels * 2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            nn.Conv1d(mid_channels * 2, final_out_channels, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(final_out_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            nn.AdaptiveAvgPool1d(features_len),
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(final_out_channels * features_len, num_classes)

    def encode(self, x):
        return self.encoder(x).reshape(x.shape[0], -1)

    def forward(self, x):
        return self.classifier(self.dropout(self.encode(x)))
