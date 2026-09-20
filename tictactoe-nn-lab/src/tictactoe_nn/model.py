"""ニューラルネットワーク（全結合の多層パーセプトロン）の定義。"""
from __future__ import annotations

import torch.nn as nn

from .data import N_CLASSES, input_dim

ACTIVATIONS = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "leaky_relu": nn.LeakyReLU,
}


class MLP(nn.Module):
    """入力 → [全結合 + 活性化] × hidden_layers → 全結合(3クラス)。

    hidden_layers=0 のときは隠れ層なし（入力から直接3クラスへの線形モデル）。
    出力は「確率になる前の値(logits)」。確率にするときは softmax を通す。
    """

    def __init__(self, input_repr: str, hidden_layers: int, units: int, activation: str):
        super().__init__()
        if activation not in ACTIVATIONS:
            raise ValueError(f"activation は {list(ACTIVATIONS)} のいずれかです: {activation!r}")
        layers = []
        dim = input_dim(input_repr)
        for _ in range(int(hidden_layers)):
            layers.append(nn.Linear(dim, int(units)))
            layers.append(ACTIVATIONS[activation]())
            dim = int(units)
        layers.append(nn.Linear(dim, N_CLASSES))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
