import numpy as np
from TransformerCore import TransformerCore
from Layer0 import Layer

class Encoder:
    def __init__(self, num_heads, d_model, d_ff, num_layers):
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_ff = d_ff
        self.num_layers = num_layers

        # Инициализация весов для всех слоёв декодера
        self.layers = [Layer(d_model, d_ff) for _ in range(num_layers)]

    def forward(self, src_emb):
        """
        Аргументы:
        - tgt: целевая последовательность (вход декодера) с формой (batch_size, tgt_len, d_model)
        - encoder_outputs: выходы энкодера с формой (batch_size, src_len, d_model)

        Возвращает:
        - выход декодера с формой (batch_size, tgt_len, d_model)
        """

        for layer in self.layers:
            src_emb = layer.transformer_layer(src_emb, self.num_heads, self.d_model, self.d_ff)





