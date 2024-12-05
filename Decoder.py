import numpy as np
from TransformerCore import TransformerCore
from Layer import Layer

class Decoder:
    def __init__(self, num_heads, d_model, d_ff, num_layers):
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_ff = d_ff
        self.num_layers = num_layers

        # Инициализация весов для всех слоёв декодера
        self.layers = [Layer(d_model, d_ff) for _ in range(num_layers)]

    def forward(self, tgt, encoder_outputs):
        """
        Аргументы:
        - tgt: целевая последовательность (вход декодера) с формой (batch_size, tgt_len, d_model)
        - encoder_outputs: выходы энкодера с формой (batch_size, src_len, d_model)

        Возвращает:
        - выход декодера с формой (batch_size, tgt_len, d_model)
        """
        x = tgt  # Начальный вход — закодированные векторы целевой последовательности

        for layer in self.layers:
            # Self-attention на текущей целевой последовательности
            x = TransformerCore.multi_head_attention(x, x, x, self.num_heads, self.d_model)

            # Cross-attention: связь с выходами энкодера
            x = TransformerCore.multi_head_attention(x, encoder_outputs, encoder_outputs, self.num_heads, self.d_model)

            # Feed Forward Network
            x = layer.feed_forward(x)

        return x



