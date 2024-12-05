import numpy as np

import numpy as np
from TransformerCore import TransformerCore
from Layer import Layer
from Decoder import Decoder

def positional_encoding(seq_len, d_model):
    pos = np.arange(seq_len)[:, np.newaxis]
    i = np.arange(d_model)[np.newaxis, :]
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / d_model)
    pos_encoding = pos * angle_rates
    pos_encoding[:, 0::2] = np.sin(pos_encoding[:, 0::2])  # Чётные индексы
    pos_encoding[:, 1::2] = np.cos(pos_encoding[:, 1::2])  # Нечётные индексы
    return pos_encoding


class TransformerModel:
    def __init__(self, vocab_size_src, vocab_size_tgt, d_model, num_heads, num_layers, d_ff, seq_len):
        self.vocab_size_src = vocab_size_src
        self.vocab_size_tgt = vocab_size_tgt
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.d_ff = d_ff
        self.seq_len = seq_len

        # Параметры слоёв
        self.encoder_embeddings = np.random.randn(vocab_size_src, d_model) * 0.1
        self.decoder_embeddings = np.random.randn(vocab_size_tgt, d_model) * 0.1
        self.encoder_layers = [Layer(d_model,d_ff) for _ in range(num_layers)]
        self.decoder_layers = [Layer(d_model,d_ff)  for _ in range(num_layers)]
        self.output_projection = np.random.randn(d_model, vocab_size_tgt) * 0.1

        #Инициализируем декодер
        self.decoder=Decoder(num_heads, d_model, d_ff, num_layers)


    def forward(self, src, tgt):
        if len(src)>len(tgt):
            ar=[0]*(len(tgt)-len(src))
            tgt=np.append(tgt,0)

        # Позиционные эмбеддинги
        embed=self.embed(src, self.encoder_embeddings)
        src_emb = self.add_positional_encoding(embed, src.shape[0], self.d_model)
        tgt_emb = self.add_positional_encoding(self.embed(tgt, self.decoder_embeddings), tgt.shape[0], self.d_model)


        # Энкодер
        for layer in self.encoder_layers:
            src_emb = layer.transformer_layer(src_emb, self.num_heads, self.d_model, self.d_ff)

        # Декодер
        self.decoder.forward(src_emb,tgt_emb)

        # Прогноз
        logits = np.matmul(tgt_emb, self.output_projection)
        return logits

    def embed(self, tokens, embeddings):
        return embeddings[tokens]

    def mpositional_encoding(self, x):
        pos_enc=positional_encoding(self.seq_len, self.d_model)
        return x + pos_enc

    def add_positional_encoding(self,x, seq_len, d_model):
        """
        Добавление позиционного кодирования
        """
        pos_enc = positional_encoding(seq_len, d_model)
        if x.shape[0] != pos_enc.shape[0]:  # Проверяем длину последовательностей
            pos_enc = pos_enc[:x.shape[0]]  # Урезаем позиционное кодирование до нужной длины
        return x + pos_enc

