import numpy as np

class AttentionLayer:

    def __init__(self, d_model, d_ff,num_heads):
        self.num_heads=num_heads
        self.d_model = d_model
        self.d_ff = d_ff
        self.W_q = np.random.randn(d_model, d_model) * 0.1
        self.W_k = np.random.randn(d_model, d_model) * 0.1
        self.W_v = np.random.randn(d_model, d_model) * 0.1
        self.W_o = np.random.randn(d_model, d_model) * 0.1
        self.q = None
        self.k = None
        self.v =None
        #Инициализируем веса для матриц внимания

    def multi_head_attention(self, q, k, v, num_heads, d_model, W_q, W_k, W_v, W_o):
        self.q=q
        self.k=k
        self.v=v
        """
        Реализация многоголового внимания с линейными преобразованиями.
        """
        if len(q.shape) == 2:
            q = np.expand_dims(q, axis=0)
            k = np.expand_dims(k, axis=0)
            v = np.expand_dims(v, axis=0)

        batch_size, seq_len, _ = q.shape
        d_k = d_model // num_heads  # Размерность каждой головы

        # Линейные преобразования для q, k, v
        q = np.matmul(q, W_q)  # (batch_size, seq_len, d_model)
        k = np.matmul(k, W_k)  # (batch_size, seq_len, d_model)
        v = np.matmul(v, W_v)  # (batch_size, seq_len, d_model)

        # Разделяем q, k, v на "головы"
        q_heads = q.reshape(batch_size, seq_len, num_heads, d_k).transpose(0, 2, 1, 3)
        k_heads = k.reshape(batch_size, seq_len, num_heads, d_k).transpose(0, 2, 1, 3)
        v_heads = v.reshape(batch_size, seq_len, num_heads, d_k).transpose(0, 2, 1, 3)

        # Скалярное произведение Q и K
        scores = np.matmul(q_heads, k_heads.transpose(0, 1, 3, 2)) / np.sqrt(d_k)
        weights = np.exp(scores - np.max(scores, axis=-1, keepdims=True))  # Стабильная softmax
        weights /= np.sum(weights, axis=-1, keepdims=True)

        # Умножаем веса на V
        attention_output = np.matmul(weights, v_heads)

        # Объединяем головы обратно
        attention_output = attention_output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, d_model)

        # Преобразуем через W_o
        attention_output = np.matmul(attention_output, W_o)

        return attention_output


    def backward_attention(self, d_output, q, k, v):
        """
        Обратный проход для многоголового внимания.

        Аргументы:
        - attn_layer: объект слоя внимания (включает параметры W_q, W_k, W_v, W_o)
        - d_output: градиенты от следующего слоя, форма (batch_size, seq_len, d_model)
        - q, k, v: запросы, ключи, значения, которые использовались при прямом проходе

        Возвращает:
        - d_q: градиенты по запросам
        - d_k: градиенты по ключам
        - d_v: градиенты по значениям
        """
        W_q, W_k, W_v, W_o = self.W_q, self.W_k, self.W_v, self.W_o

        # Градиенты через линейный выход
        d_output_heads = d_output @ W_o.T  # (batch_size, seq_len, d_model)

        # Разделение на головы
        batch_size, seq_len, d_model = d_output.shape
        num_heads = self.num_heads
        d_k = d_model // num_heads

        d_output_heads = d_output_heads.reshape(batch_size, seq_len, num_heads, d_k).transpose(0, 2, 1, 3)

        # Градиенты через внимание
        scores = np.matmul(q, k.transpose(0, 1, 3, 2)) / np.sqrt(d_k)  # (batch_size, num_heads, seq_len, seq_len)
        weights = np.exp(scores) / np.sum(np.exp(scores), axis=-1, keepdims=True)

        d_scores = np.matmul(d_output_heads, v.transpose(0, 1, 3, 2))  # Градиенты для скоринга
        d_weights = d_scores * weights * (1 - weights)  # d_softmax
        d_q = np.matmul(d_weights, k) / np.sqrt(d_k)
        d_k = np.matmul(d_weights.transpose(0, 1, 3, 2), q) / np.sqrt(d_k)
        d_v = np.matmul(d_output_heads.transpose(0, 1, 3, 2), weights)

        # Объединяем головы и возвращаем
        d_q = d_q.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, d_model)
        d_k = d_k.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, d_model)
        d_v = d_v.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, d_model)

        return d_q, d_k, d_v

    def backward(self, d_output, encoder_outputs):
        # Градиенты для self-attention
        d_self_attn = self.backward_attention(self,self.self_attn, d_output, self.q, self.k, self.v)
        # Градиенты для cross-attention
        d_cross_attn = self.backward_attention(self,self.cross_attn, d_output, encoder_outputs)
        # Градиенты для feed-forward
        d_ffn = self.backward_feed_forward(self.feed_forward, d_output)
        return d_ffn

    def backward_through_encoder_layer(layer, d_output, src):
        # Градиенты для self-attention
        d_self_attn = backward_attention(layer.self_attn, d_output)
        # Градиенты для feed-forward
        d_ffn = backward_feed_forward(layer.feed_forward, d_output)
        return d_ffn


