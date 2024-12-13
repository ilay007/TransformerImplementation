import numpy as np
from TransformerCore import TransformerCore

class Layer:

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


    def transformer_layer(self,x, num_heads, d_model):

        """
        Реализация одного слоя трансформера.

        x: входной тензор (..., seq_len, d_model)
        num_heads: количество голов
        d_model: размерность модели
        d_ff: размер скрытого слоя
        """
        # Многоголовое внимание
        attn_output = self.multi_head_attention(x, x, x, num_heads, d_model)
        attn_output = TransformerCore.layer_norm(x + attn_output)  # Резидуальная связь и нормализация

        # Feed-forward network
        ff_output = self.feed_forward(attn_output)
        output = TransformerCore.layer_norm(attn_output + ff_output)  # Резидуальная связь и нормализация



        return output

    def feed_forward(self,x):

        """
        Простая двухслойная полносвязная сеть.

        x: входной тензор (..., seq_len, d_model)
        d_ff: размер скрытого слоя
        d_model: размер выхода
        """

        z1 = np.matmul(x, self.w1) + self.b1
        a1 = TransformerCore.relu(z1)
        z2 = np.matmul(a1, self.w2) + self.b2
        return z2

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


    def backward_attention(attn_layer, d_output, q, k, v):
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
        W_q, W_k, W_v, W_o = attn_layer.W_q, attn_layer.W_k, attn_layer.W_v, attn_layer.W_o

        # Градиенты через линейный выход
        d_output_heads = d_output @ W_o.T  # (batch_size, seq_len, d_model)

        # Разделение на головы
        batch_size, seq_len, d_model = d_output.shape
        num_heads = attn_layer.num_heads
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

    def backward_feed_forward(feed_forward_layer, d_output, x):
        """
        Обратный проход для feed-forward слоя.

        Аргументы:
        - feed_forward_layer: объект слоя feed-forward
        - d_output: градиенты от следующего слоя, форма (batch_size, seq_len, d_model)
        - x: входные данные в feed-forward слой

        Возвращает:
        - d_x: градиенты входных данных
        """
        W_1, b_1, W_2, b_2 = feed_forward_layer.W_1, feed_forward_layer.b_1, feed_forward_layer.W_2, feed_forward_layer.b_2

        # Прямой проход через слой (для сохранённых значений)
        z1 = np.matmul(x, W_1) + b_1
        a1 = np.maximum(0, z1)  # ReLU

        # Градиенты через второй слой
        d_a1 = np.matmul(d_output, W_2.T)  # Градиенты активаций после ReLU
        d_W2 = np.matmul(a1.transpose(0, 2, 1), d_output).mean(axis=0)  # Градиенты по весам W_2
        d_b2 = d_output.mean(axis=(0, 1))  # Градиенты по смещению b_2

        # Градиенты через ReLU
        d_z1 = d_a1 * (z1 > 0)

        # Градиенты через первый слой
        d_W1 = np.matmul(x.transpose(0, 2, 1), d_z1).mean(axis=0)  # Градиенты по весам W_1
        d_b1 = d_z1.mean(axis=(0, 1))  # Градиенты по смещению b_1
        d_x = np.matmul(d_z1, W_1.T)  # Градиенты по входным данным

        # Обновление параметров
        feed_forward_layer.W_1 -= lr * d_W1
        feed_forward_layer.b_1 -= lr * d_b1
        feed_forward_layer.W_2 -= lr * d_W2
        feed_forward_layer.b_2 -= lr * d_b2

        return d_x

    def multi_head_attention(self,q, k, v, num_heads, d_model):
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

    def backward_through_encoder_layer(layer, d_output, src):
        # Градиенты для self-attention
        d_self_attn = backward_attention(layer.self_attn, d_output)
        # Градиенты для feed-forward
        d_ffn = backward_feed_forward(layer.feed_forward, d_output)
        return d_ffn


