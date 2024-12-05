import numpy as np
class TransformerCore:

    @staticmethod
    def relu(x):
        return np.maximum(0, x)





    @staticmethod
    def scaled_dot_product_attention(q, k, v, mask=None):
        """
        q: запросы (queries), размерность: (..., seq_len_q, d_k)
        k: ключи (keys), размерность: (..., seq_len_k, d_k)
        v: значения (values), размерность: (..., seq_len_k, d_v)
        mask: опциональная маска для игнорирования определённых позиций

        Возвращает:
        - Выход внимания (размерность как у v)
        - Матрицы весов внимания
        """
        d_k = q.shape[-1]
        scores = np.matmul(q, k.transpose(-2, -1)) / np.sqrt(d_k)  # Вычисляем скоры
        if mask is not None:
            scores += (mask * -1e9)  # Применяем маску

        attention_weights = TransformerCore.softmax(scores, axis=-1)
        output = np.matmul(attention_weights, v)  # Умножаем на значения
        return output, attention_weights

    @staticmethod
    def softmax(x, axis=-1):
        exps = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return exps / np.sum(exps, axis=axis, keepdims=True)

    @staticmethod
    def concat_heads(x):
        """
        Объединяет результаты многоголового внимания.
        """
        return np.concatenate(np.split(x, x.shape[0], axis=0), axis=-1)
    @staticmethod
    def layer_norm(x, epsilon=1e-6):
        """
        Нормализация по слоям.

        x: входной тензор (..., seq_len, d_model)
        epsilon: малое число для стабильности
        """
        mean = np.mean(x, axis=-1, keepdims=True)
        variance = np.var(x, axis=-1, keepdims=True)
        return (x - mean) / np.sqrt(variance + epsilon)
    @staticmethod
    def transformer_layer(x, num_heads, d_model, d_ff):

        """
        Реализация одного слоя трансформера.

        x: входной тензор (..., seq_len, d_model)
        num_heads: количество голов
        d_model: размерность модели
        d_ff: размер скрытого слоя
        """
        # Многоголовое внимание
        attn_output = TransformerCore.multi_head_attention(x, x, x, num_heads, d_model)
        attn_output = TransformerCore.layer_norm(x + attn_output)  # Резидуальная связь и нормализация

        # Feed-forward network
        ff_output = TransformerCore.feed_forward(attn_output, d_ff, d_model)
        output = TransformerCore.layer_norm(attn_output + ff_output)  # Резидуальная связь и нормализация

        return output

    @staticmethod
    def split_heads(x, num_heads, d_k):
        """
        Разбивает тензор на несколько голов.
        """
        return np.stack(np.split(x, num_heads, axis=-1), axis=0)


    @staticmethod
    def multi_head_attention(q, k, v, num_heads, d_model):

        """
        Реализация мультихэдового внимания с учетом правильной размерности.
        """
        if len(q.shape) == 2:
            q = np.expand_dims(q, axis=0)
            k = np.expand_dims(k, axis=0)
            v = np.expand_dims(v, axis=0)
        batch_size, seq_len, _ = q.shape
        d_k = d_model // num_heads  # Размерность каждой головы

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

        return attention_output

