import numpy as np

# --- Вспомогательные функции ---

def apply_embedding(data, embedding_matrix):
    """
    Преобразует токены в эмбеддинги с использованием матрицы эмбеддингов.
    :param data: Массив токенов [batch_size, seq_len].
    :param embedding_matrix: Матрица эмбеддингов [vocab_size, d_model].
    :return: Массив эмбеддингов [batch_size, seq_len, d_model].
    """
    return embedding_matrix[data]


def compute_loss_grad(output, target, vocab_size):
    """
    Вычисляет градиент функции потерь по выходу модели.
    :param output: Предсказания модели (логиты) [batch_size, seq_len, vocab_size].
    :param target: Истинные токены [batch_size, seq_len].
    :param vocab_size: Размер словаря.
    :return: Градиенты по выходу [batch_size, seq_len, vocab_size].
    """
    # Применяем softmax
    probs = softmax(output)

    # Создаём one-hot представление целевых токенов
    target_one_hot = np.eye(vocab_size)[target]

    # Вычисляем градиенты
    grad = (probs - target_one_hot) / output.shape[0]
    return grad


def compute_loss(output, target, vocab_size):
    """
    Вычисляет кросс-энтропийную потерю.
    :param output: Предсказания модели (логиты) [batch_size, seq_len, vocab_size].
    :param target: Истинные токены [batch_size, seq_len].
    :param vocab_size: Размер словаря.
    :return: Средняя потеря.
    """
    batch_size, seq_len, _ = output.shape
    # Применяем softmax для вычисления вероятностей
    probs = softmax(output)

    # Создаём one-hot представление целевых токенов
    target_one_hot = np.eye(vocab_size)[target]

    # Вычисляем потери: -y_true * log(y_pred)
    loss = -np.sum(target_one_hot * np.log(probs + 1e-9)) / (batch_size * seq_len)
    return loss

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)

def relu(x):
    return np.maximum(0, x)

def relu_grad(x):
    return (x > 0).astype(float)

def layer_norm(x, gamma, beta, eps=1e-6):
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    norm_x = (x - mean) / np.sqrt(var + eps)
    return gamma * norm_x + beta, norm_x, mean, var

def layer_norm_grad(dout, norm_x, gamma, mean, var, eps=1e-6):
    """
    Вычисляет градиенты для layer norm при входе 3D тензора.
    :param dout: Градиенты выхода [batch_size, seq_len, d_model].
    :param norm_x: Нормализованный вход [batch_size, seq_len, d_model].
    :param gamma: Параметр масштабирования [d_model].
    :param mean: Среднее по последнему измерению [batch_size, seq_len, 1].
    :param var: Дисперсия по последнему измерению [batch_size, seq_len, 1].
    :param eps: Число для избежания деления на 0.
    :return: Градиенты по входу, gamma и beta.
    """
    B, S, D = dout.shape  # Обработка 3D входа
    dbeta = np.sum(dout, axis=(0, 1))  # Суммируем по батчу и последовательности
    dgamma = np.sum(dout * norm_x, axis=(0, 1))  # Градиент по gamma

    dnorm_x = dout * gamma  # Применяем gamma
    dvar = np.sum(dnorm_x * (norm_x * -0.5) / (var + eps), axis=-1, keepdims=True)
    dmean = np.sum(dnorm_x * -1 / np.sqrt(var + eps), axis=-1, keepdims=True) + \
            dvar * np.sum(-2 * norm_x, axis=-1, keepdims=True) / D

    dx = dnorm_x / np.sqrt(var + eps) + dvar * 2 * norm_x / D + dmean / D
    return dx, dgamma, dbeta


def create_casual_mask(seq_len):
    """Создаёт casual mask для последовательности."""
    mask = np.triu(np.ones((seq_len, seq_len)), k=1)
    mask[mask == 1] = -np.inf
    mask[mask == 0] = 0
    return mask

def shift_left(tgt_batch):
    """Сдвигает целевое предложение влево на один токен."""
    return np.roll(tgt_batch, shift=-1, axis=1)

# --- Реализация AttentionLayer ---
class AttentionLayer:
    def __init__(self, d_model, n_heads):
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_q = np.random.randn(d_model, d_model) / np.sqrt(d_model)
        self.W_k = np.random.randn(d_model, d_model) / np.sqrt(d_model)
        self.W_v = np.random.randn(d_model, d_model) / np.sqrt(d_model)
        self.W_o = np.random.randn(d_model, d_model) / np.sqrt(d_model)
        self.input = None

    def forward(self, Q, K, V, mask=None):
        self.input = Q
        self.Q = Q @ self.W_q
        self.K = K @ self.W_k
        self.V = V @ self.W_v

        batch_size, seq_len, _ = self.Q.shape
        self.Q = self.Q.reshape(batch_size, seq_len, self.n_heads, self.d_k).transpose(0, 2, 1, 3)
        self.K = self.K.reshape(batch_size, seq_len, self.n_heads, self.d_k).transpose(0, 2, 1, 3)
        self.V = self.V.reshape(batch_size, seq_len, self.n_heads, self.d_k).transpose(0, 2, 1, 3)

        scores = (self.Q @ self.K.transpose(0, 1, 3, 2)) / np.sqrt(self.d_k)
        if mask is not None:
            scores += mask[np.newaxis, np.newaxis, :, :]  # Apply mask

        self.attn = softmax(scores)
        output = self.attn @ self.V# на выходе размер 23*8*50*64

        output = output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, -1)#после reshape размер стал (32,50,512)
        self.output = output @ self.W_o
        return self.output

    def backward(self, d_output):
        # Backward for output projection
        d_o = d_output @ self.W_o.T
        d_W_o = self.output.transpose(0, 2, 1).reshape(-1, self.d_model).T @ d_output.reshape(-1, self.d_model)

        # Reshape and split gradients for multi-head attention
        batch_size, seq_len, _ = d_o.shape
        d_o = d_o.reshape(batch_size, seq_len, self.n_heads, self.d_k).transpose(0, 2, 1, 3)

        # Backpropagate through attention weights
        d_attn = d_o @ self.V.transpose(0, 1, 3, 2)
        d_V = self.attn.transpose(0, 1, 3, 2) @ d_o

        # Backpropagate through scaled dot-product
        d_scores = d_attn * self.attn * (1 - self.attn)
        KT=self.K.transpose(0, 1, 3, 2)
        d_Q = d_scores @ self.K
        d_K = d_scores.transpose(0, 1, 3, 2) @ self.Q

        # Reshape back to original
        d_Q = d_Q.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.d_model)
        d_K = d_K.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.d_model)
        d_V = d_V.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.d_model)

        # Backward through input projections
        d_W_q = self.input.transpose(0, 2, 1).reshape(-1, self.d_model).T @ d_Q.reshape(-1, self.d_model)
        d_W_k = self.input.transpose(0, 2, 1).reshape(-1, self.d_model).T @ d_K.reshape(-1, self.d_model)
        d_W_v = self.input.transpose(0, 2, 1).reshape(-1, self.d_model).T @ d_V.reshape(-1, self.d_model)

        d_input = d_Q @ self.W_q.T + d_K @ self.W_k.T + d_V @ self.W_v.T

        return d_input, d_W_q, d_W_k, d_W_v, d_W_o


# --- Реализация Encoder ---
class Encoder:
    def __init__(self, d_model, n_heads, ff_hidden, seq_len):
        self.attention = AttentionLayer(d_model, n_heads)
        self.W1 = np.random.randn(d_model, ff_hidden) / np.sqrt(d_model)
        self.W2 = np.random.randn(ff_hidden, d_model) / np.sqrt(ff_hidden)
        self.gamma_1 = np.ones((d_model,))
        self.beta_1 = np.zeros((d_model,))
        self.gamma_2 = np.ones((d_model,))
        self.beta_2 = np.zeros((d_model,))
        self.d_model=d_model

    def forward(self, x, mask=None):
        self.input = x
        attn_out = self.attention.forward(x, x, x, mask)
        self.norm_1_out, self.norm_1_x, self.norm_1_mean, self.norm_1_var = layer_norm(attn_out + x, self.gamma_1, self.beta_1)

        ff_out = relu(self.norm_1_out @ self.W1) @ self.W2
        self.norm_2_out, self.norm_2_x, self.norm_2_mean, self.norm_2_var = layer_norm(ff_out + self.norm_1_out, self.gamma_2, self.beta_2)

        return self.norm_2_out

    def backward0(self, d_out):
        # Backward for second layer norm
        d_ff = d_out
        d_ff, dgamma_2, dbeta_2 = layer_norm_grad(d_ff, self.norm_2_x, self.gamma_2, self.norm_2_mean, self.norm_2_var)

        # Backward through feed-forward
        d_relu = d_ff @ self.W2.T

        self.norm_1_out = self.norm_1_out.reshape(-1, self.d_model)  # (batch_size * seq_len, d_model)
        d_ff = d_ff.reshape(-1, self.d_model)  # (batch_size * seq_len, d_model)

        d_W2 = self.norm_1_out.T @ d_ff
        dfd=relu_grad(self.norm_1_out @ self.W1)
        dfd = dfd.reshape(batch_size, seq_len, -1)
        d_ff_input = d_relu * dfd

        # Приведение форм
        self_input_reshaped = self.input.reshape(-1, self.input.shape[-1])  # (1600, 512)
        d_ff_input_reshaped = d_ff_input.reshape(-1, d_ff_input.shape[-1])  # (1600, 2048)

        # Умножение
        d_W1 = self_input_reshaped.T @ d_ff_input_reshaped  # Результат: (512, 2048)

        d_ff_input_projected = d_ff_input @ self.W2.T  # (32, 50, 2048) @ (2048, 512)



        # Add residual and backward through first layer norm
        #d_norm_1 = d_ff_input + self.input
        d_norm_1 = d_ff_input + self.input
        d_attn = d_norm_1
        d_attn, dgamma_1, dbeta_1 = layer_norm_grad(d_attn, self.norm_1_x, self.gamma_1, self.norm_1_mean,
                                                    self.norm_1_var)

        # Backward through attention
        d_input, d_W_q, d_W_k, d_W_v, d_W_o = self.attention.backward(d_attn)

        return d_input, d_W1, d_W2, dgamma_1, dbeta_1, dgamma_2, dbeta_2

    def backward(self, d_out):
        """
        Backward pass for the encoder.
        :param d_out: Gradient of the output [batch_size, seq_len, d_model].
        :return: Gradients of inputs and parameters.
        """
        batch_size, seq_len, d_model = self.input.shape

        # Backward for the second layer norm
        d_ffn, dgamma_2, dbeta_2 = layer_norm_grad(
            d_out, self.norm_2_x, self.gamma_2, self.norm_2_mean, self.norm_2_var
        )

        # Backward through feed-forward network (FFN)
        d_relu = d_ffn @ self.W2.T  # [batch_size, seq_len, d_model]


        d_W2=np.einsum('bsf,bsd->fd', d_ffn, self.norm_1_out)

        d_ff_input = d_relu * relu_grad(self.norm_1_out @ self.W1)  # [batch_size, seq_len, ff_hidden]
        d_W1 = np.einsum('bsd,bsf->df', self.input, d_ff_input)  # [d_model, ff_hidden]

        # Project gradients back to d_model dimension
        d_ff_input_projected = d_ff_input @ self.W2  # [batch_size, seq_len, d_model]

        # Add residual and backward through first layer norm
        d_norm_1 = d_ff_input_projected + d_out  # [batch_size, seq_len, d_model]
        d_attn, dgamma_1, dbeta_1 = layer_norm_grad(
            d_norm_1, self.norm_1_x, self.gamma_1, self.norm_1_mean, self.norm_1_var
        )

        # Backward through attention
        d_input, d_W_q, d_W_k, d_W_v, d_W_o = self.attention.backward(d_attn)

        return d_input, d_W1, d_W2, d_W_q, d_W_k, d_W_v, d_W_o, dgamma_1, dbeta_1, dgamma_2, dbeta_2


# --- Реализация Decoder ---
class Decoder:
    def __init__(self, d_model, n_heads, ff_hidden, seq_len,vocab_size):
        self.self_attention = AttentionLayer(d_model, n_heads)
        self.enc_dec_attention = AttentionLayer(d_model, n_heads)
        self.W1 = np.random.randn(d_model, ff_hidden) / np.sqrt(d_model)
        self.W2 = np.random.randn(ff_hidden, d_model) / np.sqrt(ff_hidden)
        self.gamma_1 = np.ones((d_model,))
        self.beta_1 = np.zeros((d_model,))
        self.gamma_2 = np.ones((d_model,))
        self.beta_2 = np.zeros((d_model,))
        self.gamma_3 = np.ones((d_model,))
        self.beta_3 = np.zeros((d_model,))
        self.linear_vocab = np.random.randn(d_model, vocab_size) / np.sqrt(d_model)


    def forward(self, x, enc_out, self_mask=None, enc_dec_mask=None):
        self.input = x
        self.enc_out = enc_out

        self.self_attn_out = self.self_attention.forward(x, x, x, self_mask)
        self.norm_1_out, self.norm_1_x, self.norm_1_mean, self.norm_1_var = layer_norm(self.self_attn_out + x, self.gamma_1, self.beta_1)

        self.enc_dec_attn_out = self.enc_dec_attention.forward(self.norm_1_out, enc_out, enc_out, enc_dec_mask)
        self.norm_2_out, self.norm_2_x, self.norm_2_mean, self.norm_2_var = layer_norm(self.enc_dec_attn_out + self.norm_1_out, self.gamma_2, self.beta_2)

        ff_out = relu(self.norm_2_out @ self.W1) @ self.W2
        self.norm_3_out, self.norm_3_x, self.norm_3_mean, self.norm_3_var = layer_norm(ff_out + self.norm_2_out, self.gamma_3, self.beta_3)

        self.vocab_logits = self.norm_3_out @ self.linear_vocab
        return self.vocab_logits

        #return self.norm_3_out

    def backward_vocab_projection(self,dout, W_vocab):
        """
        Обратное распространение через линейный слой для логитов словаря.
        :param dout: Градиенты по логитам [batch_size, seq_len, vocab_size].
        :param W_vocab: Матрица весов линейного слоя [d_model, vocab_size].
        :return: Градиенты по входу линейного слоя [batch_size, seq_len, d_model]
                 и обновлённые градиенты по W_vocab.
        """
        # Транспонируем W_vocab для корректного матричного умножения
        W_vocab_T = W_vocab.T  # [vocab_size, d_model]

        # Градиенты по входам
        d_input = np.matmul(dout, W_vocab_T)  # [batch_size, seq_len, d_model]

        # Градиенты по W_vocab
        # Транспонируем вход для умножения с dout
        d_W_vocab = np.einsum('bsd,bse->de', dout, self.norm_3_out)  # [d_model, vocab_size]

        return d_input, d_W_vocab

    def backward(self, d_out):
        # 1. Обратное распространение через линейный слой
        d_vocab, d_W_vocab = self.backward_vocab_projection(d_out, self.linear_vocab)

        # 2. Обратное распространение через слой нормализации (после FFN)
        d_ffn, dgamma_3, dbeta_3 = layer_norm_grad(d_vocab, self.norm_3_x, self.gamma_3, self.norm_3_mean,
                                                   self.norm_3_var)


        # 3. Обратное распространение через FFN
        d_relu = d_ffn @ self.W2.T  # [batch_size, seq_len, d_model]
        d_W2 = np.einsum('bsf,bsd->fd', d_relu,self.norm_1_out)  # [d_model, ff_hidden]
        d_ff_input = d_relu * relu_grad(self.norm_2_out @ self.W1)  # [batch_size, seq_len, ff_hidden]
        d_W1 = np.einsum('bsf,bsd->fd', self.input, d_ff_input)  # [d_model, ff_hidden]

        d_ff_input_projected = d_ff_input @ self.W2  # [batch_size, seq_len, d_model]

        # Add residual and backward through second layer norm
        d_norm_2 = d_ff_input_projected + self.norm_1_out
        d_enc_dec = d_norm_2
        d_enc_dec, dgamma_2, dbeta_2 = layer_norm_grad(d_enc_dec, self.norm_2_x, self.gamma_2, self.norm_2_mean,
                                                       self.norm_2_var)

        # Backward through encoder-decoder attention
        d_input_dec, d_W_q_dec, d_W_k_dec, d_W_v_dec, d_W_o_dec = self.enc_dec_attention.backward(d_enc_dec)

        # Add residual and backward through first layer norm
        d_norm_1 = d_input_dec + self.input
        d_self_attn = d_norm_1
        d_self_attn, dgamma_1, dbeta_1 = layer_norm_grad(d_self_attn, self.norm_1_x, self.gamma_1, self.norm_1_mean,
                                                         self.norm_1_var)

        # Backward through self-attention
        d_input_self, d_W_q_self, d_W_k_self, d_W_v_self, d_W_o_self = self.self_attention.backward(d_self_attn)

        return (d_input_self, d_W1, d_W2, d_W_q_self, d_W_k_self, d_W_v_self, d_W_o_self,
                d_W_q_dec, d_W_k_dec, d_W_v_dec, d_W_o_dec, dgamma_1, dbeta_1, dgamma_2, dbeta_2, dgamma_3, dbeta_3)






# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # --- Обучение ---
    # Гиперпараметры
    d_model = 512
    n_heads = 8
    ff_hidden = 2048
    seq_len = 50
    vocab_size_en = 10000  # Английский словарь
    vocab_size_fr = 10000  # Французский словарь
    epochs = 10
    batch_size = 32
    learning_rate = 0.001

    # Создаём модель
    encoder = Encoder(d_model=d_model, n_heads=n_heads, ff_hidden=ff_hidden, seq_len=seq_len)
    decoder = Decoder(d_model=d_model, n_heads=n_heads, ff_hidden=ff_hidden, seq_len=seq_len,vocab_size=vocab_size_en)

    # Пример данных
    # Инициализация эмбеддингов

    embedding_matrix_en = np.random.randn(vocab_size_en, d_model) / np.sqrt(d_model)
    embedding_matrix_fr = np.random.randn(vocab_size_fr, d_model) / np.sqrt(d_model)

    train_data_en = np.random.randint(0, vocab_size_en, (1000, seq_len))
    train_data_fr = np.random.randint(0, vocab_size_fr, (1000, seq_len))

    for epoch in range(epochs):
        total_loss = 0
        for i in range(0, len(train_data_en), batch_size):
            # Получаем батч данных
            src_batch = train_data_en[i:i + batch_size]
            tgt_batch = train_data_fr[i:i + batch_size]

            # Преобразуем в эмбеддинги
            src_embedded = apply_embedding(src_batch, embedding_matrix_en)  # [batch_size, seq_len, d_model]
            tgt_embedded = apply_embedding(tgt_batch, embedding_matrix_fr)  # [batch_size, seq_len, d_model]

            # Сдвигаем целевые данные
            tgt_input = shift_left(tgt_batch)
            tgt_input_embedded = apply_embedding(tgt_input, embedding_matrix_fr)  # [batch_size, seq_len, d_model]

            # Создаём casual mask
            casual_mask = create_casual_mask(seq_len)

            # Прямой проход
            src_encoded = encoder.forward(src_embedded)
            output = decoder.forward(tgt_input_embedded, src_encoded, self_mask=casual_mask)

            # Вычисление потерь
            loss = compute_loss(output, tgt_batch, vocab_size_fr)
            total_loss += loss

            # Вычисляем градиенты
            d_output = compute_loss_grad(output, tgt_batch, vocab_size_fr)

            # Обратное распространение
            d_decoder = decoder.backward(d_output)
            d_input_self = d_decoder[0]
            d_encoder = encoder.backward(d_input_self)

            # Обновление параметров (градиентный спуск)
            learning_rate = 0.001

            # Обновляем параметры Encoder
            encoder.W1 -= learning_rate * d_encoder[1]
            #encoder.W2 -= learning_rate * d_encoder[2] - ошибка W2 размерность 2048,512
            encoder.gamma_1 -= learning_rate * d_encoder[7]
            encoder.beta_1 -= learning_rate * d_encoder[8]
            encoder.gamma_2 -= learning_rate * d_encoder[9]
            encoder.beta_2 -= learning_rate * d_encoder[10]

            # Обновляем параметры Attention в Encoder
            encoder.attention.W_q -= learning_rate * d_encoder[3]
            encoder.attention.W_k -= learning_rate * d_encoder[4]
            encoder.attention.W_v -= learning_rate * d_encoder[5]
            encoder.attention.W_o -= learning_rate * d_encoder[6]

            # Обновляем параметры Decoder
            decoder.W1 -= learning_rate * d_decoder[1]
            decoder.W2 -= learning_rate * d_decoder[2]  #ошибка W2 размерность 2048 на 512 у d_decoder[2] размерность 512 на 512
            decoder.gamma_1 -= learning_rate * d_decoder[11]
            decoder.beta_1 -= learning_rate * d_decoder[12]
            decoder.gamma_2 -= learning_rate * d_decoder[13]
            decoder.beta_2 -= learning_rate * d_decoder[14]
            decoder.gamma_3 -= learning_rate * d_decoder[15]
            decoder.beta_3 -= learning_rate * d_decoder[16]

            # Обновляем параметры Attention в Decoder (Self Attention)
            decoder.self_attention.W_q -= learning_rate * d_decoder[3]
            decoder.self_attention.W_k -= learning_rate * d_decoder[4]
            decoder.self_attention.W_v -= learning_rate * d_decoder[5]
            decoder.self_attention.W_o -= learning_rate * d_decoder[6]

            # Обновляем параметры Attention в Decoder (Encoder-Decoder Attention)
            decoder.enc_dec_attention.W_q -= learning_rate * d_decoder[7]
            decoder.enc_dec_attention.W_k -= learning_rate * d_decoder[8]
            decoder.enc_dec_attention.W_v -= learning_rate * d_decoder[9]
            decoder.enc_dec_attention.W_o -= learning_rate * d_decoder[10]

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.4f}")

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
