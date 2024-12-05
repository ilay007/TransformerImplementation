
from Transormer import TransformerModel
from TransformerCore import TransformerCore
import numpy as np





def train(model, dataset, num_epochs=10, lr=0.001):
    for epoch in range(num_epochs):
        total_loss = 0
        for src, tgt in dataset:
            tgt_shifted = tgt[1:]  # Сдвигаем целевые токены для предсказания
            logits = model.forward(src, tgt[:-1])  # Предсказания

            loss = cross_entropy_loss(logits, tgt_shifted)
            total_loss += loss

            #Обратный проход
            backward_pass(model, src, tgt, tgt_shifted, logits, lr=lr)

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(dataset)}")


def translate(model, src, max_len=20):
    src_emb = model.add_positional_encoding(model.embed(src, model.encoder_embeddings))
    for layer in model.encoder_layers:
        src_emb = layer.transformer_layer(src_emb, model.num_heads, model.d_model, model.d_ff)

    # Начинаем с токена начала последовательности
    tgt = [0]  # SOS токен
    for _ in range(max_len):
        tgt_emb = model.add_positional_encoding(model.embed(np.array(tgt), model.decoder_embeddings))
        for layer in model.decoder_layers:
            tgt_emb = layer.transformer_layer(tgt_emb, model.num_heads, model.d_model, model.d_ff)

        logits = np.matmul(tgt_emb[-1], model.output_projection)  # Прогноз для последнего токена
        next_token = np.argmax(logits)
        tgt.append(next_token)

        if next_token == 1:  # EOS токен
            break

    return tgt



def cross_entropy_loss(logits, targets):
    probs = TransformerCore.softmax(logits, axis=-1)
    log_probs = -np.log(probs + 1e-9)
    loss = np.sum(log_probs[np.arange(len(targets)), targets]) / len(targets)
    return loss

def backward_pass1(model, src, tgt, tgt_shifted, logits, lr=0.001):
    """
    Реализация обратного прохода для вычисления градиентов.
    """
    grad_output = TransformerCore.softmax(logits, axis=-1)
    grad_output[np.arange(len(tgt_shifted)), tgt_shifted] -= 1
    grad_output /= len(tgt_shifted)

    # Обновление параметров выходного слоя
    d_output_projection = np.matmul(model.decoder_embeddings.T, grad_output)
    model.output_projection -= lr * d_output_projection

    # (Обновления параметров энкодера и декодера будут аналогичными)

def backward_pass(model, src, tgt, tgt_shifted, logits, lr=0.001):
    """
    Обратное распространение и обновление параметров энкодера и декодера.

    Аргументы:
    - model: объект трансформера с энкодером и декодером
    - src: входная последовательность (batch_size, src_len, d_model)
    - tgt: целевая последовательность (batch_size, tgt_len, d_model)
    - tgt_shifted: целевая последовательность, сдвинутая на один токен (batch_size, tgt_len-1, d_model)
    - logits: выходы модели перед softmax (batch_size, tgt_len-1, vocab_size)
    - lr: скорость обучения

    Возвращает:
    - Потери (например, кросс-энтропийные).
    """
    # --- Шаг 1: Вычисление ошибки ---
    batch_size, tgt_len, vocab_size = logits.shape
    # Одноhot-кодирование целевых меток
    targets_one_hot = np.zeros_like(logits)
    targets_one_hot[np.arange(batch_size)[:, None], np.arange(tgt_len), tgt_shifted] = 1
    # Кросс-энтропийная потеря
    probs = np.exp(logits) / np.sum(np.exp(logits), axis=-1, keepdims=True)
    loss = -np.sum(targets_one_hot * np.log(probs + 1e-8)) / batch_size

    # Градиент по логитам
    d_logits = (probs - targets_one_hot) / batch_size

    # --- Шаг 2: Обратное распространение через линейный слой ---
    d_decoder_outputs = np.matmul(d_logits, model.output_embedding.T)
    d_output_embedding = np.matmul(d_logits.transpose(1, 2, 0), model.decoder_outputs).mean(axis=2)
    d_output_bias = d_logits.mean(axis=(0, 1))

    # --- Шаг 3: Обратное распространение через декодер ---
    d_tgt = d_decoder_outputs
    for layer in reversed(model.decoder.layers):
        d_tgt = layer.backward_through_decoder(layer, d_tgt, model.encoder_outputs)

    # --- Шаг 4: Обратное распространение через энкодер ---
    d_src = np.zeros_like(model.encoder_outputs)
    for layer in reversed(model.encoder.layers):
        d_src = backward_through_encoder_layer(layer, d_src, src)

    # --- Шаг 5: Обновление параметров ---
    # Для выходного слоя
    model.output_embedding -= lr * d_output_embedding
    model.output_bias -= lr * d_output_bias

    # Для слоёв энкодера и декодера
    update_encoder_parameters(model.encoder.layers, lr)
    update_decoder_parameters(model.decoder.layers, lr)

    return loss




# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # Параметры модели
    vocab_size_src = 100
    vocab_size_tgt = 100
    d_model = 64
    num_heads = 4
    num_layers = 2
    d_ff = 128
    seq_len = 10

    # Создание модели
    model = TransformerModel(vocab_size_src, vocab_size_tgt, d_model, num_heads, num_layers, d_ff, seq_len)

    # Игрушечный набор данных
    dataset = [
        (np.random.randint(0, vocab_size_src, seq_len), np.random.randint(0, vocab_size_tgt, seq_len))
        for _ in range(100)
    ]

    # Обучение
    train(model, dataset, num_epochs=5, lr=0.01)

    # Проверка
    src_example = np.random.randint(0, vocab_size_src, seq_len)
    print("Input:", src_example)
    print("Translation:", translate(model, src_example))

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
