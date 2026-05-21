#LSTM

from tensorflow.keras.layers import SimpleRNN, LSTM, GRU, Bidirectional, Dense, Embedding
from tensorflow.keras.models import Sequential
from tensorflow.keras.datasets import imdb
import numpy as np

vocab_size = 5000
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=vocab_size)

print(x_train[0])

word_idx = imdb.get_word_index()
word_idx = {i: word for word, i in word_idx.items()}
print([word_idx[i] for i in x_train[0]])

print("Max length of a review:: ", len(max((x_train+x_test), key=len)))
print("Min length of a review:: ", len(min((x_train+x_test), key=len)))

from tensorflow.keras.preprocessing import sequence

max_words = 400
embd_len = 32
x_train = sequence.pad_sequences(x_train, maxlen=max_words)
x_test = sequence.pad_sequences(x_test, maxlen=max_words)
x_valid, y_valid = x_train[:64], y_train[:64]
x_train, y_train = x_train[64:], y_train[64:]

"""LSTM(Long Short Term Memory)

"""

# Defining LSTM model
lstm_model = Sequential(name="LSTM_Model")
lstm_model.add(Embedding(vocab_size, embd_len, input_length=max_words))
lstm_model.add(LSTM(128, activation='relu',
                    return_sequences=False))
lstm_model.add(Dense(1, activation='sigmoid'))

print(lstm_model.summary())

lstm_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

history = lstm_model.fit(x_train, y_train, batch_size=64, epochs=5, verbose=2, validation_data=(x_valid, y_valid))

print()
print("LSTM model Score ---> ", lstm_model.evaluate(x_test, y_test, verbose=0))
