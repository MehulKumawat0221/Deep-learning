#GRU

from tensorflow.keras.layers import SimpleRNN, LSTM, GRU, Bidirectional, Dense, Embedding
from tensorflow.keras.models import Sequential
from tensorflow.keras.datasets import imdb
import numpy as np

vocab_size = 5000
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=vocab_size)

print(x_train[0])

#Getting all the words from word_index dictionary
word_idx = imdb.get_word_index()
#Originally the index number of a value and not a key,
#hence converting the index as key and the wordd as value
word_idx = {i: word for word, i in word_idx.items()}
# again printunf the review
print([word_idx[i] for i in x_train[0]])

# Get the minimum and the maximum length of reviews
print("Max length of a review:: ", len(max((x_train+x_test), key=len)))
print("Min length of a review:: ", len(min((x_train+x_test), key=len)))

from tensorflow.keras.preprocessing import sequence

# keeping a fixed length of all reviews to max words
max_words = 400
embd_len = 32
x_train = sequence.pad_sequences(x_train, maxlen=max_words)
x_test = sequence.pad_sequences(x_test, maxlen=max_words)
x_valid, y_valid = x_train[:64], y_train[:64]
x_train, y_train = x_train[64:], y_train[64:]

# Defining GRU model
gru_model = Sequential(name="GRU_Model")
gru_model.add(Embedding(vocab_size, embd_len, input_length=max_words))
gru_model.add(GRU(128, activation='tanh', return_sequences=False))
gru_model.add(Dense(1, activation='sigmoid'))

#printing the Summary
print(gru_model.summary())

# compiling the model
gru_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# training the GRU model
history2 = gru_model.fit(x_train, y_train, batch_size=64, epochs=5, verbose=2, validation_data=(x_valid, y_valid))

# printing model score on test data
print()

print("GRU model Score ---> ", gru_model.evaluate(x_test, y_test, verbose=0))

gru_model = Sequential(name="GRU_Model")
gru_model.add(Embedding(input_dim=vocab_size, output_dim=embd_len, input_shape=(max_words,)))
gru_model.add(GRU(128, activation='tanh', return_sequences=False))
gru_model.add(Dense(1, activation='sigmoid'))

gru_model.build()   # builds the model explicitly
print(gru_model.summary())

