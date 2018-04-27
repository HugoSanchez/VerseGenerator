"""
I took this model from https://machinelearningmastery.com/text-generation-lstm-recurrent-neural-networks-python-keras/

All credits go to Jason Brownlee: https://machinelearningmastery.com/author/jasonb/

"""
import numpy
import string
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils

#Loading text and converting to lowercase
filename = 'nBIGsongs.txt'
raw_text = open(filename, encoding = "ISO-8859-1").read()
table = str.maketrans({key: None for key in string.punctuation})
raw_text = raw_text.translate(table).lower()

#Characters to integers conversion (1st creat set, 2nd map it):
# Set function returns "an unordered collection with no duplicate elements"
# Sorted function orders it // Enumerate function "allows us to loop over something and have an automatic counter"
chars = sorted(list(set(raw_text)))
char_to_int = dict((c, i) for i, c in enumerate(chars))
print(chars)

# Summarize dataset:
n_chars = len(raw_text)
n_vocab = len(chars)
print("Total Characters: ", n_chars)
print ("Total Vocab: ", n_vocab)

# Each training pattern of the network is comprised of 100 time steps of one character (X) followed by one character
# output (y). When creating these sequences, we slide this window along the whole book one character at a time,
# allowing each character a chance to be learned from the 100 characters that preceded it
# (except the first 100 characters of course).

seq_length = 100

dataX = []
dataY = []

for i in range(0, n_chars - seq_length, 1):
    seq_in = raw_text[i:i + seq_length]
    seq_out = raw_text[i + seq_length]
    dataX.append([char_to_int[char] for char in seq_in])
    dataY.append(char_to_int[seq_out])
n_patterns = len(dataX)
ny_patterns = len(dataY)
print("Total Patterns: ", n_patterns)
print("Total Y Patterns: ", ny_patterns)

# Transforming training data to LSTM standards [sample, time steps, features]
X = numpy.reshape(dataX, (n_patterns, seq_length, 1))
X = X / float(n_vocab)

# Hot encoding output patterns
y = np_utils.to_categorical(dataY)

# define the LSTM model

model = Sequential()
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
model.add(Dropout(0.10))
model.add(LSTM(256, return_sequences=True))
model.add(LSTM(256))
model.add(Dense(y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')

# define the checkpoint
filepath="weights-improvement-Model2-{epoch:02d}-{loss:.4f}-bigger2.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

#We finally fit the model (It's suuuuper slow)
model.fit(X, y, epochs=40, batch_size=128, callbacks=callbacks_list)
