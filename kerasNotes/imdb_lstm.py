from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Activation, Embedding
from keras.layers import LSTM
from keras.datasets import imdb

import numpy as np
np.random.seed(1337)

max_features = 20000
maxlen = 80
batch_size = 32

print "loading data..."
(X_train, y_train), (X_test, y_test) = imdb.load_data(nb_words=max_features)
print len(X_train), "train sequences"
print len(X_test), "test sequences"

print "pad the sequences (sample x time)"
X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
X_test = sequence.pad_sequences(X_test, maxlen=maxlen)
print "shape of train: ", X_train.shape
print "shape of test: ", X_test.shape

print "stacking the layers..."
model = Sequential()
model.add(Embedding(max_features, 128, dropout=0.2))
model.add(LSTM(128, dropout=0.2, dropout_U=0.2))
model.add(Dense(1))
model.add(Activation("sigmoid"))

model.compile(loss="binary_crossentropy",
              optimizer="adam"),
              metrics=["accuracy"])

print "begin to train..."
model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=20,
          validation_data=(X_test, y_test))
score, acc = model.evaluate(X_test, y_test,
                            batch_size=batch_size)

print "score", score
print "accuracy", acc
