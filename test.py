import numpy as np
from keras.layers import Dense, InputLayer, Embedding, Activation, LSTM,Bidirectional
from keras.models import Sequential
from keras.optimizers import Adam
import numpy as np
import pickle

word2index=pickle.load(open('words.pkl','rb'))
tag2index=pickle.load(open('tags.pkl','rb'))
test_x=pickle.load(open('test_x','rb'))
test_y=pickle.load(open('test_y','rb'))


def to_categorical(sequences, categories):
    cat_sequences = []
    for s in sequences:
        cats = []
        for item in s:
            cats.append(np.zeros(categories))
            cats[-1][item] = 1.0
        cat_sequences.append(cats)
    return np.array(cat_sequences)

model = Sequential()
model.add(InputLayer(input_shape=(210,)))
model.add(Embedding(len(word2index), 128))
model.add(Bidirectional(LSTM(units=128, return_sequences=True,recurrent_dropout=0.2, dropout=0.2)))
model.add(Bidirectional(LSTM(units=128, return_sequences=True,recurrent_dropout=0.2, dropout=0.2)))
model.add(Dense(len(tag2index)))
model.add(Activation('softmax'))
model.load_weights('model/lstm.h5')
model.compile(loss='categorical_crossentropy',
              optimizer=Adam(0.001),
              metrics=['accuracy'])
scores = model.evaluate(test_x, to_categorical(test_y, len(tag2index)))
print(f"{model.metrics_names[1]}: {scores[1] * 100}")  # acc: 98.39311069478103