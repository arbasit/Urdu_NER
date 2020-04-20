import ast
import codecs
import json
import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Dense, InputLayer, Embedding, Activation, LSTM,Bidirectional
from keras.models import Sequential
from keras.optimizers import Adam
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import numpy as np
import pickle



def logits_to_tokens(sequences, index):
    token_sequences = []
    for categorical_sequence in sequences:
        token_sequence = []
        for categorical in categorical_sequence:
            token_sequence.append(index[np.argmax(categorical)])

        token_sequences.append(token_sequence)

    return token_sequences


def to_categorical(sequences, categories):
    cat_sequences = []
    for s in sequences:
        cats = []
        for item in s:
            cats.append(np.zeros(categories))
            cats[-1][item] = 1.0
        cat_sequences.append(cats)
    return np.array(cat_sequences)


def get_words(sentences):
    words = set([])
    for sentence in sentences:
        for word in sentence:
            words.add(word)
    return words


def get_tags(sentences_tags):
    tags = set([])
    for tag in sentences_tags:
        for t in tag:
            tags.add(t)
    return tags


def get_train_sentences_x(train_sentences, word2index):
    train_sentences_x = []
    for sentence in train_sentences:
        sentence_index = []
        for word in sentence:
            try:
                sentence_index.append(word2index[word])
            except KeyError:
                sentence_index.append(word2index['-OOV-'])

        train_sentences_x.append(sentence_index)
    return train_sentences_x


def get_test_sentences_x(test_sentences, word2index):
    test_sentences_x = []
    for sentence in test_sentences:
        sentence_index = []
        for word in sentence:
            try:
                sentence_index.append(word2index[word])
            except KeyError:
                sentence_index.append(word2index['-OOV-'])
        test_sentences_x.append(sentence_index)
    return test_sentences_x


def get_train_tags_y(train_tags, tag2index):
    train_tags_y = []
    for tags in train_tags:
        train_tags_y.append([tag2index[t] for t in tags])
    return train_tags_y


def get_test_tags_y(test_tags, tag2index):
    test_tags_y = []
    for tags in test_tags:
        test_tags_y.append([tag2index[t] for t in tags])
    return test_tags_y


tagged_sentences = codecs.open("data/data.txt", encoding="utf-8").readlines()

print(tagged_sentences[0])
print("Tagged sentences: ", len(tagged_sentences))

sentences, sentence_tags = [], []
for i,tagged_sentence in enumerate(tagged_sentences):
    sentence, tags = zip(*ast.literal_eval(tagged_sentence.strip()))
    sentences.append(np.array(sentence))
    sentence_tags.append(np.array(tags))


words = get_words(sentences)
tags = get_tags(sentence_tags)

word2index = {w: i + 2 for i, w in enumerate(list(words))}
word2index['-PAD-'] = 0
word2index['-OOV-'] = 1

tag2index = {t: i + 1 for i, t in enumerate(list(tags))}
tag2index['-PAD-'] = 0

with open('words.pkl','wb') as f:
    pickle.dump(word2index,f)

with open('tags.pkl','wb') as f:
    pickle.dump(tag2index,f)

(train_sentences,
 test_sentences,
 train_tags,
 test_tags) = train_test_split(sentences, sentence_tags, test_size=0.2)

train_sentences_x = get_train_sentences_x(train_sentences, word2index)
train_tags_y = get_train_tags_y(train_tags, tag2index)

test_sentences_x = get_test_sentences_x(test_sentences, word2index)
test_tags_y = get_test_tags_y(test_tags, tag2index)


MAX_LENGTH = len(max(train_sentences_x, key=len))


train_sentences_x = pad_sequences(train_sentences_x, maxlen=MAX_LENGTH, padding='post')
train_tags_y = pad_sequences(train_tags_y, maxlen=MAX_LENGTH, padding='post')
test_sentences_x = pad_sequences(test_sentences_x, maxlen=MAX_LENGTH, padding='post')
test_tags_y = pad_sequences(test_tags_y, maxlen=MAX_LENGTH, padding='post')

pickle.dump(test_sentences_x,open('test_x','wb'))
pickle.dump(test_tags_y,open('test_y','wb'))

with open('model.txt','w') as f:
    st='Input('+str(MAX_LENGTH)+')\n'+'Embedding('+str(len(word2index))+',128)\n'+'Dense('+str(len(tag2index))+')'
    f.write(st)

model = Sequential()
model.add(InputLayer(input_shape=(MAX_LENGTH,)))
model.add(Embedding(len(word2index), 128))
model.add(Bidirectional(LSTM(units=128, return_sequences=True,recurrent_dropout=0.2, dropout=0.2)))
model.add(Bidirectional(LSTM(units=128, return_sequences=True,recurrent_dropout=0.2, dropout=0.2)))
model.add(Dense(len(tag2index)))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(0.001),
              metrics=['accuracy'])
history = model.fit(train_sentences_x, to_categorical(train_tags_y, len(tag2index)), batch_size=32, epochs=10,
                    validation_split=0.2).history
model.save("model/lstm.h5")
model.summary()


