from keras.layers import Dense, InputLayer, Embedding, Activation, LSTM,Bidirectional
from keras.models import Sequential
from keras.optimizers import Adam
from keras.preprocessing.sequence import pad_sequences
import pickle
import spacy
import numpy as np

tok=spacy.blank('ur')
MAX_LENGTH=210
def logits_to_tokens(sequences, index):
    token_sequences = []
    for categorical_sequence in sequences:
        token_sequence = []
        for categorical in categorical_sequence:
            token_sequence.append(index[np.argmax(categorical)])

        token_sequences.append(token_sequence)

    return token_sequences

word2index=pickle.load(open('words.pkl','rb'))
tag2index=pickle.load(open('tags.pkl','rb'))

sentences=[]
og_sent=[]

with open('urdu.txt','rb') as f:
    txt=f.read().decode('utf-8')
    toks=tok(txt)
    count=0
    index_sent=[]
    sent=[]
    for each in toks:
        word=str(each)
        if word in word2index:
            index_sent.append(word2index[word])
        else:
            index_sent.append(word2index['-OOV-'])
        sent.append(word)
        count+=1
        if count>=MAX_LENGTH:
            sentences.append(index_sent)
            og_sent.append(sent)
            sent=[]
            index_sent=[]
            count=0
    og_sent.append(sent)
    sentences.append(index_sent)


sentences=pad_sequences(sentences,maxlen=MAX_LENGTH,padding='post')



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


predictions = model.predict(sentences)
print(predictions[0][0])
tagged=(logits_to_tokens(predictions, {i: t for t, i in tag2index.items()}))

print(len(og_sent),len(tagged))
with open('pre.txt','wb') as f:
    for i,v in enumerate(og_sent):
        for x in range(0,len(v)):
            ln=v[x]+'\t'+tagged[i][x]+'\n'
            f.write(ln.encode('utf-8'))
        f.write('\n\n'.encode('utf-8'))