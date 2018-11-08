import numpy as np
from sklearn.model_selection import train_test_split
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding
from keras.preprocessing import sequence
import string
from keras.utils import to_categorical
import keras
from keras.layers import Dense, MaxPool2D, Conv2D, Dropout, TimeDistributed, LSTM, Flatten, Input, Add, Merge
from keras.layers.wrappers import Bidirectional
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential, Model
from keras.utils.vis_utils import plot_model
from keras.layers.merge import add
import matplotlib.pyplot as plt
import os


(X_train, y_train), _= imdb.load_data()

X_train, X_test, x2_train, y_test = train_test_split(X_train, y_train,
                                                    test_size=0.3,
                                                    stratify=y_train)

word_to_id = imdb.get_word_index()
word_to_id = {k: (v+3) for k, v in word_to_id.items()}
word_to_id["<PAD>"] = 0
word_to_id["<START>"] = 1
word_to_id["<OOV>"] = 2
id_to_word = {v: k for k, v in word_to_id.items()}

print("Done loading data")

X_train_sen = []
for i in range(X_train.shape[0]):
    temp = ''
    for j in X_train[i]:
        temp += id_to_word.get(j)+' '
    X_train_sen.append(temp)
print("Done turning data into strings list")

# generic vocabulary
characters = ['<PAD>', '<START>', '<EOS>'] + list(string.printable)  # we add pad, start, and end-of-sentence
characters.remove('\x0b')
characters.remove('\x0c')

VOCABULARY_SIZE = len(characters)
char2ind = {c:i for i,c in enumerate(characters)}
print("vocabulary len = %d" % VOCABULARY_SIZE)
print("Done tokenizing characters")

#%%
max_sentence_length = 200
sen_tokenized = [[char2ind['<START>']] + [char2ind[c] for c in review if c in char2ind] + [char2ind['<EOS>']] for review in X_train_sen]
x_train = np.array(sequence.pad_sequences(sen_tokenized,maxlen=max_sentence_length, truncating='post'))
y_train = np.roll(x_train, -1, axis=-1)  # we want to predict the next character
y_train[:, -1] = char2ind['<EOS>']
y_train = np.array([to_categorical(y, num_classes=VOCABULARY_SIZE) for y in y_train])
print("Done tokenizing the data")
print(x_train.shape)
print(y_train.shape)
print(x2_train.shape)

embedding_vecor_length = 256
lstm_dim = 256
model1 = Sequential()
model1.add(Embedding(VOCABULARY_SIZE, embedding_vecor_length, input_length=max_sentence_length))
# ml2 = LSTM(128, return_sequences=True)(ml1)

input2 = Input(shape=(1,))
l2 = Dense(64, activation='relu')(input2)
l3 = Dense(128, activation='relu')(l2)
l4 = Dense(256, activation='relu')(l3)

merge = add([l4, model1.output])
lstm1 = LSTM(lstm_dim,return_sequences=True, dropout=0.2, recurrent_dropout=0.05)(merge)
lstm2 = LSTM(lstm_dim,return_sequences=True, dropout=0.2, recurrent_dropout=0.05)(lstm1)
lstm3 = LSTM(lstm_dim,return_sequences=True, dropout=0.2, recurrent_dropout=0.05)(lstm2)

fc = TimeDistributed(Dense(VOCABULARY_SIZE, activation='softmax'))(lstm3)

model = Model([model1.input, input2], fc)

model.summary()

model.compile(
     loss='categorical_crossentropy',
     optimizer='rmsprop',
     metrics=['accuracy']
 )

history = model.fit([x_train, x2_train], y_train, epochs=40, verbose=2, batch_size=512,validation_split=0.1,)
training_loss = history.history['loss']
test_loss = history.history['val_loss']

# Create count of the number of epochs
epoch_count = range(1, len(training_loss) + 1)

# Visualize loss history
plt.plot(epoch_count, training_loss, 'r--')
plt.plot(epoch_count, test_loss, 'b-')
plt.legend(['Training Loss', 'Test Loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show();


def sample(preds, temperature=1.0):
    """Helper function to sample an index from a probability array"""
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def generate_text(model, seed="I am", max_len=200, diversity=0.5, tag=0):
    """Generate characters from a given seed"""
    result = np.zeros((max_len,))
    result[0] = char2ind['<START>']
    next_res_ind = 1
    for s in seed:
        result[next_res_ind] = char2ind[s]
        next_res_ind = next_res_ind + 1

    print("[" + seed + "]", end='')

    next_char = seed[-1]
    while next_char != '<EOS>' and next_res_ind < max_len:
        model.reset_states()
        temp = np.zeros((1, 200))
        temp[0, :] = result
        if tag:
            y = model.predict([temp, np.ones((1, 1))])[0][next_res_ind - 1]
        else:
            y = model.predict([temp, np.zeros((1, 1))])[0][next_res_ind - 1]
        next_char_ind = sample(y, temperature=diversity)
        next_char = characters[next_char_ind]
        result[next_res_ind] = next_char_ind
        next_res_ind = next_res_ind + 1
        print(next_char, end='')
    print()


for i in range(5):
    generate_text(
        model,
        seed="I "
    )
    print()

for i in range(10):
    generate_text(
        model,
        seed="The "
        , tag=1
    )
    print()