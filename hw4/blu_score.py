from sklearn.model_selection import train_test_split
from nltk.translate.bleu_score import corpus_bleu
import numpy as np
from keras.datasets import imdb
import keras

max_length = 100

# topwords = 6000
(X_train, y_train), _ = imdb.load_data(num_words=None, skip_top=0, maxlen=None, start_char=1, oov_char=2, index_from=4,)

_, X2_train, _, _ = train_test_split(X_train,
                                     y_train,
                                     test_size=0.1,
                                     random_state=1,
                                     stratify=y_train,)


word_to_id = imdb.get_word_index()
word_to_id = {k: (v+4) for k, v in word_to_id.items()}
word_to_id["<PAD>"] = 0
word_to_id["<START>"] = 1
word_to_id["<OOV>"] = 2
word_to_id["<EOS>"] = 3

id_to_word = {v: k for k, v in word_to_id.items()}

tests = []
temp = ''
for x in X2_train:
    for word in x:
        temp += id_to_word[word] + ' '
    tests.append(temp)
    temp = ''

max_review_length = 100
def sample(preds, temperature=1.05):
    """Helper function to sample an index from a probability array"""
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def generate_text(model, seed="I thought", max_len=max_review_length, diversity=0.5, tag=0):
    """Generate words from a given seed"""
    result = np.zeros((max_len,))
    result[0] = word_to_id["<START>"]
    next_res_ind = 1
    seed_list = seed.split()
    for s in seed_list:
        result[next_res_ind] = word_to_id[s]
        next_res_ind = next_res_ind + 1

    # print("[" + seed + "]", end='')
    gen_text = seed
    i = 0
    if tag == 1:
        new_tag = 1
    if tag == 0:
        new_tag = 0
    if tag == 2:
        new_tag = 0
    next_word = seed[-1]
    while next_word != '<EOS>' and next_res_ind < max_len:
        i = i + 1
        if i == int(np.floor(max_len/2)):
            new_tag = 1
        model.reset_states()
        temp = np.zeros((1, max_len))
        temp[0, :] = result
        if new_tag:
            y = model.predict([temp, np.ones((1, 1))])[0][next_res_ind - 1]
        else:  # tag == 0:
            y = model.predict([temp, np.zeros((1, 1))])[0][next_res_ind - 1]
        next_word_ind = sample(y[3:], temperature=diversity)
        next_word = id_to_word[next_word_ind+3]
        result[next_res_ind] = next_word_ind+3
        next_res_ind = next_res_ind + 1
        #print(next_word + " ", end='')
        gen_text += next_word + " "
    #print()
    return gen_text

model = keras.models.load_model('best_text_gen_3.h5')

# for i in range(5):
#     result = generate_text(
#         model,
#         seed="i thought ",
#         tag=2
#     )
#     print()

# evaluate the skill of the model
def evaluate_model(model, test, max_length):
    actual, predicted = list(), list()
    # step over the whole set
    for i in range(30):
        # generate description
        yhat = generate_text(model, seed="i thought ", tag=i%3)
        # store actual and predicted
        references = [d.split() for d in test]
        actual.append(references)
        predicted.append(yhat.split())
    # calculate BLEU score
    print('BLEU-1: %f' % corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))
    print('BLEU-2: %f' % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))
    print('BLEU-3: %f' % corpus_bleu(actual, predicted, weights=(0.3, 0.3, 0.3, 0)))
    print('BLEU-4: %f' % corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25)))


evaluate_model(model, tests,  max_length)