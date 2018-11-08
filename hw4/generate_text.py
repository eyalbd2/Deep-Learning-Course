import numpy as np
from keras.datasets import imdb
import keras

word_to_id = imdb.get_word_index()
word_to_id = {k: (v+4) for k, v in word_to_id.items()}
word_to_id["<PAD>"] = 0
word_to_id["<START>"] = 1
word_to_id["<OOV>"] = 2
word_to_id["<EOS>"] = 3

id_to_word = {v: k for k, v in word_to_id.items()}
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

    print("[" + seed + "]", end='')
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
        print(next_word + " ", end='')
    print()
    return result

model = keras.models.load_model('best_text_gen_3.h5')

for i in range(5):
    result = generate_text(
        model,
        seed="i thought ",
        tag=2
    )
    print()
