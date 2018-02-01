from config import *
# from keras.utils import to_categorical HAHA
import numpy as np

def to_categorical(i, length):
    one_hot_encoding = np.zeros(length)
    one_hot_encoding[i] = 1
    return one_hot_encoding

with open("./jane_austen_text.txt", "r") as jane_text_file:
    jane_text = jane_text_file.read()

    # instead of map, list comprehension
    #instead of just ord, we want a one-hot array.
    jane_int_text = [
        to_categorical(ord(char), NUM_CHARS)
        for char in jane_text
    ]
    
    length = len(jane_int_text) // BATCH_SIZE # we want an integer instead of a float
    sub_texts = []
    for i in range(BATCH_SIZE):
        sub_texts.append(jane_int_text[(length*i):(length*(i+1))])

    num_batches = length // BATCH_STRING_LENGTH
    batches = []

    for i in range(num_batches):
        batch = []

        for j in range(BATCH_SIZE):
            batch.append(sub_texts[j][(BATCH_STRING_LENGTH*i):(BATCH_STRING_LENGTH*(i+1))])

        batches.append(batch)

    batches = np.array(batches)
    print(batches.shape)
