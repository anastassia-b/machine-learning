from keras.layers import Dense, SimpleRNN
from keras.models import Sequential
from keras.utils import to_categorical
import numpy as np

BATCH_SIZE_LINES = 64
LINE_CHAR_LENGTH = 128
BATCH_CHAR_LENGTH = BATCH_SIZE_LINES * LINE_CHAR_LENGTH

with open('anna-simplified.txt', 'r') as f:
    text = f.read()

NUM_BATCHES = len(text) // BATCH_CHAR_LENGTH

text_array = np.fromstring(text, np.uint8)

text_x_array = text_array[:(NUM_BATCHES * BATCH_CHAR_LENGTH)]
text_x_matrix = text_x_array.reshape((-1, LINE_CHAR_LENGTH))
train_x = None

text_y_array = text_array[1:((NUM_BATCHES * BATCH_CHAR_LENGTH) + 1)]
text_y_matrix = text_y_array.reshape((-1, LINE_CHAR_LENGTH))
train_y = None

for i in range(BATCH_SIZE_LINES):
    batch_x = text_x_matrix[
        i::BATCH_SIZE_LINES,
        :
    ]
    batch_y = text_y_matrix[
        i::BATCH_SIZE_LINES,
        :
    ]

    if train_x is None:
        train_x = batch_x
        train_y = batch_y
    else:
        train_x = np.concatenate([train_x, batch_x], axis = 0)
        train_y = np.concatenate([train_y, batch_y], axis = 0)


print("SHAPES")
print(train_x.shape)
print(train_y.shape)

train_x = to_categorical(train_x, 256)
train_y = to_categorical(train_y, 256)

print("SHAPES")
print(train_x.shape)
print(train_y.shape)

model = Sequential()
model.add(SimpleRNN(
    units = 256,
    activation = 'relu',
    stateful = True,
    batch_input_shape = (
        BATCH_SIZE_LINES,
        LINE_CHAR_LENGTH,
        256
    ),
    return_sequences = True,
    unroll = True,
))
model.add(Dense(
    256,
    activation = 'softmax',
))

model.compile(
    loss = 'categorical_crossentropy',
    optimizer = 'adam',
    metrics = ['accuracy'],
)

model.fit(
    train_x,
    train_y,
    batch_size = BATCH_SIZE_LINES,
    epochs = 10,
)

model.save('anna.h5')
