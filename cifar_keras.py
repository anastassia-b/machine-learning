from keras.datasets import cifar10
from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from keras.models import Sequential
from keras.optimizers import Adam
from keras.utils import to_categorical

(train_x, train_y), (test_x, test_y) = cifar10.load_data()

train_x = (train_x - 128) / 128
test_x = (test_x - 128) / 128

train_y = to_categorical(train_y)
test_y = to_categorical(test_y)

model = Sequential()
model.add(Conv2D(
    16,
    (3, 3),
    activation = 'relu',
    input_shape = (32, 32, 3)
))
model.add(MaxPooling2D(
    (2, 2)
))

model.add(Flatten(
))
model.add(Dense(
    10,
    activation = 'softmax'
))

LEARNING_RATE = 0.001
optimizer = Adam(lr = LEARNING_RATE)
model.compile(
    loss = 'categorical_crossentropy',
    optimizer = optimizer,
    metrics = ['accuracy'],
)

model.fit(
    train_x,
    train_y,
    epochs = 100,
    batch_size = 100,
)
