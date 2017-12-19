# Thanks Ned!
# Source: https://gist.github.com/ruggeri/d939b1630b6d51b5c1e60b3e006d6ff3
# Resource: https://keras.io/getting-started/functional-api-guide/

import keras.backend as K
from keras.datasets import mnist
from keras.layers import Dense, Input, Lambda
from keras.models import Model
from keras.optimizers import Adam
from keras.utils import to_categorical
import numpy as np

# Create an input layer, which allocates a tf.placeholder tensor:
input_tensor = Input(shape = (28, 28))

# We could use a Keras Flatten layer like so:
# from keras.layers import Flatten
# flattened_input = Flatten()(input_tensor)

# But instead, we will create a "Lambda layer"-- a custom layer we can do TF stuff in:
flatten_layer = Lambda(
    lambda ipt: K.reshape(ipt, (-1, 28 * 28))
)

flattened_input = flatten_layer(input_tensor)

# Create a dense layer, feed it in the input_tensor:
hidden_layer = Dense(10, activation = 'relu')
hidden_output = hidden_layer(flattened_input)
# More common to write^ as:
# hidden_output = Dense(10, activation = 'relu')(flattened_input)

output_layer = Dense(10, activation = 'softmax')
classification_output = output_layer(hidden_output)
# classification_output = Dense(10, activation = 'softmax')(hidden_output)

# We specify what tensors will be fed in, adn which are the final outputs:
model = Model([input_tensor], [classification_output])

# Build the optimizer
LEARNING_RATE = 0.001
optimizer = Adam(lr = LEARNING_RATE)

# "custom" loss function
def categorical_cross_entropy(y_true, y_pred):
    #should be TF tensor objects coming into this function.
    print(y_true)
    print(y_pred)

    # K (backend) functions typically manipulate TF tensors directly.
    # Use K.epsilon() so we don't take the log of zero (NaN).
    # This wouldn't be a problem if we were working in logits, but we used softmax in the final layer.
    # in TensorFlow domain, logits are the values to be used as input to softmax.

    return K.mean(
        K.sum(y_true * -K.log(y_pred + K.epsilon()), axis = 1)
    )

model.compile(
    #this just says to use our cross_entropy function
    #you can have multiple outputs, in which case you would specify multiple loss fxs
    loss = [categorical_cross_entropy],
    loss_weights = [1.0],
    optimizer = optimizer,
    metrics = ['accuracy'],
)

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_mean = np.mean(x_train)
x_stddev = np.std(x_train)

#normalize
x_train = (x_train - x_mean) / x_stddev
x_test = (x_test - x_mean) / x_stddev

#one hot encode
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

model.fit(
    x_train,
    y_train,
    validation_data = (x_test, y_test),
    epochs = 3
)
