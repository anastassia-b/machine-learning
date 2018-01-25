from keras.models import Model
from keras.layers import Input, Dense, Reshape, Lambda, Conv2D, BatchNormalization, Cropping2D, Add, Activation
import tensorflow as tf
import numpy as np

#input is actually a tensor, not layer
input_tensor = Input((256, 256, 3))

def padding(tensor):
    return tf.pad(
        tensor,
        [[0, 0], [40, 40], [40, 40], [0, 0]], #we only want to pad the middle 2 (not batch or the channels)
        mode='SYMMETRIC',
    )

padding_layer = Lambda(padding)
padded_input = padding_layer(input_tensor)

conv_layer1 = Conv2D(
    32, (9, 9), strides=(1, 1), padding='valid', activation = 'relu'
)

conv_output1 = conv_layer1(padded_input)

conv_layer2 = Conv2D(
    64, (3, 3), strides=(2, 2), padding='valid', activation = 'relu'
)

conv_output2 = conv_layer2(conv_output1)

conv_layer3 = Conv2D(
    128, (3, 3), strides=(2, 2), padding='valid', activation = 'relu'
)

conv_output3 = conv_layer3(conv_output2)

def residual_block(input_tensor):
    r_conv_layer1 = Conv2D(
        128, (3, 3), strides=(1, 1), padding='valid', activation = 'linear'
    )
    r_conv_output1 = r_conv_layer1(input_tensor)
    r_bn_output1 = BatchNormalization()(r_conv_output1)
    r_bn_output1 = Activation('relu')(r_bn_output1)

    #we want different weights
    r_conv_layer2 = Conv2D(
        128, (3, 3), strides=(1, 1), padding='valid', activation = 'linear'
    )

    r_conv_output2 = r_conv_layer2(r_bn_output1)
    r_bn_output2 = BatchNormalization()(r_conv_output2)

    cropped_input = Cropping2D(cropping=((2, 2), (2, 2)))(input_tensor)
    return Add()([cropped_input, r_bn_output2])


residual1 = residual_block(conv_output3)
residual2 = residual_block(residual1)
residual3 = residual_block(residual2)
residual4 = residual_block(residual3)
residual5 = residual_block(residual4)

#we want to resize the image (bigger) then a normal convolution

scaled_output1 = Lambda(lambda input_tensor: tf.image.resize_bilinear(
    input_tensor,
    (128, 128)
))(residual5)

scaled_output1 = Conv2D(
    64,
    (3, 3),
    strides=(1, 1),
    padding='SAME',
    activation="relu"
)(scaled_output1)

scaled_output2 = Lambda(lambda input_tensor: tf.image.resize_bilinear(
    input_tensor,
    (256, 256)
))(scaled_output1)

scaled_output2 = Conv2D(
    32,
    (3, 3),
    strides=(1, 1),
    padding='SAME',
    activation="relu"
)(scaled_output2)

scaled_output3 = Conv2D(
    3,
    (9, 9),
    strides=(1, 1),
    padding='SAME',
    activation="relu"
)(scaled_output2)

model = Model(inputs=input_tensor, outputs=scaled_output3)
model.summary()


#input for the loss model will be the output of the generator. in the model.py
#get coco dataset.

# generator = transformer used here interchangably.
