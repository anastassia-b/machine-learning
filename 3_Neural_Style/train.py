from model import featurization_model
from keras.models import Model
from keras.layers import Input, Dense, Reshape
from keras.optimizers import Adam
import keras.backend as K
from utils import load_image
import numpy as np

content_im_data = load_image('./images/tubingen1024.jpeg')

print("Calculating content features")
content_value, *_ = featurization_model.predict(np.expand_dims(content_im_data, axis=0))
print(content_value.shape)

style_im_data = load_image('./images/starry-night1024.jpeg')
print("Calculating style matrices")

_, *style_values = featurization_model.predict(np.expand_dims(style_im_data, axis=0))
value_shapes = [value.shape for value in style_values]
print(value_shapes)

target_values = [content_value, *style_values]

#fight overfitting with regularization!
dummy_input_tensor = Input(shape=(1,))
image_layer = Dense((768 * 1024 * 3), activation='linear', use_bias=False)

image_tensor = image_layer(dummy_input_tensor)
reshaped_image = Reshape((768, 1024, 3))(image_tensor)

content_tensor, *style_tensors = featurization_model(reshaped_image)
feature_tensors = [content_tensor, *style_tensors]

training_model = Model(inputs=dummy_input_tensor, outputs=feature_tensors)
training_model.summary()

optimizer = Adam(lr=0.001)
training_model.compile(loss='mean_squared_error', optimizer=optimizer)
training_model.fit(
    #one example, one feature, the value is 1 since 1s function
    np.ones([1, 1]),
    target_values,
    batch_size=1,
    epochs=1,
    verbose=2,
    callbacks=None
)
