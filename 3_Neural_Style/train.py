from model import featurization_model
from keras.models import Model
from keras.layers import Input, Dense, Reshape
from keras.optimizers import Adam
from keras.callbacks import LambdaCallback
import keras.backend as K
from utils import load_image
import numpy as np
from utils import save_image

content_im_data = load_image('./images/content_milan.jpeg')

print("Calculating content features")
#we only had 1 image to train, but we needed 2 dim, 1 * image dimensions
content_value, *_ = featurization_model.predict(np.expand_dims(content_im_data, axis=0))
print(content_value.shape)

style_im_data = load_image('./images/style_monet2.jpeg')
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

# instead of the reshaped_image, we will feed in the output of the generator network.

content_tensor, *style_tensors = featurization_model(reshaped_image)
feature_tensors = [content_tensor, *style_tensors]

training_model = Model(inputs=dummy_input_tensor, outputs=feature_tensors)
training_model.summary()

def save_int_image(epoch_idx, logs):
    if epoch_idx % 100 == 99:
        flattened_image_data = image_layer.get_weights()[0]
        # flattened_image_data = K.eval(flattened_image_tensor)
        image_data = np.reshape(flattened_image_data, (768, 1024, 3))
        save_image(f'./images/result{epoch_idx:04}.jpeg', image_data)


optimizer = Adam(lr=10.0) #changed to 10 from 0.001
training_model.compile(
    loss='mean_squared_error',
    optimizer=optimizer,
    loss_weights=[2.5, *([1]*5)])
training_model.fit(
    #one example, one feature, the value is 1 since 1s function
    np.ones([1, 1]),
    target_values,
    batch_size=1,
    epochs=3000,
    verbose=2,
    callbacks=[LambdaCallback(on_epoch_end=save_int_image)]
)
