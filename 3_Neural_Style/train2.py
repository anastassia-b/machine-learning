from model import featurization_model
from keras.models import Model
from keras.layers import Input, Dense, Reshape
from keras.optimizers import Adam
from keras.callbacks import LambdaCallback
import keras.backend as K
from utils import load_image, load_resized_image
import numpy as np
from utils import save_image
from transformer_network import model as transformer_model
from dataset import batch_generator, num_batches


#removed content image, replacing with image batches

style_im_data = load_resized_image('./images/style_monet2.jpeg')
print("Calculating style matrices")

_, *style_values = featurization_model.predict(np.expand_dims(style_im_data, axis=0))
value_shapes = [value.shape for value in style_values]
print(value_shapes)

# target_values = [content_value, *style_values]

# instead of the reshaped_image, we will feed in the output of the generator network.

# import transformer model to for the input image
# 1. create an input (256x256x3)
input_tensor = Input(shape=(256, 256, 3))
#2. feed into transformer network
output_image = transformer_model(input_tensor)

#3. feed that into the loss network.
content_tensor, *style_tensors = featurization_model(output_image)
feature_tensors = [content_tensor, *style_tensors]
#4. that gives us content and style outputs. that's end-to-end, now we train this.

training_model = Model(inputs=input_tensor, outputs=feature_tensors)
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

training_model.fit_generator(
    #one example, one feature, the value is 1 since 1s function
    batch_generator(featurization_model, style_values),
    #batch gen gives x and y values
    steps_per_epoch=num_batches,
    epochs=3000,
    # callbacks=[LambdaCallback(on_epoch_end=save_int_image)]
)


# ls | head | parallel
