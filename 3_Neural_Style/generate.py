from keras.models import load_model
from keras.models import Model
import pdb
import numpy as np
import tensorflow as tf
from utils import load_resized_image, save_image

model = load_model('weights.023-1.56e+04.hdf5', custom_objects={
    'tf':tf
})

# pdb.set_trace()
content_image = load_resized_image('./images/tubingen1024.jpeg')

input_tensor = model.layers[1].inputs[0]
output_tensor = model.layers[1].layers[-1].output

generator_model = Model(inputs=input_tensor, outputs=output_tensor)
generator_model.summary()

output_image = generator_model.predict(np.expand_dims(content_image, axis=0))
save_image('./results/tubingen_weights.023.jpg', output_image[0, :, :, :])
