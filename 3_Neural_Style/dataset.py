import os
import re
from utils import load_resized_image
from PIL import Image
import numpy as np


jpg_dir = "./images"

# list files and import them
file_names = os.listdir(jpg_dir)

# reg ex to filter files that dont end in .jpg
matched_files = []
num_images = len(matched_files)

pattern = re.compile('.*jpg$')

for file_name in file_names:
    if pattern.match(file_name):
        matched_files.append(os.path.join(jpg_dir, file_name))

# training dataset passed to training function to keras
# might be too large
# so well run a generateor, give some files, then give more

batch_size = 1
num_batches = num_images / batch_size

def generator_of_file_batches():
    for idx in range(0, len(matched_files), batch_size):
        yield matched_files[idx:(idx+batch_size)]

def generator_of_loaded_images():
    for file_batch in generator_of_file_batches():
        loaded_images = []
        for file_name in file_batch:
            li = load_resized_image(file_name)
            loaded_images.append(li)
            #instead of a list of the things, give an extra dimension
        yield np.array(loaded_images)

def batch_generator(featurization_model, style_values):
    while True:
        for loaded_images in generator_of_loaded_images():
            #we need to run the network to calculate the contents
            content_value, *_ = featurization_model.predict(loaded_images)
            print(loaded_images.shape, content_value.shape, style_values[0].shape)
            yield (loaded_images, [content_value, *style_values])

#TODO: if batch size is not 1, must replicate style values.
for x in generator_of_loaded_images():
    print(x.shape)
