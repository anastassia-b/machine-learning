from PIL import Image
import numpy as np

BGR_MEANS = np.array([103.939, 116.779, 123.68], dtype = np.float64)

def load_image(path):
    im = Image.open(path)
    im_data = np.array(im)
    if im_data.shape[2] == 4:
        im_data = im_data[:, :, :3]
    red = np.copy(im_data[:, :, 0])
    blue = np.copy(im_data[:, :, 2])
    im_data[:, :, 0] = blue
    im_data[:, :, 2] = red
    centered_im = im_data - BGR_MEANS
    return centered_im
    # print(im_data.shape)

def load_resized_image(path):
    im = Image.open(path)
    im = im.resize((256, 256))
    im_data = np.array(im)

    #black and white image
    if len(im_data.shape) == 2:
        im_data = im_data[:, :, np.newaxis]
        im_data = np.repeat(im_data, 3, axis=2)

    if im_data.shape[2] == 4:
        im_data = im_data[:, :, :3]

    #image with alpha opacity channel
    red = np.copy(im_data[:, :, 0])
    blue = np.copy(im_data[:, :, 2])
    im_data[:, :, 0] = blue
    im_data[:, :, 2] = red
    centered_im = im_data - BGR_MEANS
    return centered_im

def save_image(path, im_data):
    im_data = im_data + BGR_MEANS
    #we want a copy instead of a view
    blue = np.copy(im_data[:, :, 0])
    red = np.copy(im_data[:, :, 2])
    im_data[:, :, 0] = red
    im_data[:, :, 2] = blue
    im_data = np.clip(im_data, 0, 255)
    im_data = im_data.astype(np.uint8)
    image = Image.fromarray(im_data)
    image.save(path)
