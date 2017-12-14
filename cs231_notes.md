# Convolutional Neural Networks
Very similar to ordinary NN except: ConvNet architectures make the explicit assumption that the inputs are images, which allows us to encode certain properties into the architecture. These then make the forward function more efficient to implement and vastly reduce the amount of parameters in the network.

## Architecture Overview

Regular Neural Nets don’t scale well to full images. In CIFAR-10, images are only of size 32x32x3 (32 wide, 32 high, 3 color channels), so a single fully-connected neuron in a first hidden layer of a regular Neural Network would have 32*32*3 = 3072 weights.

## CNN Layers
Example architecture: INPUT -> CONV -> RELU -> POOL -> FC

### Convolutional Layer
Three hyperparameters control the size of the output volume: the depth, stride and zero-padding.
Depth -> number of filters
Stride -> size which we slide the filter
Zero-padding -> padding the input volumes with zeroes on the borders.

### Pooling Layer


### Normalization Layer


### Fully-Connected Layer
It is worth noting that the only difference between FC and CONV layers is that the neurons in the CONV layer are connected only to a local region in the input, and that many of the neurons in a CONV volume share parameters. However, the neurons in both layers still compute dot products, so their functional form is identical.

### Converting Fully-Connected Layers to Convolutional Layers


## CNN Architectures

### Layer Patterns
The most common form of a ConvNet architecture stacks a few CONV-RELU layers, follows them with POOL layers, and repeats this pattern until the image has been merged spatially to a small size. At some point, it is common to transition to fully-connected layers. The last fully-connected layer holds the output, such as the class scores.

Prefer a stack of small filter CONV to one large receptive field CONV layer.
