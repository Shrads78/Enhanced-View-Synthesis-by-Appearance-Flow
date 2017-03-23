import numpy as np

import scipy.misc
from scipy.signal import medfilt
import time
import os
import h5py
import pdb
from keras.preprocessing.image import load_img, img_to_array


from viewsyn_architecture import *


from keras.models import Sequential
from keras.layers import Convolution2D, ZeroPadding2D, MaxPooling2D, Flatten, Dense, Dropout
from keras import backend as K

# path to the model weights file.
#weights_path = 'vgg16_weights.h5'
weights_path = '../model/weights.39-376.69.hdf5'

# util function to convert a tensor into a valid image
def deprocess(x):
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1
    
    x += 0.5
    x = np.clip(x, 0, 1)
    
    x *= 255
    x = np.clip(x, 0, 255).astype('uint8')
    #x = medfilt(x,3)
    return x

def autoencoder_model(w_path):
    t_auto = build_transformed_autoencoder()
    t_auto.load_weights(w_path)
    sequential_part1 = 'sequential_1'
    sequential_part2 = 'sequential_1'
    input_layer = 'convolution2d_input_1'
    output_layer = 'convolution2d_6'
    model = Model(input=t_auto.get_layer(sequential_part1).get_layer(input_layer).input,
                  output=t_auto.get_layer(sequential_part2).get_layer(output_layer).output)
    return model


# Creates the VGG models and loads weights
model = autoencoder_model(weights_path)

# Specify input and output of the network
input_img = model.layers[0].input

# List of the generated images after learning
kept_images = []

# Update coefficient
learning_rate = 500
nsteps = 500
img_height = 224
img_width = 224
image_path = '799de8b0527ea329c725388bb41d64e3/model_views/0_20.png'

for layer_index in [1,2,3,4]: #130 flamingo, 351 hartebeest, 736 pool table, 850 teddy bear
    print('Processing filter %d' % layer_index)
    layer_output = model.layers[-layer_index].output
    start_time = time.time()
    
    # The loss is the activation of the neuron for the chosen class
    #pdb.set_trace()
    loss = K.mean(layer_output[:, :, :, 0:-1:4])
    
    # we compute the gradient of the input picture wrt this loss
    grads = K.gradients(loss, input_img)[0]
    
    grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)
    
    # this function returns the loss and grads given the input picture
    # also add a flag to disable the learning phase (in our case dropout)
    iterate = K.function([input_img, K.learning_phase()], [loss, grads])
    
    np.random.seed(1337)  # for reproducibility
    # we start from a gray image with some random noise
    #input_img_data = np.random.random((1, img_width, img_height, 3)) * 20 + 128 # (1,) for batch axis
    input_img_data = load_img(image_path, target_size=(img_height, img_width))
    input_img_data = img_to_array(input_img_data).reshape(1,224,224,3)
    
    
    # we run gradient ascent for 1000 steps
    for i in range(nsteps):
        loss_value, grads_value = iterate([input_img_data, 0]) # 0 for test phase
        input_img_data += grads_value * learning_rate # Apply gradient to image
        
        #print('Current loss value:', loss_value)
    
    # decode the resulting input image and add it to the list
    img = deprocess(input_img_data[0])
    kept_images.append((img, loss_value))
    end_time = time.time()
    #print('Filter %d processed in %ds' % (class_index, end_time - start_time))


#Compute the size of the grid
n = int(np.ceil(np.sqrt(len(kept_images))))

# build a black picture with enough space for the kept_images
img_height = 224
img_width = 224
margin = 5
height = n * img_height + (n - 1) * margin
width = n * img_width + (n - 1) * margin
stitched_res = np.zeros((height, width, 3))
#pdb.set_trace()
# fill the picture with our saved filters
for i in range(n):
    for j in range(n):
        if len(kept_images) <= i * n + j:
            break
        img, loss = kept_images[i * n + j]
        #pdb.set_trace()
        stitched_res[(img_height + margin) * i: (img_height + margin) * i + img_height,
                     (img_width + margin) * j: (img_width + margin) * j + img_width, :] = img

# save the result to disk
scipy.misc.toimage(stitched_res, cmin=0, cmax=255).save('naive_results_%dx%d.png' % (n, n)) # Do not use scipy.misc.imsave because it will normalize the image pixel value between 0 and 255
