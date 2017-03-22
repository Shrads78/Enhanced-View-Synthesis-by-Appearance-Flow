from scipy.misc import imsave
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from keras.utils import visualize_util
from keras.layers import *
from bilinear_layer import Bilinear
import pdb

def save_as_image(filepath, images):
	for i in range(0, len(images)):
		filename = filepath+str(i)+".png"
		imsave(filename, images[i])

def show_image(image):
	plt.imshow(np.squeeze(image))
	plt.show()


def plot_architecture(network, filepath):
	visualize_util.plot(network, to_file=filepath)

def plot_nested_architecture(network, folderpath):
	#first plot full network
	plot_architecture(network, folderpath+'full.png')

	#pdb.set_trace()
	#now plot all layers
	for layer in network.layers:
		if type(layer) is InputLayer or type(layer) is Merge or type(layer) is Bilinear:
			continue

		name = layer.name
		plot_architecture(network.get_layer(name), folderpath+name+'.png')

def visualize_intermediate_outputs(model, layer_name, input_img):
	imsave("input_image.png", np.squeeze(input_img))
	
	#intermediate layer 
	intermediate_model = Model(input=model.input, output=model.get_layer(layer_name).output)
	intermediate_output = intermediate_model.predict(input_img)
	
	intermediate_output = np.rollaxis(np.squeeze(first_conv_output), 2)
	pdb.set_trace()
	#save_all_activations(intermediate_output, 100, "../results/first_conv/")
	
	