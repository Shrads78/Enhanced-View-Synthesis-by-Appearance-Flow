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