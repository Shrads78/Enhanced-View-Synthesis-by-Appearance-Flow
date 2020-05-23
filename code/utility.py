from scipy.misc import imsave
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from keras.utils import visualize_util
from keras.preprocessing import image
from keras.layers import *
from bilinear_layer import Bilinear
import data_generators as d_gen
import pdb, os

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

def load_test_image_view(current_chair_folder):
	img = []
	vpt_transformation = []
	vpt_array = np.zeros((19))
	cur_idx = 0
	for filename in os.listdir(current_chair_folder):
		if '.png' not in filename: continue
		# Getting image
		im = image.img_to_array(image.load_img((current_chair_folder + filename)))
		img.append(np.asarray(d_gen.subtract_mean(im)))
		# Making a viewpoint transformation
		tmp = vpt_array
		tmp[cur_idx] = 1
		vpt_transformation += [tmp]
		cur_idx += 1
		cur_idx %= 19

	return np.array(img), np.array(vpt_transformation)

def load_data_bilinear(current_chair_folder):
	img = []

	for filename in os.listdir(current_chair_folder):
		if '.png' not in filename: continue
		im = image.img_to_array(image.load_img((current_chair_folder + filename)))
		# pdb.set_trace()
		x = np.zeros((224, 224,1))
		y = np.zeros((224, 224,1))
		for i in range(224):
			for j in range(224):
				x[i][j][0] = j * 1.2
				y[i][j][0] = i * 1.2

		im = np.concatenate((im, y, x), axis = 2)
		img.append(np.asarray(im))
		
	return np.array(img)