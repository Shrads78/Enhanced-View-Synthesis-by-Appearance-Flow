from keras.preprocessing import image
from keras.models import Sequential, Model
from keras.layers import *
from keras import backend as K
from bilinear_layer import Bilinear
from keras.callbacks import *
import numpy as np
import utility as util
import data_generators as d_gen
import h5py, pdb, os
import viewsyn_architecture as model

def get_and_save_activation_outputs(output, activation_indices, folder, activation_part):

	output = np.moveaxis(output, 3, 1)
	for i in range(len(output)):
		current_output = output[i]
		filter_output = []
		
		for current_index in activation_indices:
			filter_output += [current_output[current_index]]

		util.save_as_image("../data/visualization_output/" + folder + str(i) + "_" + "activation_" + activation_part + "_", np.array(filter_output))

def load_model_and_tests(f_network, weights_path):
	net = f_network()
	net.load_weights(weights_path)

	current_chair_folder = "../data/debug_input/"
	test_data, vpt_transformation = util.load_test_image_view(current_chair_folder)

	return net, test_data, vpt_transformation

def visualize_activation_replication_network():
	weights_path = '../model/weights.00-10.00.hdf5'
	folder = "replication_network/"

	# Getting the initial network setup with loading weights.
	f_network = model.build_replication_network
	net, test_data, vpt_transformation = load_model_and_tests(f_network, weights_path)
	
	# Visualization of 6th Convolutional Filter Activation Output.	
	conv2d6_model = Model(input = net.get_layer('sequential_1').get_layer('convolution2d_input_1').input, 
						output = net.get_layer('sequential_1').get_layer('convolution2d_6').output)
	
	conv2d6_output = conv2d6_model.predict(test_data)
	activation_indices = [0, 63, 127, 191, 255, 319, 383, 447]
	get_and_save_activation_outputs(conv2d6_output, activation_indices, folder, "conv2d6")

	# Visualization of 1st Convolutional Filter Activation Output.
	conv2d1_model = Model(input = net.get_layer('sequential_1').get_layer('convolution2d_input_1').input, 
						output = net.get_layer('sequential_1').get_layer('convolution2d_1').output)

	conv2d1_output = conv2d1_model.predict(test_data)
	activation_indices = [0, 2, 4, 6, 8, 10, 12, 14]
	get_and_save_activation_outputs(conv2d1_output, activation_indices, folder, "conv2d1")

	# Building of intermediate model to get intermediate output to feed into the Deconvolutional SubNetwork.
	fc_model = Model(input = net.input, output = net.get_layer('merge_1').output)
	fc_output = fc_model.predict([test_data, vpt_transformation])

	# Visualization of 6th Deconvolutional Activation Output.
	deconv_model = Model(input = net.get_layer('sequential_3').get_layer('dense_input_2').input,
							output = net.get_layer('sequential_3').get_layer('deconvolution2d_5').output)
	
	deconv_output = deconv_model.predict(fc_output)
	activation_indices = [1, 3, 5, 7, 9, 11, 13, 15]
	get_and_save_activation_outputs(deconv_output, activation_indices, folder, "deconv2d5")


def visualize_activation_transformed_autoencoder():
	weights_path = '../model/weights.39-376.69.hdf5'
	folder = "transformed_autoencoder/"

	# Getting the initial network setup with loading weights.
	f_network = model.build_transformed_autoencoder
	net, test_data, vpt_transformation = load_model_and_tests(f_network, weights_path)
	
	# Visualization of 6th Convolutional Filter Activation Output.
	conv2d6_model = Model(input = net.get_layer('sequential_1').get_layer('convolution2d_input_1').input, 
						output = net.get_layer('sequential_1').get_layer('convolution2d_6').output)
	
	conv2d6_output = conv2d6_model.predict(test_data)
	activation_indices = [0, 63, 127, 191, 255, 319, 383, 447]
	get_and_save_activation_outputs(conv2d6_output, activation_indices, folder, "conv2d6")

	# Visualization of 1st Convolutional Filter Activation Output.
	conv2d1_model = Model(input = net.get_layer('sequential_1').get_layer('convolution2d_input_1').input, 
						output = net.get_layer('sequential_1').get_layer('convolution2d_1').output)

	conv2d1_output = conv2d1_model.predict(test_data)
	activation_indices = [0, 2, 4, 6, 8, 10, 12, 14]
	get_and_save_activation_outputs(conv2d1_output, activation_indices, folder, "conv2d1")

	# Building of intermediate model to get intermediate output to feed into the Deconvolutional SubNetwork.
	fc_model = Model(input = net.input, output = net.get_layer('merge_1').output)
	fc_output = fc_model.predict([test_data, vpt_transformation])

	# Visualization of 6th Deconvolutional Activation Output.
	deconv_model = Model(input = net.get_layer('sequential_3').get_layer('dense_input_2').input,
							output = net.get_layer('sequential_3').get_layer('deconvolution2d_5').output)
	
	deconv_output = deconv_model.predict(fc_output)
	activation_indices = [1, 3, 5, 7, 9, 11, 13, 15]
	get_and_save_activation_outputs(deconv_output, activation_indices, folder, "deconv2d5")

def visualize_activation_autoencoder():
	weights_path = '../model/sb_weights/weights.09-0.95.hdf5'
	folder = "autoencoder/"

	# Getting the initial network setup with loading weights.
	f_network = model.build_autoencoder
	net, test_data, _ = load_model_and_tests(f_network, weights_path)
	
	# Visualization of 6th Convolutional Filter Activation Output.
	conv2d6_model = Model(input = net.input, output = net.get_layer('convolution2d_6').output)
	
	conv2d6_output = conv2d6_model.predict(test_data)
	activation_indices = [0, 63, 127, 191, 255, 319, 383, 447]
	get_and_save_activation_outputs(conv2d6_output, activation_indices, folder, "conv2d6")

	# Visualization of 1st Convolutional Filter Activation Output.
	conv2d1_model = Model(input = net.input, output = net.get_layer('convolution2d_1').output)
	
	conv2d1_output = conv2d1_model.predict(test_data)
	activation_indices = [0, 2, 4, 6, 8, 10, 12, 14]
	get_and_save_activation_outputs(conv2d1_output, activation_indices, folder, "conv2d1")

	# Visualization of 6th Deconvolutional Activation Output.
	deconv_model = Model(input = net.input, output = net.get_layer('deconvolution2d_5').output)
	
	deconv_output = deconv_model.predict(test_data)
	activation_indices = [1, 3, 5, 7, 9, 11, 13, 15]
	get_and_save_activation_outputs(deconv_output, activation_indices, folder, "deconv2d5")


if __name__ == '__main__':

	# visualize_activation_replication_network()
	# visualize_activation_transformed_autoencoder()
	visualize_activation_autoencoder()
