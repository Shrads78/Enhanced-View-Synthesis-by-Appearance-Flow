import os
import numpy as np
import viewsyn_architecture as model
import data_generators as d_gen
import viewsyn_training as train_model
from functools import partial

def load_test_data(path='../data/test/'):
	#TODO:
	return None


def run_autoencoder(test_images):
	#build architecture of network
	autoencoder = model.build_autoencoder()

	#data generator list
	f_generate_list = d_gen.generate_data_list
	f_generate_data = d_gen.generate_data_autoencoder
	
	#train autoencoder
	try:
		hist = train_model.train_network(autoencoder, f_generate_list, f_generate_data)
	finally:
		autoencoder.save('../model/weights-interrupt-autoencoder.' + str(len(os.listdir('../model/'))) + '.hdf5')

	#test autoencoder
	

def run_transformed_autoencoder(test_images):
	#build architecture of network
	t_autoencoder = model.build_transformed_autoencoder()

	#data generator list
	f_generate_list = d_gen.generate_data_dictionary
	f_generate_data = d_gen.generate_data_trans_autoencoder
	
	#train transformed autoencoder
	try:
		hist = train_model.train_network(t_autoencoder, f_generate_list, f_generate_data)
	finally:
		t_autoencoder.save('../model/weights-interrupt-t_autoencoder.' + str(len(os.listdir('../model/'))) + '.hdf5')


	#test transformed autoencoder

	
def run_replication(test_images):
	#build architecture of network
	replication_net = model.build_replication_network()

	#data generator list
	f_generate_list = d_gen.generate_data_dictionary
	f_generate_data = d_gen.generate_data_replication
	
	#train replication network
	try:
		hist = train_model.train_network(replication_net, f_generate_list, f_generate_data)
	finally:
		replication_net.save('../model/weights-interrupt-replication_net.' + str(len(os.listdir('../model/'))) + '.hdf5')


	#test replication network

def run_five_channel_network(test_images):
	#build architecture of network
	five_channel_net = model.build_five_channel_network()

	#data generator list
	f_generate_list = d_gen.generate_data_dictionary
	f_generate_data = partial(d_gen.generate_data_replication, first_output_name='sequential_2')
	
	#load pre trained ../model/weights from autoencoder
	#train_model.load_autoenocoder_model_../model/weights(five_channel_net, '../model/../model/weights.29-377.08.hdf5')
	
	#train five channel network
	try:
		hist = train_model.train_network(five_channel_net, f_generate_list, f_generate_data)
	finally:
		five_channel_net.save('../model/weights-interrupt-five_channel_net.' + str(len(os.listdir('../model/'))) + '.hdf5')



	#test five channel network

if __name__ == '__main__':
	test_images = load_test_data()

	# run_autoencoder(test_images)
	run_transformed_autoencoder(test_images)
	# run_replication(test_images)
	# run_five_channel_network(test_images)
