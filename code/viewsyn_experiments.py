import os
import numpy as np
import viewsyn_architecture as model
import data_generators as d_gen
import viewsyn_training as train_model
import viewsyn_testing as test_model
from functools import partial
import utility as util


def run_autoencoder():
	#build architecture of network
	autoencoder = model.build_autoencoder()

	#data generator list
	f_generate_list = d_gen.generate_data_list
	f_generate_data = d_gen.generate_data_autoencoder
	
	#train autoencoder
	try:
		hist = train_model.train_network(autoencoder, f_generate_list, f_generate_data)
	finally:
		autoencoder.save('../model/weights-interrupt-autoencoder.hdf5')

	#test autoencoder
	test_loss, test_accuracy = test_model.evaluate_network(autoencoder, f_generate_list, f_generate_data)

	print "test_loss=", test_loss, ", test_accuracy=", test_accuracy


def run_transformed_autoencoder():
	#build architecture of network
	t_autoencoder = model.build_transformed_autoencoder()

	#data generator list
	f_generate_list = d_gen.generate_data_dictionary
	f_generate_data = d_gen.generate_data_trans_autoencoder
	
	train transformed autoencoder
	try:
		hist = train_model.train_network(t_autoencoder, f_generate_list, f_generate_data)
	finally:
		t_autoencoder.save('../model/weights-interrupt-t_autoencoder.hdf5')


	#test transformed autoencoder
	test_loss, test_accuracy = test_model.evaluate_network(t_autoencoder, f_generate_list, f_generate_data)
	
	print "test_loss=", test_loss, ", test_accuracy=", test_accuracy

	
def run_replication():
	#build architecture of network
	replication_net = model.build_replication_network()

	#plot network
	util.plot_nested_architecture(replication_net, '../visualize_net/replication/')
	
	#data generator list
	f_generate_list = d_gen.generate_data_dictionary
	f_generate_data = d_gen.generate_data_replication
	
	#train replication network
	try:
		hist = train_model.train_network(replication_net, f_generate_list, f_generate_data)
	finally:
		replication_net.save('../model/weights-interrupt-replication_net.hdf5')


	#test replication network
	test_loss, test_accuracy = test_model.evaluate_network(replication_net, f_generate_list, f_generate_data)
	
	print "test_loss=", test_loss, ", test_accuracy=", test_accuracy

def run_five_channel_network():
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
		five_channel_net.save('../model/weights-interrupt-five_channel_net.hdf5')


	#test five channel network
	test_loss, test_accuracy = test_model.evaluate_network(five_channel_net, f_generate_list, f_generate_data)
	
	print "test_loss=", test_loss, ", test_accuracy=", test_accuracy


if __name__ == '__main__':
	
	# run_autoencoder()
	run_transformed_autoencoder()
	# run_replication()
	# run_five_channel_network()
