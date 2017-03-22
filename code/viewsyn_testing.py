import keras

def evaluate_network(network, f_generate_list, f_generate_data):

	test_files, val_files = f_generate_list(dataPath='../data/test/', fraction_train=1.0)
	
	test_loss, test_accuracy = network.evaluate_generator(f_generate_data(test_files, 200), 100, max_q_size=10, pickle_safe=False)

	return test_loss, test_accuracy
