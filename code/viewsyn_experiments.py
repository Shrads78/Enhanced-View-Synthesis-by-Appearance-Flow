import numpy as np
import viewsyn_model as model

def load_test_data():
	#TODO
	return None

def run_autoencoder(test_images):
	autoencoder = model.build_autoencoder()

	#train autoencoder
	hist = model.train_autoencoder(autoencoder)

	#test autoencoder
	#autoencoder.load_weights('../model/weights.19-225.48.hdf5')
	#model.test_autoencoder(autoencoder, test_images)
	

if __name__ == '__main__':
	test_images = load_test_data()

	run_autoencoder(test_images)
