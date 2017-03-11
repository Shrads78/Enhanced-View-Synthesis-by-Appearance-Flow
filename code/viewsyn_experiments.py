import numpy as np
import viewsyn_model as model

def load_data():
	return False

def run_autoencoder(train_images, test_images):
	autoencoder = model.build_autoencoder()

	hist = model.train_model(autoencoder, train_images)
	
if __name__ == '__main__':
	train_images, test_images = load_data()
