import os
from keras.preprocessing import image
import numpy as np
import pdb
import random
import itertools
from keras.utils.np_utils import *

data_folder_chair = "../data/chairs/"

def read_image(filename, target_size):
	img = image.load_img(filename, target_size = target_size)
	img = image.img_to_array(img)
	return img

#TODO: To read image only when required, Check for one_hot_encoding if it is working properly
def read_data_viewsyn(target_size, test_split_ratio):
	train_images_input = []
	train_images_output = []
	train_images_T = []

	test_images_input = []
	test_images_output = []
	test_images_T = []

	for image_folder in os.listdir(data_folder_chair):

		# Current folder to get the images
		current_folder = data_folder_chair + image_folder + '/model_views/'
		
		# Store all the images of same elevation in a dict
		elevation_dict = {}
		
		for current_file in os.listdir(current_folder):
			# pdb.set_trace()
			azi, ele = map(int, current_file.replace('.png', '').split('_'))
			ele /= 5
			azi /= 20
			if ele not in elevation_dict:
				elevation_dict[ele] = []
			elevation_dict[ele].append( [(current_folder + current_file, target_size), azi] )

		# pdb.set_trace()
		for ele in elevation_dict:
			random.shuffle(elevation_dict[ele])
			all_tuples = list(itertools.combinations(elevation_dict[ele], 2))
			
			l = int(test_split_ratio * len(all_tuples))
			test_split = all_tuples[:l]
			train_split = all_tuples[l:]

			# pdb.set_trace()
			for data_pair in train_split:
				train_images_input.append(read_image(data_pair[0][0], target_size))
				train_images_output.append(read_image(data_pair[1][0], target_size))
				train_images_T.append(data_pair[1][1] - data_pair[0][1])

			for data_pair in test_split:
				test_images_input.append(read_image(data_pair[0][0], target_size))
				test_images_output.append(read_image(data_pair[1][0], target_size))
				test_images_T.append(data_pair[1][1] - data_pair[0][1])

	return np.array(train_images_input), np.array(train_images_output), to_categorical(train_images_T), np.array(test_images_input), np.array(test_images_output), to_categorical(test_images_T)

def read_data_autoencoder(target_size, test_split_ratio):
	images_fname = []

	for image_folder in os.listdir(data_folder_chair):
		if image_folder == '.DS_Store': continue
		# Current folder to get the images
		current_folder = data_folder_chair + image_folder + '/model_views/'
		
		for current_file in os.listdir(current_folder):
			if current_file == '.DS_Store': continue
			images_fname.append(current_folder + current_file)


	random.shuffle(images_fname)
	
	l = int(test_split_ratio * len(images_fname))
	test_images_fname = images_fname[:l]
	train_images_fname = images_fname[l:]

	test_images = [read_image(current_image, target_size) for current_image in test_images_fname]
	train_images = [read_image(current_image, target_size) for current_image in train_images_fname]

	return np.array(train_images), np.array(test_images)


if __name__ == '__main__':
	train, test = read_data_autoencoder((224, 224), 0.2)
	np.save('../data/train_images_autoencoder.npy', train)
	np.save('../data/test_images_autoencoder.npy', test)