import os
from keras.preprocessing import image
import numpy as np
import pdb
import random
import itertools

data_folder_chair = "./chairs/"

def read_image(filename, target_size):
	img = image.load_img(filename, target_size = target_size)
	img = image.img_to_array(img)
	return img.reshape((1,)+img.shape)

def read_data(target_size, test_split_ratio):
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
			if ele not in elevation_dict:
				elevation_dict[ele] = []
			elevation_dict[ele].append( [read_image(current_folder + current_file, target_size), azi] )

		# pdb.set_trace()
		for ele in elevation_dict:
			random.shuffle(elevation_dict[ele])
			all_tuples = list(itertools.combinations(elevation_dict[ele], 2))
			
			test_split = all_tuples[:int(test_split_ratio * len(all_tuples))]
			train_split = all_tuples[int(test_split_ratio * len(all_tuples)):]

			# pdb.set_trace()
			for data_pair in train_split:
				train_images_input.append(data_pair[0][0])
				train_images_output.append(data_pair[1][0])
				train_images_T.append(data_pair[1][1] - data_pair[0][1])

			for data_pair in test_split:
				test_images_input.append(data_pair[0][0])
				test_images_output.append(data_pair[1][0])
				test_images_T.append(data_pair[1][1] - data_pair[0][1])

	return np.array(train_images_input), np.array(train_images_output), np.array(train_images_T), np.array(test_images_input), np.array(test_images_output), np.array(test_images_T)


if __name__ == '__main__':
	lists = read_data((224, 224, 3), 0.2)
	for current_list in lists:
		print len(lists)