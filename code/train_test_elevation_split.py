import os
import random

def make_dir_if_not_present(dirname):
	if not os.path.exists(dirname):
		os.makedirs(dirname)
		return True
	return False

if __name__ == '__main__':
	data_folder_chair = "../data/chairs/"
	train_folder_name = "../data/train/"
	test_folder_name = "../data/test/"
	test_split_ratio = 0.2
	test_file_list = []
	train_file_list = []
	
	
	make_dir_if_not_present(train_folder_name)
	make_dir_if_not_present(test_folder_name)

	for image_folder in os.listdir(data_folder_chair):
		if image_folder == ".DS_Store" or image_folder == "trainShapes.p"  or image_folder == "testShapes.p": continue
		# Current folder to get the images
		current_folder = data_folder_chair + image_folder + '/model_views/'

		if make_dir_if_not_present(train_folder_name + image_folder) == False: continue
		make_dir_if_not_present(train_folder_name + image_folder + '/model_views/')

		if make_dir_if_not_present(test_folder_name + image_folder) == False: continue
		make_dir_if_not_present(test_folder_name + image_folder + '/model_views/')

		print "new folder = ", current_folder

		# Store all the images of same elevation in a dict
		elevation_dict = {}
		
		for current_file in os.listdir(current_folder):
			# pdb.set_trace()
			if current_file == ".DS_Store" or image_folder == "trainShapes.p"  or image_folder == "testShapes.p": continue
			azi, ele = map(int, current_file.replace('.png', '').split('_'))
			if ele not in elevation_dict:
				elevation_dict[ele] = []
			
			#Adding file path to elevation dictionary
			elevation_dict[ele].append(current_folder + current_file)

		# pdb.set_trace()
		for ele in elevation_dict:
			current_dict = elevation_dict[ele]
			random.shuffle(current_dict)
			
			l = int(test_split_ratio * len(current_dict))
			test_split = current_dict[:l]
			train_split = current_dict[l:]

			# pdb.set_trace()
			for file_path in test_split:
				os.system("cp " + file_path + " " + test_folder_name + image_folder + '/model_views/')
				test_file_list += [file_path]

			for file_path in train_split:
				os.system("cp " + file_path + " " + train_folder_name + image_folder + '/model_views/')
				train_file_list += [file_path]
