import numpy as np
from PIL import Image
import os
import pdb
import shutil 
from numpy import random
import math

def img_mask_gen(imgpath):
	im = Image.open(imgpath).convert('L').point(lambda x: 0 if x<=0 or x>=250 else 255,'1')
	return im	

def get_azimuth_transformation(in_path, out_path):
	in_f = in_path.split('/')[-1]
	out_f = out_path.split('/')[-1]

	in_azimuth = int(in_f.split('_')[0])
	out_azimuth = int(out_f.split('_')[0])

	angle_difference = out_azimuth - in_azimuth

	if angle_difference < 0 and abs(angle_difference) > 180:
		angle_difference += 360
	elif angle_difference > 180:
		angle_difference -= 360

	azimuth_bin = angle_difference/20 + 9
	
	azimuth_onehot = np.zeros((1,19))
	azimuth_onehot[0][azimuth_bin] = 1
	
	return azimuth_onehot

def subtract_mean(img):
	m1 = 104 * np.ones((224, 224))
	m2 = 117 * np.ones((224, 224))
	m3 = 123 * np.ones((224, 224))
	m_all = np.concatenate((m1[:,:,None], m2[:,:,None], m3[:,:,None]), axis=2)
	img = img - m_all
	return img


def generate_data_autoencoder(dataArr, batch_size):
	while 1:
		R = random.sample(range(len(dataArr)), batch_size)
		img4 = []	
		
		try:
			for r in R:	
				fp = dataArr[r]	
				currImgPath = fp
				#print currImgPath
				if '.png' in currImgPath:
					img = np.asarray(Image.open(currImgPath).convert('RGB'), dtype=np.uint8)
					#msk = imgMaskGen(currImgPath)
					img4.append(img)
		except:
			continue

		img4 = np.asarray(img4)
		yield {'image_input': img4}, {'sequential_2': img4}
		


def generate_data_trans_autoencoder(data_dict, batch_size):
	while 1:
		in_imgb = []
		out_imgb = []
		mskb = []
		view_transformationb = []
		i=0

		while i < batch_size:
			#randomly sample a model
			models = data_dict.keys()
			rand_model = random.choice(models)

			#randomly sample a elevation from the chosen model
			elevations = data_dict[rand_model].keys()
			rand_elev = random.choice(elevations)

			in_img_path, out_img_path = random.choice(data_dict[rand_model][rand_elev], 2)
			#print in_img_path, out_img_path
			view_transformation = get_azimuth_transformation(in_img_path, out_img_path)

			if np.argmax(view_transformation) > 9:
				continue

			in_img = np.asarray(Image.open(in_img_path).convert('RGB'), dtype=np.uint8)
			out_img = np.asarray(Image.open(out_img_path).convert('RGB'), dtype=np.uint8)
			
			#subtract mean
			in_img = subtract_mean(in_img)
			out_img = subtract_mean(out_img)

			msk = np.reshape(np.asarray(img_mask_gen(out_img_path)), (224, 224, 1))

			in_imgb.append(in_img)
			out_imgb.append(out_img)
			mskb.append(msk)
			view_transformationb.append(view_transformation[0])

			i += 1
			
		
		yield ({'image_input': np.asarray(in_imgb), 'view_input':np.asarray(view_transformationb)}, 
			{'sequential_3': np.asarray(out_imgb)})
		

def generate_data_replication(data_dict, batch_size, first_output_name='bilinear_1'):
	while 1:
		in_imgb = []
		out_imgb = []
		mskb = []
		view_transformationb = []

		for i in range(batch_size):
			#randomly sample a model
			models = data_dict.keys()
			rand_model = random.choice(models)

			#randomly sample a elevation from the chosen model
			elevations = data_dict[rand_model].keys()
			rand_elev = random.choice(elevations)

			in_img_path, out_img_path = random.choice(data_dict[rand_model][rand_elev], 2)
			
			view_transformation = get_azimuth_transformation(in_img_path, out_img_path)

			in_img = np.asarray(Image.open(in_img_path).convert('RGB'), dtype=np.uint8)
			out_img = np.asarray(Image.open(out_img_path).convert('RGB'), dtype=np.uint8)
			
			#subtract mean
			in_img = subtract_mean(in_img)
			out_img = subtract_mean(out_img)
			
			msk = np.reshape(np.asarray(img_mask_gen(out_img_path)), (224, 224, 1))

			in_imgb.append(in_img)
			out_imgb.append(out_img)
			mskb.append(msk)
			view_transformationb.append(view_transformation[0])
			
		# print np.asarray(in_imgb).shape, np.asarray(out_imgb).shape
		yield ({'image_input': np.asarray(in_imgb), 'view_input': np.asarray(view_transformationb)}, 
			{first_output_name: np.asarray(out_imgb), 'sequential_4': np.asarray(mskb)})


def generate_data_dictionary(dataPath='../data/train/', fraction_train=0.8):
	val_data_dict = {}
	train_data_dict = {}

	i=1
	for path,dirs,files in os.walk(dataPath):
		#print path
		for dr in dirs:
			#print dr
			if dr!='model_views' and dr != '':
				drPath = path+'/'+dr
				if '//' not in drPath:
					#print drPath
					shutil.rmtree(drPath)
			#pruning complete
			elif dr =='model_views':
				train_data_dict[i]={}
				val_data_dict[i]={}

				inpath = os.path.join(dataPath,path[len(dataPath):]) + '/'+dr
				for files in os.walk(inpath):
					for fList in files:					
						for f in fList:
							if '.png' in f:
								#find elevation of file
								elevation = int(f.split('_')[1].replace('.png', ''))

								if elevation not in train_data_dict[i]:
									train_data_dict[i][elevation] = []
									val_data_dict[i][elevation] = []

								readLoc = inpath + '/'+f
								
								train_data_dict[i][elevation].append(readLoc)
								
				#assign 20% data to val_data_dict
				for e in train_data_dict[i]:
					d = train_data_dict[i][e]
					train_data_dict[i][e] = []

					random.shuffle(d)
					split_index = int(len(d)*fraction_train)
					train_data_dict[i][e].extend(d[0:split_index])
					val_data_dict[i][e].extend(d[split_index:])

				
				i += 1

	
	#pdb.set_trace()
	return train_data_dict, val_data_dict

def generate_data_list(dataPath='../data/train/', fraction_train=0.8):
	dataArr = []
	for path,dirs,files in os.walk(dataPath):
		#print path
		for dr in dirs:
			#print dr
			if dr!='model_views' and dr != '':
				drPath = path+'/'+dr
				if '//' not in drPath:
					#print drPath
					shutil.rmtree(drPath)
			#pruning complete
			elif dr =='model_views':
				inpath = os.path.join(dataPath,path[len(dataPath):]) + '/'+dr
				for files in os.walk(inpath):
					for fList in files:					
						for f in fList:
							if '.png' in f:
								readLoc = inpath + '/'+f
								#print readLoc
								dataArr.append(readLoc)

	dataArr = np.asarray(dataArr)
	
	np.random.shuffle(dataArr)

	return dataArr[:int(math.ceil(fraction_train*len(dataArr)))], dataArr[int(math.ceil(fraction_train*len(dataArr))):]
