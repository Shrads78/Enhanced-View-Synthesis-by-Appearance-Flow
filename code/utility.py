from scipy.misc import imsave
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from PIL import Image
import os
import pdb
import shutil 
import random

def save_as_image(filepath, images):
	for i in range(0, len(images)):
		filename = filepath+str(i)+".png"
		imsave(filename, images[i])

def show_image(image):
	plt.imshow(np.squeeze(image))
	plt.show()

def img_mask_gen(imgpath):
	im = Image.open(imgpath).convert('L').point(lambda x: 0 if x<=0 or x>=250 else 255,'1')
	return im	

def get_azimuth_transformation(in_path, out_path):
	in_f = in_path.split('/')[-1]
	out_f = out_path.split('/')[-1]

	in_azimuth = int(in_f.split('_')[0]) / 20
	out_azimuth = int(out_f.split('_')[0]) / 20
	azimuth_bin = (in_azimuth - out_azimuth) % 19
	
	azimuth_onehot = np.zeros((1,19))
	azimuth_onehot[0][azimuth_bin] = 1
	
	return azimuth_onehot

def generate_data_from_list(data_dict, batch_size):
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

			in_img_path, out_img_path = random.sample(data_dict[rand_model][rand_elev], 2)
			
			view_transformation = get_azimuth_transformation(in_img_path, out_img_path)

			if np.argmax(view_transformation) > 9:
				continue

			in_img = np.asarray(Image.open(in_img_path).convert('RGB'), dtype=np.uint8)
			out_img = np.asarray(Image.open(out_img_path).convert('RGB'), dtype=np.uint8)
			
			msk = np.reshape(np.asarray(img_mask_gen(out_img_path)), (224, 224, 1))

			in_imgb.append(in_img)
			out_imgb.append(out_img)
			mskb.append(msk)
			view_transformationb.append(view_transformation[0])

			i += 1
			
		#pdb.set_trace()
		yield ({'image_input': np.asarray(in_imgb), 'view_input':np.asarray(view_transformationb)}, 
			{'sequential_3': np.asarray(out_imgb)})
		

def generate_data_dictionary(dataPath='../data/train/'):
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
					split_index = int(len(d)*0.8)
					train_data_dict[i][e].extend(d[0:split_index])
					val_data_dict[i][e].extend(d[split_index:])

				
				i += 1

	
	#pdb.set_trace()
	return train_data_dict, val_data_dict

