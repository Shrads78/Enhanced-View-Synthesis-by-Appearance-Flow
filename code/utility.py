from scipy.misc import imsave
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import Image
import os

def save_as_image(images):
	for i in range(0, len(images)):
		filename = filepath+str(i)+".png"
		imsave(filename, images[i])

def show_image(image):
	plt.imshow(np.squeeze(image))
	plt.show()

def img_mask_gen(imgpath):
	im = Image.open(imgpath).convert('L').point(lambda x: 0 if x<=0 or x>=250 else 255,'1')
	return im	

def generate_autoencoder_data_from_list(dataArr):
	while 1:
		for fp in dataArr:	
			currImgPath = fp[0]
			img = Image.open(currImgPath)
			#msk = imgMaskGen(currImgPath)
			yield ({'image': img}, {'output': img})

def generate_data_array_for_autoencoder(dataPath='../data/chairs/'):
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
								dataArr.append([readLoc])

	dataArr = np.asarray(dataArr)
	np.random.shuffle(dataArr)
	return dataArr

