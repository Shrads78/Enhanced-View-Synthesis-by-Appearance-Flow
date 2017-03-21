from scipy.misc import imsave
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def save_as_image(filepath, images):
	for i in range(0, len(images)):
		filename = filepath+str(i)+".png"
		imsave(filename, images[i])

def show_image(image):
	plt.imshow(np.squeeze(image))
	plt.show()



