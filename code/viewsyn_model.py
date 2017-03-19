import keras
from keras.layers.convolutional import *
from keras.layers import *
from keras.models import Sequential, Model
from keras.optimizers import *
from keras.callbacks import *
import utility as util
import pdb
import math

def get_optimizer(name = 'adagrad', l_rate = 0.0001, dec = 0.0, b_1 = 0.9, b_2 = 0.999, mom = 0.5, rh = 0.9):
	eps = 1e-8
	
	adam = Adam(lr = l_rate, beta_1 = b_1, beta_2 = b_2, epsilon = eps, decay = dec)
	sgd = SGD(lr = l_rate, momentum = mom, decay = dec, nesterov = True)
	rmsp = RMSprop(lr = l_rate, rho = rh, epsilon = eps, decay = dec)
	adagrad = Adagrad(lr = l_rate, epsilon = eps, decay = dec)
	
	optimizers = {'adam': adam, 'sgd':sgd, 'rmsp': rmsp, 'adagrad': adagrad}

	return optimizers[name]

def build_viewpoint_encoder():
	#define network architecture for viewpoint transformation encoder
	model =  Sequential()

	#2 fully connected layers
	model.add(Dense(128, input_dim=19, activation='relu'))
	model.add(Dense(256, activation='relu'))

	return model

def build_autoencoder():
	image_input = Input(shape=(224, 224, 3,), name='image_input')
	view_input = Input(shape=(19,), name='view_input')

	#define network architecture for encoder
	image_encoder = Sequential()

	#6 convoltuion layers
	image_encoder.add(Convolution2D(16, 3, 3, border_mode='same', subsample = (2,2), activation = 'relu',
			input_shape=(224, 224, 3)))
	image_encoder.add(Convolution2D(32, 3, 3, border_mode='same', subsample = (2,2), activation = 'relu'))
	image_encoder.add(Convolution2D(64, 3, 3, border_mode='same', subsample = (2,2), activation = 'relu'))
	image_encoder.add(Convolution2D(128, 3, 3, border_mode='same', subsample = (2,2), activation = 'relu'))
	image_encoder.add(Convolution2D(256, 3, 3, border_mode='same', subsample = (2,2), activation = 'relu'))
	image_encoder.add(Convolution2D(512, 3, 3, border_mode='same', subsample = (2,2), activation = 'relu'))

	#Flatten 
	image_encoder.add(Flatten())

	#2 fully connected layers
	image_encoder.add(Dense(4096, activation='relu'))
	image_encoder.add(Dropout(p=0))
	image_encoder.add(Dense(4096, activation='relu'))
	image_encoder.add(Dropout(p=0))

	#view encoder 
	view_encoder = build_viewpoint_encoder()

	image_output = image_encoder(image_input)
	view_output = view_encoder(view_input)

	image_view_out = merge([image_output, view_output], mode='concat', concat_axis=1)
	
	#define network architecture for decoder
	
	#2 fully connected layers
	model = Sequential()
	model.add(Dense(4096, input_dim=4352, activation='relu'))
	#model.add(Dense(4096, activation='relu'))
	
	#reshape to 2D
	model.add(Reshape((8, 8, 64)))
	
	#5 upconv layers
	model.add(Deconvolution2D(256, 3, 3, (None, 15, 15,256), border_mode='same', subsample = (2,2), activation = 'relu'))
	model.add(Deconvolution2D(128, 3, 3, (None, 29, 29, 128), border_mode='same', subsample = (2,2), activation = 'relu'))
	model.add(Deconvolution2D(64, 3, 3, (None, 57, 57, 64), border_mode='same', subsample = (2,2), activation = 'relu'))
	model.add(Deconvolution2D(32, 3, 3, (None, 113, 113, 32),border_mode='same', subsample = (2,2), activation = 'relu'))
	model.add(Deconvolution2D(16, 3, 3, (None, 225, 225, 16), border_mode='same', subsample = (2,2), activation = 'relu'))
	
	#output layer, add 3 RGB channels for reconstructed output view
	model.add(Deconvolution2D(3, 3, 3, (None, 225, 225, 3), border_mode='same', subsample = (1,1), activation = 'relu'))

	#add a resize layer to resize (225, 225) to (224, 224)
	model.add(Reshape((225*225,3)))
	model.add(Lambda(lambda x: x[:,:50176,])) # throw away some
	model.add(Reshape((224,224,3)))

	main_output = model(image_view_out)

	encoder_decoder = Model(input=[image_input, view_input], output=main_output)
	#compile model
	opt = get_optimizer('adam')
	encoder_decoder.compile(optimizer=opt, metrics=['accuracy'], loss='mean_squared_error')

	print encoder_decoder.summary()
	return encoder_decoder


def train_autoencoder(autoencoder):
	
	#Callbacks
	hist = History()
	checkpoint = ModelCheckpoint('../model/weights.{epoch:02d}-{val_loss:.2f}.hdf5', monitor='val_loss', verbose=1, save_best_only=True, mode='auto', period=5)
	callbacks_list = [hist, checkpoint]

	val_dict, train_dict = util.generate_data_dictionary(dataPath='./../data/train/')	
	
	#for d in util.generate_autoencoder_data_from_list(dataArr):
	#	pdb.set_trace()

	history = autoencoder.fit_generator(util.generate_data_from_list(train_dict, 128), samples_per_epoch=280000, nb_epoch=100, verbose=1, callbacks=callbacks_list,
		 validation_data=util.generate_data_from_list(val_dict, 128*0.2), nb_val_samples=280000*0.2, class_weight=None, initial_epoch=0)	

	print hist.history
	return hist

def test_autoencoder(autoencoder, test_data):
	
	test_image = test_data[1:2,]
	output = autoencoder.predict(test_image)
	
	util.show_image(output)

	
