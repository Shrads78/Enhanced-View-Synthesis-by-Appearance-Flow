import keras
from keras.layers.convolutional import *
from keras.layers import *
from keras.models import Sequential, Model
from keras.optimizers import *
from bilinear_layer import Bilinear
import utility as util
import pdb
from keras import backend as K
from masked_loss import *

def get_optimizer(name = 'adagrad', l_rate = 0.0001, dec = 0.0, b_1 = 0.9, b_2 = 0.999, mom = 0.5, rh = 0.9):
	eps = 1e-8
	
	adam = Adam(lr = l_rate, beta_1 = b_1, beta_2 = b_2, epsilon = eps, decay = dec)
	sgd = SGD(lr = l_rate, momentum = mom, decay = dec, nesterov = True)
	rmsp = RMSprop(lr = l_rate, rho = rh, epsilon = eps, decay = dec)
	adagrad = Adagrad(lr = l_rate, epsilon = eps, decay = dec)
	
	optimizers = {'adam': adam, 'sgd':sgd, 'rmsp': rmsp, 'adagrad': adagrad}

	return optimizers[name]

def build_image_encoder():
	#define network architecture for image encoder
	model = Sequential()

	#6 convoltuion layers
	model.add(Convolution2D(16, 3, 3, border_mode='same', subsample = (2,2), activation = 'relu',
			input_shape=(224, 224, 3)))
	model.add(Convolution2D(32, 3, 3, border_mode='same', subsample = (2,2), activation = 'relu'))
	model.add(Convolution2D(64, 3, 3, border_mode='same', subsample = (2,2), activation = 'relu'))
	model.add(Convolution2D(128, 3, 3, border_mode='same', subsample = (2,2), activation = 'relu'))
	model.add(Convolution2D(256, 3, 3, border_mode='same', subsample = (2,2), activation = 'relu'))
	model.add(Convolution2D(512, 3, 3, border_mode='same', subsample = (2,2), activation = 'relu'))

	#Flatten 
	model.add(Flatten())

	#2 fully connected layers
	model.add(Dense(4096, activation='relu'))
	model.add(Dropout(p=0))
	model.add(Dense(4096, activation='relu'))
	model.add(Dropout(p=0))

	return model

def build_viewpoint_encoder():
	#define network architecture for viewpoint transformation encoder
	model =  Sequential()

	#2 fully connected layers
	model.add(Dense(128, input_dim=19, activation='relu'))
	model.add(Dense(256, activation='relu'))

	return model
	

def build_common_decoder(input_dim=4352):

	#define network architecture for decoder
	model =  Sequential()

	#1 fully connected layers
	model.add(Dense(4096, input_dim=input_dim, activation='relu')) #4096+256
	#This layer is extra, keeping it here for backward compatibility
	#model.add(Dense(4096, activation='relu')) 
	
	#reshape to 2D
	model.add(Reshape((8, 8, 64)))
	
	#5 upconv layers
	model.add(Deconvolution2D(256, 3, 3, (None, 15, 15,256), border_mode='same', subsample = (2,2), activation = 'relu'))
	model.add(Deconvolution2D(128, 3, 3, (None, 29, 29, 128), border_mode='same', subsample = (2,2), activation = 'relu'))
	model.add(Deconvolution2D(64, 3, 3, (None, 57, 57, 64), border_mode='same', subsample = (2,2), activation = 'relu'))
	model.add(Deconvolution2D(32, 3, 3, (None, 113, 113, 32),border_mode='same', subsample = (2,2), activation = 'relu'))
	model.add(Deconvolution2D(16, 3, 3, (None, 225, 225, 16), border_mode='same', subsample = (2,2), activation = 'relu'))
	
	return model

def output_layer_decoder(model, n_channel):
	
	#output layer, add 3 RGB channels for reconstructed output view
	model.add(Deconvolution2D(n_channel, 3, 3, (None, 225, 225, n_channel), border_mode='same', subsample = (1,1)))#, activation = 'relu'))

	#add a resize layer to resize (225, 225) to (224, 224)
	model.add(Reshape((225*225,n_channel)))
	model.add(Lambda(lambda x: x[:,:50176,])) # throw away some
	model.add(Reshape((224,224,n_channel)))

	
	#print model.summary()
	return model

def build_five_channel_network():
	image_input = Input(shape=(224, 224, 3,), name='image_input')
	view_input = Input(shape=(19,), name='view_input')

	image_encoder = build_image_encoder()

	decoder = build_common_decoder()
	decoder = output_layer_decoder(decoder, 5) #5 channels in output layer

	#add bilinear layer
	decoder.add(Bilinear())

	view_encoder = build_viewpoint_encoder()

	mask_decoder = build_common_decoder()
	mask_decoder = output_layer_decoder(mask_decoder, 1)

	
	image_output = image_encoder(image_input)
	view_output = view_encoder(view_input)

	image_view_out = merge([image_output, view_output], mode='concat', concat_axis=1)

	main_output = decoder(image_view_out)
	mask_output = mask_decoder(image_view_out)

	encoder_decoder = Model(input=[image_input, view_input], output=[main_output, mask_output])


	opt = get_optimizer('adam')
	encoder_decoder.compile(optimizer=opt, metrics=['accuracy'],
				loss={'sequential_2': maskedl1loss, 'sequential_4': 'binary_crossentropy'},
              			loss_weights={'sequential_2': 1.0, 'sequential_4': 0.1})

	print encoder_decoder.summary()

	return encoder_decoder

def build_replication_network():
	image_input = Input(shape=(224, 224, 3,), name='image_input')
	view_input = Input(shape=(19,), name='view_input')

	image_encoder = build_image_encoder()
	view_encoder = build_viewpoint_encoder()

	decoder = build_common_decoder()
	decoder = output_layer_decoder(decoder, 2) #replication
	
	mask_decoder = build_common_decoder()
	mask_decoder = output_layer_decoder(mask_decoder, 1)

	
	image_output = image_encoder(image_input)
	view_output = view_encoder(view_input)

	image_view_out = merge([image_output, view_output], mode='concat', concat_axis=1)

	main_output = decoder(image_view_out)
	bilinear_in = merge([image_input, main_output], mode='concat', concat_axis=3)
	bilinear_out = Bilinear()(bilinear_in)
	
	mask_output = mask_decoder(image_view_out)

	
	encoder_decoder = Model(input=[image_input, view_input], output=[bilinear_out, mask_output])

	opt = get_optimizer('adam')
	encoder_decoder.compile(optimizer=opt, metrics=['accuracy'],
				loss={'bilinear_1': maskedl1loss, 'sequential_4': 'binary_crossentropy'},
              			loss_weights={'bilinear_1': 1.0, 'sequential_4': 0.1})

	print encoder_decoder.summary()

	return encoder_decoder

def build_transformed_autoencoder():
	image_input = Input(shape=(224, 224, 3,), name='image_input')
	view_input = Input(shape=(19,), name='view_input')

	image_encoder = build_image_encoder()
	view_encoder = build_viewpoint_encoder()

	decoder = build_common_decoder()
	decoder = output_layer_decoder(decoder, 3) #transoformed autoencoder
	
	image_output = image_encoder(image_input)
	view_output = view_encoder(view_input)

	image_view_out = merge([image_output, view_output], mode='concat', concat_axis=1)
	
	main_output = decoder(image_view_out)

	transformed_autoencoder = Model(input=[image_input, view_input], output=main_output)
	#compile model
	opt = get_optimizer('adam')
	transformed_autoencoder.compile(optimizer=opt, metrics=['accuracy'], loss=maskedl1loss)

	print transformed_autoencoder.summary()
	return transformed_autoencoder

def build_transformed_autoencoder_maskstream():
	image_input = Input(shape=(224, 224, 3,), name='image_input')
	view_input = Input(shape=(19,), name='view_input')

	image_encoder = build_image_encoder()
	view_encoder = build_viewpoint_encoder()

	decoder = build_common_decoder()
	decoder = output_layer_decoder(decoder, 3) #transoformed autoencoder
	
	mask_decoder = build_common_decoder()
	mask_decoder = output_layer_decoder(mask_decoder, 1)

	image_output = image_encoder(image_input)
	view_output = view_encoder(view_input)

	image_view_out = merge([image_output, view_output], mode='concat', concat_axis=1)
	
	main_output = decoder(image_view_out)

	mask_output = mask_decoder(image_view_out)

	transformed_autoencoder = Model(input=[image_input, view_input], output=[main_output, mask_output])
	#pdb.set_trace()
	#compile model
	opt = get_optimizer('adam')
	transformed_autoencoder.compile(optimizer=opt, metrics=['accuracy'],
				loss={'sequential_3': maskedl1loss, 'sequential_4': 'binary_crossentropy'},
              			loss_weights={'sequential_3': 1.0, 'sequential_4': 0.1})

	print transformed_autoencoder.summary()
	return transformed_autoencoder

def build_autoencoder():
	autoencoder = Sequential()

	#6 convolution layers
	autoencoder.add(Convolution2D(16, 3, 3, border_mode='same', subsample = (2,2), activation = 'relu',
			input_shape=(224, 224, 3)))
	autoencoder.add(Convolution2D(32, 3, 3, border_mode='same', subsample = (2,2), activation = 'relu'))
	autoencoder.add(Convolution2D(64, 3, 3, border_mode='same', subsample = (2,2), activation = 'relu'))
	autoencoder.add(Convolution2D(128, 3, 3, border_mode='same', subsample = (2,2), activation = 'relu'))
	autoencoder.add(Convolution2D(256, 3, 3, border_mode='same', subsample = (2,2), activation = 'relu'))
	autoencoder.add(Convolution2D(512, 3, 3, border_mode='same', subsample = (2,2), activation = 'relu'))

	#Flatten 
	autoencoder.add(Flatten())

	#2 fully connected layers
	autoencoder.add(Dense(4096, activation='relu'))
	autoencoder.add(Dropout(p=0))
	autoencoder.add(Dense(4096, activation='relu'))
	autoencoder.add(Dropout(p=0))

	
	#define network architecture for decoder
	
	#2 fully connected layers
	autoencoder.add(Dense(4096, activation='relu'))
	autoencoder.add(Dense(4096, activation='relu'))
	
	#reshape to 2D
	autoencoder.add(Reshape((8, 8, 64)))
	
	#5 upconv layers
	autoencoder.add(Deconvolution2D(256, 3, 3, (None, 15, 15,256), border_mode='same', subsample = (2,2), activation = 'relu'))
	autoencoder.add(Deconvolution2D(128, 3, 3, (None, 29, 29, 128), border_mode='same', subsample = (2,2), activation = 'relu'))
	autoencoder.add(Deconvolution2D(64, 3, 3, (None, 57, 57, 64), border_mode='same', subsample = (2,2), activation = 'relu'))
	autoencoder.add(Deconvolution2D(32, 3, 3, (None, 113, 113, 32),border_mode='same', subsample = (2,2), activation = 'relu'))
	autoencoder.add(Deconvolution2D(16, 3, 3, (None, 225, 225, 16), border_mode='same', subsample = (2,2), activation = 'relu'))
	
	#output layer, add 3 RGB channels for reconstructed output view
	autoencoder.add(Deconvolution2D(3, 3, 3, (None, 225, 225, 3), border_mode='same', subsample = (1,1), activation = 'relu'))

	#add a resize layer to resize (225, 225) to (224, 224)
	autoencoder.add(Reshape((225*225,3)))
	autoencoder.add(Lambda(lambda x: x[:,:50176,])) # throw away some
	autoencoder.add(Reshape((224,224,3)))

	#compile autoencoder
	opt = get_optimizer('adam')
	autoencoder.compile(optimizer=opt, metrics=['accuracy'], loss=maskedl1loss)


	print autoencoder.summary()
	return autoencoder
	