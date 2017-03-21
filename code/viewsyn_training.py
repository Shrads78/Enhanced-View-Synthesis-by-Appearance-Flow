from keras.callbacks import *
import constants as const
import h5py

def train_network(network, f_generate_list, f_generate_data):

	# note that we are passing a list of Numpy arrays as training data
	# since the model has 2 inputs and 2 outputs
	#Callbacks
	hist = History()
	checkpoint = ModelCheckpoint('../model/weights.{epoch:02d}-{val_loss:.2f}.hdf5', monitor='val_loss', verbose=1, save_best_only=True, mode='auto', period=5)
	callbacks_list = [hist, checkpoint, tensor_callback]

	
	train_data_dict, val_data_dict = f_generate_list(dataPath = "../data/train/")

	b_size = const.batch_size
	s_epoch = const.samples_per_epoch

	history = network.fit_generator(f_generate_data(train_data_dict, b_size), samples_per_epoch=s_epoch, nb_epoch=100, verbose=1, callbacks=callbacks_list,
				validation_data=f_generate_data(val_data_dict, int(b_size*0.2)), nb_val_samples=int(s_epoch*0.2), class_weight=None, initial_epoch=0)

	print hist.history
	return hist

def load_autoenocoder_model_weights(model, weights_path):
	weights = h5py.File(weights_path)

	# Subset of full network which resembles to autoencoder
	layers = model.layers
	image_encoder_network = layers[2].layers
	image_decoder_network = layers[5].layers
	combined_network = np.concatenate((image_encoder_network, image_decoder_network))
	
	for layer in combined_network:
			layer_name = layer.name
			
			# Not present in autoencoder layer
			if 'bilinear_1' in layer_name: continue
			
			# Dimension changes for these two layers due to viewpoint transformation(dense_3) and Appearance flow(deconvolution2d_6)
			if 'dense_3' in layer_name or 'deconvolution2d_6' in layer_name:

				# Getting original set of weights from autoencoder network
				pretrained_w = weights['model_weights'][layer_name].values()[0]
				pretrained_b = weights['model_weights'][layer_name].values()[1]
				
				# Adding the padding weights due to extra channels.
				if 'dense_3' in layer_name:
					padding_w = layer.get_weights()[0][-256:,]
					new_weight_matrix = [np.concatenate((pretrained_w.value, padding_w)), pretrained_b]
				else:
					padding_w = layer.get_weights()[0][:,:,:,-2:]
					padding_b = layer.get_weights()[1][-2:]
					new_weight_matrix = [np.concatenate((pretrained_w.value, padding_w), axis = 3), np.concatenate((pretrained_b, padding_b))]

				#Setting the new weights
				layer.set_weights(np.array(new_weight_matrix))

			# Set of weights for other layers. Dimension doesn't changes.
			else:
				layer.set_weights(weights['model_weights'][layer_name].values())
