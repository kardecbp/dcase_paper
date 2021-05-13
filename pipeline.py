import numpy as np
import tensorflow as tf
import random
import os
from keras import backend as K
tf.set_random_seed(5)
np.random.seed(5)
random.seed(5)
os.environ['PYTHONHASHSEED'] = '0'
import pandas as pd
import librosa
from metrics import mapk
from keras.callbacks import EarlyStopping, LearningRateScheduler, ModelCheckpoint, ReduceLROnPlateau
from keras.utils import to_categorical
from augmentation import get_random_eraser, MixupGenerator
from sklearn.model_selection import StratifiedKFold
from keras.preprocessing.image import ImageDataGenerator
from models import mixed_cnn, mini_domain, mini_domain2, domain_model, pretrained_model
import pickle
import psutil
import gc
import cv2


dir_path = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(dir_path, 'data')
models_path = os.path.join(dir_path, 'models')
preds_path = os.path.join(dir_path, 'predictions')
train_path = os.path.join(data_dir, 'audio_train')
test_path = os.path.join(data_dir, 'audio_test')


class Config(object):
	# configuration class, all hyperparameters should be specified here
	def __init__(self, sampling_rate=44100, audio_duration=4, n_classes=41,
				 transform='cqt', learning_rate=1e-3, 
				 max_epochs=250, n_coef=96, batch_size=64, reduce_patience=7, reduce_factor=0.5, 
				 min_lr=0.00003125, es_patience=20, shift_duration=3,
				 random_state=5, mixup=True, alpha=0.2, width_shift_range=0.1,
				 zoom_range=0.1, num_filters=16):
		self.sampling_rate = sampling_rate
		self.audio_duration = audio_duration
		self.n_classes = n_classes
		if transform in ['cqt', 'melspec']:
			self.transform = transform
		else:
			self.transform = None
		self.n_coef = n_coef
		self.learning_rate = learning_rate
		self.max_epochs = max_epochs
		self.batch_size = batch_size
		self.reduce_patience = reduce_patience
		self.reduce_factor = reduce_factor
		self.min_lr = min_lr
		self.es_patience = es_patience
		self.random_state = random_state
		self.mixup = mixup
		self.alpha = alpha
		self.width_shift_range = width_shift_range
		self.zoom_range = zoom_range
		self.shift_duration = shift_duration
		self.num_filters = num_filters

		self.audio_length = self.sampling_rate * self.audio_duration
		self.step_length = self.sampling_rate * self.shift_duration

		if transform:
			self.dim = (self.n_coef, 1 + int(np.floor(self.audio_length/512)), 1)

		self.spec_len = 1 + int(np.floor(self.audio_length/512))
		self.step = int(np.floor(self.step_length/512))


def load_data(df, config, data_dir):
	# load and process all audio files, pad those which are shorter than specified 
	all_data = []
	counter = 0
	for i, fname in enumerate(df.index):
		file_path = os.path.join(data_dir, fname)
		data, _ = librosa.core.load(file_path, sr=config.sampling_rate, res_type="kaiser_fast")
		if config.audio_length > len(data):
			diff = config.audio_length - len(data)
			to_pad = diff / 2
			if not (diff % 2):
				to_pad_1, to_pad_2 = to_pad, to_pad
			else:
				to_pad_1, to_pad_2 = math.floor(to_pad), math.ceil(to_pad)
			data = np.pad(data, (int(to_pad_1), int(to_pad_2)), 'constant') 

		if config.transform == 'cqt':
			data = librosa.core.cqt(data, sr=config.sampling_rate, n_bins=config.n_coef).astype(np.complex64)
		elif config.transform == 'melspec':
			data = librosa.feature.melspectrogram(data, sr=config.sampling_rate, n_mels=config.n_coef).astype(np.complex64)
		all_data.append(data)
		
	return all_data


def precompute_dataset_size(X, config):
	# precomputing size of the dataset to save memory by creating empty numpy array
	counter = 0
	for i in range(len(X)):
		if X[i].shape[1] == config.spec_len:
			counter += 1
		else:
			start =  0
			while True:
				if (start+config.spec_len) <  X[i].shape[1]:
					start += config.step
					counter += 1
				else:
					break
	shape = (counter, config.n_coef, config.spec_len)

	return shape


def create_dataset(X, y, config, mode='train', pretrained=False):
	# spliting longer files and creating dataset, original ids are saved in true_ids list
	true_ids, y_fin = [], []
	shape = precompute_dataset_size(X, config)
	if pretrained:
		shape = (shape[0], 160, 160)
	X_fin, ex_counter = np.empty(shape, dtype=np.float32), 0
	for i in range(len(X)):
		if X[i].shape[1] == config.spec_len:
			example = librosa.amplitude_to_db(X[i], ref=np.max).astype(np.float32)
			if pretrained:
				example = cv2.resize(example, (160, 160))
			X_fin[ex_counter] = example
			true_ids.append(i)
			ex_counter += 1
			
			if mode == 'train':
				y_fin.append(y[i])
		else:
			start =  0
			while True:
				if (start+config.spec_len) <  X[i].shape[1]:
					example = X[i][:, start:(start+config.spec_len)]
					start += config.step
					example = librosa.amplitude_to_db(example, ref=np.max).astype(np.float32)
					if pretrained:
						example = cv2.resize(example, (160, 160))
					X_fin[ex_counter] = example
					true_ids.append(i)
					ex_counter += 1
					
					if mode == 'train':
						 y_fin.append(y[i])	
				else:
					break

	y_fin =  np.array(y_fin)
	X_fin = np.expand_dims(X_fin, axis=-1)
	
	return X_fin, y_fin, true_ids


def train_val_split(train):	
	# creating unique train-validation split, which is used for all models
	# validation data contains only manually varified examples, label distribution of train and validation data is the same
	train_mv, train_nmv = train[train['manually_verified'] == 1], train[train['manually_verified'] == 0]
	train_nmv_df = pd.DataFrame({'class': train_nmv['label'].value_counts().index, 'count': train_nmv['label'].value_counts().values})
	train_mv_df = pd.DataFrame({'class': train_mv['label'].value_counts().index, 'count': train_mv['label'].value_counts().values})
	total_counts = pd.merge(train_mv_df, train_nmv_df, on='class')
	new_names = {'count_x': 'mv_count', 'count_y': 'nmv_count'}
	total_counts.rename(columns=new_names, inplace=True)
	total_counts['val_frac'] = round((total_counts['mv_count'] + total_counts['nmv_count']) * 0.1)
	LABELS = list(train.label.unique())
	train_inds = list(train[train['manually_verified'] == 0].index.values)
	val_inds = []
	for label in LABELS:
		tmp_df = train[(train['label'] == label) & (train['manually_verified'] == 1)]
		num_val = total_counts[total_counts['class'] == label]['val_frac'].values[0]
		indices = tmp_df.index.values
		random.shuffle(indices)
		val_inds_tmp, train_inds_tmp = indices[:int(num_val)], indices[int(num_val):]
		train_inds.extend(train_inds_tmp), val_inds.extend(val_inds_tmp)
	return train_inds, val_inds


def split_data(X_tmp, y_tmp, true_ids, train_split, val_split):
	# generate train and validation datasets
	X_train, y_train, X_val, y_val, ids_train, ids_val = [], [], [], [], [], []

	for i in range(len(true_ids)):
		if true_ids[i] in train_split:
			X_train.append(X_tmp[i])
			y_train.append(y_tmp[i])
			ids_train.append(true_ids[i])
		elif true_ids[i] in val_split:
			X_val.append(X_tmp[i])
			y_val.append(y_tmp[i])
			ids_val.append(true_ids[i])

	return np.array(X_train), np.array(y_train), np.array(X_val), np.array(y_val), ids_train, ids_val


def preprocess(config, pretrained):
	# unified prepocessing
	train = pd.read_csv(os.path.join(data_dir, 'train.csv'))
	train.set_index("fname", inplace=True)
	label_ids = {label: i for i, label in enumerate(list(train.label.unique()))}
	train["label_ids"] = train['label'].apply(lambda x: label_ids[x])
	y = to_categorical(train["label_ids"])
	X = load_data(train, config, train_path)
	train.reset_index(inplace=True)
	train_split, val_split = train_val_split(train)
	X, y, true_ids = create_dataset(X, y, config, pretrained=pretrained)
	X_train, y_train, X_val, y_val, ids_train, ids_val = split_data(X, y, true_ids, train_split, val_split)

	return X_train, y_train, X_val, y_val, label_ids, ids_train, ids_val


def evaluate(preds, ids):
	# generating sample prediction by averaging all predictions with the same id
	real_preds = {}
	for i in range(len(ids)):
		if ids[i] not in real_preds:
			real_preds[ids[i]] = preds[i]
			real_preds[ids[i]] = np.expand_dims(real_preds[ids[i]], axis=-1)
		else:
			new_pred = np.expand_dims(preds[i], axis=-1)
			real_preds[ids[i]] = np.concatenate((real_preds[ids[i]], new_pred), axis=1)
	
	mean_preds = {key: np.mean(real_preds[key], axis=1) for key in real_preds}
	predictions = []
	for key in sorted(mean_preds.keys()):
		predictions.append(mean_preds[key])
	
	return predictions


def train(model_name, config, model_id=1):
	# training model
	model_dict = {1: mixed_cnn(config), 2: domain_model(config),
				  3: mini_domain(config), 4: mini_domain2(config),
				  5: pretrained_model(config, 'inception'), 6: pretrained_model(config, 'mobile')}

	pretrained = True if model_id in [5, 6] else False
	X_train, y_train, X_val, y_val, label_ids, ids_train, ids_val = preprocess(config, pretrained)

	if model_id == 2:
		# reshaping data for the domain model
		X_train = np.transpose(X_train, (0, 2, 1, 3))
		X_val = np.transpose(X_val, (0, 2, 1, 3))

	mean = np.mean(X_train, axis=0)
	std = np.std(X_train, axis=0)
	X_train = (X_train - mean) / std
	X_val = (X_val - mean) / std

	if model_id == 5 or model_id == 6:
		# adding channels for pretrained models
		mean_exp = np.repeat(mean[np.newaxis, :, :, :], X_train.shape[0], axis=0)
		mean_exp_val = np.repeat(mean[np.newaxis, :, :, :], X_val.shape[0], axis=0)
		X_train = np.concatenate([X_train, mean_exp, mean_exp], axis=-1)
		X_val = np.concatenate([X_val, mean_exp_val, mean_exp_val], axis=-1)
	
	model_save_dir = os.path.join(models_path, 'final')
	preds_save_dir = os.path.join(preds_path, 'final')
	
	gen = ImageDataGenerator(width_shift_range=config.width_shift_range, 
							 zoom_range=config.zoom_range, 
							 preprocessing_function=get_random_eraser(v_l=np.min(X_train), v_h=np.max(X_train)))
	if not os.path.exists(model_save_dir):
		os.mkdir(model_save_dir)
	if not os.path.exists(preds_save_dir):
		os.mkdir(preds_save_dir)

	model = model_dict[model_id]
	model_save_path = os.path.join(model_save_dir, model_name + '.h5')
	
	checkpoint = ModelCheckpoint(model_save_path, monitor='val_loss', verbose=1, save_best_only=True)
	early = EarlyStopping(monitor="val_loss", mode="min", patience=config.es_patience)
	reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=config.reduce_factor, 
								  patience=config.reduce_patience, verbose=1, 
								  min_lr=config.min_lr)

	callbacks_list = [checkpoint, early, reduce_lr]
	generator = MixupGenerator(X_train, y_train, alpha=config.alpha, batch_size=config.batch_size, datagen=gen)
	
	history = model.fit_generator(generator(), 
								  steps_per_epoch=X_train.shape[0] / config.batch_size,
								  callbacks=callbacks_list,
								  epochs=config.max_epochs,
								  # epochs = 3,
								  validation_data=(X_val, y_val))
	
	K.clear_session()
	tf.reset_default_graph()


def predict(model_name, config, model_id=1):
	model_dict = {1: mixed_cnn(config), 2: domain_model(config),
				  3: mini_domain(config), 4: mini_domain2(config),
				  5: pretrained_model(config, 'inception'), 6: pretrained_model(config, 'mobile')}

	pretrained = True if model_id in [5, 6] else False
	X_train, y_train, X_val, y_val, label_ids, ids_train, ids_val = preprocess(config, pretrained)

	if model_id == 2:
		X_train = np.transpose(X_train, (0, 2, 1, 3))
		X_val = np.transpose(X_val, (0, 2, 1, 3))

	
	mean = np.mean(X_train, axis=0)
	std = np.std(X_train, axis=0)
	X_train = (X_train - mean) / std
	X_val = (X_val - mean) / std

	if model_id == 5 or model_id == 6:
		mean_exp = np.repeat(mean[np.newaxis, :, :, :], X_train.shape[0], axis=0)
		mean_exp_val = np.repeat(mean[np.newaxis, :, :, :], X_val.shape[0], axis=0)
		X_train = np.concatenate([X_train, mean_exp, mean_exp], axis=-1)
		X_val = np.concatenate([X_val, mean_exp_val, mean_exp_val], axis=-1)
	
	model_save_dir = os.path.join(models_path, 'final')
	preds_save_dir = os.path.join(preds_path, 'final')
	
	model = model_dict[model_id]
	model_save_path = os.path.join(model_save_dir, model_name + '.h5')
	model.load_weights(model_save_path)
	
	preds_train = model.predict(X_train, batch_size=config.batch_size, verbose=1)
	preds_val = model.predict(X_val, batch_size=config.batch_size, verbose=1)

	preds_train_avrg = evaluate(preds_train, ids_train)
	preds_val_avrg = evaluate(preds_val, ids_val)
	X_train, X_val = None, None
	
	test = pd.read_csv(os.path.join(data_dir, "sample_submission.csv"))
	test.set_index("fname", inplace=True)

	X_test = load_data(test, config, test_path)
	
	X_test, _, test_ids = create_dataset(X_test, y=None, config=config, mode='test', pretrained=pretrained)
	if model_id == 2:
		X_test = np.transpose(X_test, (0, 2, 1, 3))
	X_test = (X_test - mean) / std

	if model_id == 5 or model_id == 6:
		mean_exp = np.repeat(mean[np.newaxis, :, :, :], X_test.shape[0], axis=0)
		X_test = np.concatenate([X_test, mean_exp, mean_exp], axis=-1)
	
	preds_test = model.predict(X_test, batch_size=config.batch_size, verbose=1)
	preds_test_avrg = evaluate(preds_test, test_ids)
	
	pred_train_name = 'preds_train_' + model_name + '.pickle'
	pred_val_name = 'preds_val_' + model_name + '.pickle'
	pred_test_name = 'preds_test_' + model_name + '.pickle'

	pickle.dump(preds_train_avrg, open(os.path.join(preds_save_dir, pred_train_name), 'wb'))
	pickle.dump(preds_val_avrg, open(os.path.join(preds_save_dir, pred_val_name), 'wb'))
	pickle.dump(preds_test_avrg, open(os.path.join(preds_save_dir, pred_test_name), 'wb'))

	K.clear_session()
	tf.reset_default_graph()


if __name__ == '__main__':
	configs = [Config(transform='cqt', batch_size=32, n_coef=96, audio_duration=4, shift_duration=3),
			   Config(transform='cqt', batch_size=32, n_coef=96, audio_duration=4, shift_duration=3)]
	models = ['model2_cqt_4s', 'model5_cqt_4s']
	model_ids = [2, 5]
	for i in range(len(models)):
		train(models[i], configs[i], model_ids[i])
		predict(models[i], configs[i], model_ids[i])
	
	