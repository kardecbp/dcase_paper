from keras import losses, models, optimizers
import tensorflow as tf
from keras.activations import relu, softmax
from keras.layers import (Conv1D, Dense, Dropout, GlobalAveragePooling1D, 
                          GlobalMaxPooling2D, GlobalMaxPooling1D, Input, MaxPooling2D, concatenate, Conv2D, 
                          MaxPooling2D, ELU, Reshape, BatchNormalization, Lambda,
                          AveragePooling2D, concatenate, Flatten)
from keras.utils import Sequence, to_categorical
from keras.models import Sequential
from keras.layers.merge import Concatenate, add, Add
from keras import initializers
from keras import backend as K
from keras.applications import VGG16, ResNet50, InceptionV3, InceptionResNetV2, MobileNet
from ann_visualizer.visualize import ann_viz

# from keras.applications.densenet import DenseNet121



def pretrained_model(config, model='inception'):
	if model == 'inception':
		pretrained = InceptionV3(weights='imagenet', include_top=False)
	elif model == 'mobile':
		pretrained = MobileNet(weights='imagenet', include_top=False, input_shape=(160, 160, 3))	
	inp = pretrained.input
	x = pretrained.output
	x = GlobalMaxPooling2D()(x)
	x = Dense(64, activation='relu')(x)
	x = Dropout(0.2)(x)
	x = Dense(64, activation='relu')(x)
	x = Dropout(0.2)(x)
	output = Dense(config.n_classes, activation='softmax')(x)
	model = models.Model(inputs=inp, outputs=output)
	# opt = optimizers.Adam(config.learning_rate)
	opt = optimizers.SGD(config.learning_rate)
	model.compile(optimizer=opt, loss=losses.categorical_crossentropy, metrics=['acc'])
	return model




def mixed_cnn(config):
	input_shape = (config.dim[0], config.dim[1], 1)

	model = Sequential()
	model.add(Conv2D(64, (7, 3),  input_shape=input_shape, strides=(1, 2), activation='relu', padding='same'))
	model.add(MaxPooling2D((4, 1), strides=(2, 1)))
	model.add(BatchNormalization())

	model.add(Conv2D(128, (7, 1),  strides=(1, 1), activation='relu', padding='same'))
	model.add(MaxPooling2D((4, 2), strides=(2, 2)))
	model.add(BatchNormalization())

	model.add(Conv2D(128, (5, 1),  strides=(1, 1), activation='relu', padding='valid'))
	model.add(BatchNormalization())
	model.add(Conv2D(128, (1, 5),  strides=(1, 1), activation='relu', padding='same'))
	model.add(BatchNormalization())

	model.add(GlobalMaxPooling2D())
	model.add(Dropout(0.25))
	
	model.add(BatchNormalization())
	model.add(Dense(64, activation='relu'))
	model.add(Dropout(0.25)) 
	model.add(Dense(64, activation='relu'))
	model.add(Dropout(0.25)) 
	model.add(Dense(config.n_classes, activation='softmax'))

	model.compile(loss='categorical_crossentropy',
	          optimizer=optimizers.Adam(lr=config.learning_rate),
	          metrics=['accuracy'])
	return model



def temporal_model(config):
	input_shape = (config.dim[0], config.dim[1], 1)

	inp = Input(input_shape, name='input')
	f1 = Conv2D(16, (3, 32), activation='relu', padding='same')(inp)
	m1 = MaxPooling2D((2, 2))(f1)
	b1 = BatchNormalization()(m1)
	print(K.int_shape(b1))

	f2 = Conv2D(16, (3, 64), activation='relu', padding='same')(inp)
	m2 = MaxPooling2D((2, 2))(f2)
	b2 = BatchNormalization()(m2)
	print(K.int_shape(b2))

	f3 = Conv2D(16, (3, 128), activation='relu', padding='same')(inp)
	m3 = MaxPooling2D((2, 2))(f3)
	b3 = BatchNormalization()(m3)
	print(K.int_shape(b3))

	f4 = Conv2D(16, (8, 3), activation='relu', padding='same')(inp)
	m4 = MaxPooling2D((2, 2))(f4)
	b4 = BatchNormalization()(m4)
	print(K.int_shape(b4))

	f5 = Conv2D(16, (16, 3), activation='relu', padding='same')(inp)
	m5 = MaxPooling2D((2, 2))(f5)
	b5 = BatchNormalization()(m5)
	print(K.int_shape(b5))

	f6 = Conv2D(16, (32, 3), activation='relu', padding='same')(inp)
	m6 = MaxPooling2D((2, 2))(f6)
	b6 = BatchNormalization()(m6)
	print(K.int_shape(b6))

	concat = Concatenate(axis=3)([b1, b2, b3, b4, b5, b6])
	
	fcon = Conv2D(64, (5, 5),  activation='relu', padding='valid')(concat)
	bcon = BatchNormalization()(fcon)
	print(K.int_shape(bcon))

	gmpool = GlobalMaxPooling2D()(bcon)

	drop = Dropout(0.25)(gmpool)
	dense1 = Dense(64, activation='relu')(drop)
	drop1 = Dropout(0.25)(dense1)
	dense2 = Dense(64, activation='relu')(drop1)
	drop2 = Dropout(0.25)(dense2)
	output = Dense(config.n_classes, activation='softmax')(drop2)

	model = models.Model(inputs=inp, outputs=output)
	opt = optimizers.Adam(config.learning_rate)
	model.compile(optimizer=opt, loss=losses.categorical_crossentropy, metrics=['acc'])

	return model


def mini_domain2(config):
	input_shape = (config.dim[0], config.dim[1], 1)

	inp = Input(input_shape, name='input')

	f11 = Conv2D(64, (8, 3), strides=(1, 2), activation='relu', padding='same')(inp)
	m11 = MaxPooling2D((4, 1), strides=(2, 1))(f11)
	b11 = BatchNormalization()(m11)
	print(K.int_shape(b11))

	f12 = Conv2D(128, (8, 1),  strides=(1, 1), activation='relu', padding='same')(b11)
	m12 = MaxPooling2D((4, 2), strides=(2, 2))(f12) 
	b12 = BatchNormalization()(m12)

	f21 = Conv2D(64, (16, 3), strides=(1, 2), activation='relu', padding='same')(inp)
	m21 = MaxPooling2D((4, 1), strides=(2, 1))(f21)
	b21 = BatchNormalization()(m21)

	f22 = Conv2D(128, (16, 1),  strides=(1, 1), activation='relu', padding='same')(b21)
	m22 = MaxPooling2D((4, 2), strides=(2, 2))(f22) 
	b22 = BatchNormalization()(m22)


	f31 = Conv2D(64, (32, 3), strides=(1, 2), activation='relu', padding='same')(inp)
	m31 = MaxPooling2D((4, 1), strides=(2, 1))(f31)
	b31 = BatchNormalization()(m31)

	f32 = Conv2D(128, (32, 1),  strides=(1, 1), activation='relu', padding='same')(b21)
	m32 = MaxPooling2D((4, 2), strides=(2, 2))(f32) 
	b32 = BatchNormalization()(m32)

	concat = Concatenate(axis=3)([b12, b22, b32])

	fcon = Conv2D(128, (5, 1),  strides=(1, 1), activation='relu', padding='valid')(concat)
	bcon = BatchNormalization()(fcon)
	fcon2 = Conv2D(128, (1, 5),  strides=(1, 1), activation='relu', padding='same')(bcon)
	bcon2 = BatchNormalization()(fcon2)

	gmpool = GlobalMaxPooling2D()(bcon2)

	drop = Dropout(0.25)(gmpool)
	dense1 = Dense(64, activation='relu')(drop)
	drop1 = Dropout(0.25)(dense1)
	dense2 = Dense(64, activation='relu')(drop1)
	drop2 = Dropout(0.25)(dense2)
	output = Dense(config.n_classes, activation='softmax')(drop2)

	model = models.Model(inputs=inp, outputs=output)
	opt = optimizers.Adam(config.learning_rate)
	model.compile(optimizer=opt, loss=losses.categorical_crossentropy, metrics=['acc'])

	return model	

def domain_model(config):
	input_shape = (config.dim[1], config.dim[0], 1)

	inp = Input(input_shape, name='input')

	# timbre features

	# 0.9 * input_shape[1]
	input_pad_7 = Lambda(lambda y: tf.pad(y, [[0, 0], [3, 3], [0, 0], [0, 0]], "CONSTANT"))(inp)
	input_pad_3 = Lambda(lambda y: tf.pad(y, [[0, 0], [1, 1], [0, 0], [0, 0]], "CONSTANT"))(inp)

	f1 = Conv2D(config.num_filters, (7, int(0.9 * input_shape[1])), padding='valid', activation='relu')(input_pad_7)
	b1 = BatchNormalization()(f1)
	p1 = MaxPooling2D(pool_size=(1, K.int_shape(b1)[2]), strides=(1, K.int_shape(b1)[2]))(b1)
	p1 = Lambda(lambda y: K.squeeze(y, axis=2))(p1)


	f2 = Conv2D(config.num_filters * 2, (3, int(0.9 * input_shape[1])), padding='valid', activation='relu')(input_pad_3)
	b2 = BatchNormalization()(f2)
	p2 = MaxPooling2D(pool_size=(1, K.int_shape(b2)[2]), strides=(1, K.int_shape(b2)[2]))(b2)
	p2 = Lambda(lambda y: K.squeeze(y, axis=2))(p2)


	f3 = Conv2D(config.num_filters * 4, (1, int(0.9 * input_shape[1])), padding='valid', activation='relu')(inp)
	b3 = BatchNormalization()(f3)
	p3 = MaxPooling2D(pool_size=(1, K.int_shape(b3)[2]), strides=(1, K.int_shape(b3)[2]))(b3)
	p3 = Lambda(lambda y: K.squeeze(y, axis=2))(p3)

	# 0.4 * input_shape[1]
	
	f4 = Conv2D(config.num_filters, (7, int(0.4 * input_shape[1])), padding='valid', activation='relu')(input_pad_7)
	b4 = BatchNormalization()(f4)
	p4 = MaxPooling2D(pool_size=(1, K.int_shape(b4)[2]), strides=(1, K.int_shape(b4)[2]))(b4)
	p4 = Lambda(lambda y: K.squeeze(y, axis=2))(p4)


	f5 = Conv2D(config.num_filters * 2, (3, int(0.4 * input_shape[1])), padding='valid', activation='relu')(input_pad_3)
	b5 = BatchNormalization()(f5)
	p5 = MaxPooling2D(pool_size=(1, K.int_shape(b5)[2]), strides=(1, K.int_shape(b5)[2]))(b5)
	p5 = Lambda(lambda y: K.squeeze(y, axis=2))(p5)


	f6 = Conv2D(config.num_filters * 4, (1, int(0.4 * input_shape[1])), padding='valid', activation='relu')(inp)
	b6 = BatchNormalization()(f6)
	p6 = MaxPooling2D(pool_size=(1, K.int_shape(b6)[2]), strides=(1, K.int_shape(b6)[2]))(b6)
	p6 = Lambda(lambda y: K.squeeze(y, axis=2))(p6)
	
	# temporal features

	avrg = AveragePooling2D(pool_size=(1, input_shape[1]), strides=(1, input_shape[1]))(inp)
	avrg = Lambda(lambda y: K.squeeze(y, axis=3))(avrg)

	temp1 = Conv1D(config.num_filters, 165, padding='same', activation='relu')(avrg)
	temp1 = BatchNormalization()(temp1)

	temp2 = Conv1D(2 * config.num_filters, 128, padding='same', activation='relu')(avrg)
	temp2 = BatchNormalization()(temp2)

	temp3 = Conv1D(4 * config.num_filters, 64, padding='same', activation='relu')(avrg)
	temp3 = BatchNormalization()(temp3)

	temp4 = Conv1D(8 * config.num_filters, 32, padding='same', activation='relu')(avrg)
	temp4 = BatchNormalization()(temp4)

	concat = Concatenate(axis=2)([p1, p2, p3, p4, p5, p6, temp1, temp2, temp3, temp4])
	front = Lambda(lambda y: K.expand_dims(y, axis=-1))(concat)
	
	conv1 = Conv2D(512, (7, K.int_shape(front)[2]), padding='valid', activation='relu', name='cnn1')(front)
	bconv1 = BatchNormalization()(conv1)
	
	# bconv1_t = Lambda(lambda y: K.permute_dimensions(y, (0, 1, 3, 2)))(bconv1)
	bconv1_pad = Lambda(lambda y: tf.pad(y, [[0, 0], [3, 3], [0, 0], [0, 0]], "CONSTANT"))(bconv1)
	
	conv2 = Conv2D(512, (7, K.int_shape(bconv1_pad)[2]), padding='valid', activation='relu', name='cnn2')(bconv1_pad)
	# conv2_t = Lambda(lambda y: K.permute_dimensions(y, (0, 1, 3, 2)))(conv2)
	bconv2 = BatchNormalization()(conv2)
	
	res = Add()([bconv2, bconv1])
	pool1 = MaxPooling2D(pool_size=[2, 1], strides=[2, 1])(res)
	pool1_pad = Lambda(lambda y: tf.pad(y, [[0, 0], [3, 3], [0, 0], [0, 0]], "CONSTANT"))(pool1)
	conv3 = Conv2D(512, (7, K.int_shape(pool1_pad)[2]), padding='valid', activation='relu', name='cnn3')(pool1_pad)
	# conv3_t = Lambda(lambda y: K.permute_dimensions(y, (0, 1, 3, 2)))(conv3)
	bconv3 = BatchNormalization()(conv3)
	res2 = Add()([bconv3, pool1])

	max_pool = Lambda(lambda y: K.max(y, axis=1))(res2)
	avrg_pool = Lambda(lambda y: K.mean(y, axis=1))(res2)
	
	pool = Concatenate(axis=2)([max_pool, avrg_pool])
	flat = Flatten()(pool)
	flat_drop = Dropout(0.4)(flat)

	dense = Dense(300, activation='relu')(flat_drop)
	dense_drop = Dropout(0.4)(dense)

	output = Dense(config.n_classes, activation='softmax')(dense_drop)
	# print(K.int_shape(output))
	model = models.Model(inputs=inp, outputs=output)
	opt = optimizers.Adam(config.learning_rate)
	model.compile(optimizer=opt, loss=losses.categorical_crossentropy, metrics=['acc'])
	# print(model.summary())

	return model


def mini_domain(config):
	input_shape = (config.dim[0], config.dim[1], 1)
	inp = Input(input_shape, name='input')

	f1 = Conv2D(48, (8, 7), padding='same', activation='relu', kernel_initializer=initializers.he_uniform())(inp)
	b1 = BatchNormalization()(f1)
	print(K.int_shape(b1))

	f2 = Conv2D(32, (32, 7), padding='same', activation='relu', kernel_initializer=initializers.he_uniform())(inp)
	b2 = BatchNormalization()(f2)
	print(K.int_shape(b2))

	f3 = Conv2D(16, (64, 7), padding='same', activation='relu', kernel_initializer=initializers.he_uniform())(inp)
	b3 = BatchNormalization()(f3)
	print(K.int_shape(b3))

	f4 = Conv2D(16, (90, 7), padding='same', activation='relu', kernel_initializer=initializers.he_uniform())(inp)
	b4 = BatchNormalization()(f4)
	print(K.int_shape(b4))
	concat = Concatenate(axis=3)([b1, b2, b3, b4])
	print(K.int_shape(concat))
	pool = MaxPooling2D((5, 5))(concat)
	
	fin = Conv2D(120, (2, 2), padding='valid', activation='relu', kernel_initializer=initializers.he_uniform())(pool)
	print(K.int_shape(fin))

	# pool2 = MaxPooling2D((7, 8))(fin)
	# print(K.int_shape(pool2))

	flat = GlobalMaxPooling2D()(fin)

	dense = Dense(64, activation='relu')(flat)
	dense_drop = Dropout(0.2)(dense)

	dense2 = Dense(64, activation='relu')(dense_drop)
	dense_drop2 = Dropout(0.2)(dense2)

	output = Dense(config.n_classes, activation='softmax')(dense_drop2) 
	print(K.int_shape(output))

	model = models.Model(inputs=inp, outputs=output)
	opt = optimizers.Adam(config.learning_rate)
	model.compile(optimizer=opt, loss=losses.categorical_crossentropy, metrics=['acc'])

	return model













