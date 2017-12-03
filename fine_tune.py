#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Written by Håvard Thom
"""Finetune model functions"""
from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint

from cnn.networks import PretrainedVGG16NoTop
import config as cfg
from utils import get_model_files, load_model, decay_lr

def train_fine_tune_model(nb_epochs, batch_size):
	"""Train fine-tune model"""
	# Create pretrained VGG16 convolutional base
	model = PretrainedVGG16NoTop.build(depth=cfg.img_channels,
									   width=cfg.img_width,
									   height=cfg.img_height)

	# Get pretrained bottleneck model files
	bm_weights_dir = cfg.model_weights_dir.replace('finetune', 'bottleneck')
	bm_arch_dir = cfg.model_arch_dir.replace('finetune', 'bottleneck')
	bm_arch_path, bm_weights_path = get_model_files(bm_arch_dir, bm_weights_dir)

	# Load bottleneck model
	bottleneck_model = load_model(bm_arch_path, bm_weights_path)

	# Add the pretrained bottleneck model on top of the pretrained VGG16 convolutional base
	model.add(bottleneck_model)

	# Set the first 14 layers (up to the last convolutional block)
	# to non-trainable (weights will not be updated)
	for layer in model.layers[:14]:
		layer.trainable = False

	# Compile model
	model.compile(loss='{}_crossentropy'.format(cfg.classmode),
				  optimizer=SGD(lr=1e-4, momentum=0.9, nesterov=True),
				  metrics=['accuracy'])

	# Print model summary
	model.summary()

	# Initialize ImageDataGenerator for training data (with data augmentation)
	train_datagen = ImageDataGenerator(
			featurewise_center=True,
			width_shift_range=0.05,
			height_shift_range=0.1,
			horizontal_flip=True,
			vertical_flip=True,
			fill_mode='nearest')

	# Set imagenet mean (which is subtracted from images)
	train_datagen.mean = np.array([103.939, 116.779, 123.68],
								  dtype=np.float32).reshape(3,1,1)

	# Initialize ImageDataGenerator for val data
	val_datagen = ImageDataGenerator(featurewise_center=True)
	val_datagen.mean = np.array([103.939, 116.779, 123.68],
								dtype=np.float32).reshape(3,1,1)

	# Create flow_from directory generators which takes the path to a directory,
	# and generates batches of augmented/normalized data
	train_generator = train_datagen.flow_from_directory(
									cfg.train_data_dir,
									target_size=(cfg.img_height, cfg.img_width),
									batch_size=batch_size,
									class_mode=cfg.classmode)

	val_generator = val_datagen.flow_from_directory(
								cfg.val_data_dir,
								target_size=(cfg.img_height, cfg.img_width),
								batch_size=batch_size,
								class_mode=cfg.classmode,
								shuffle=False)

	# Save weights with best val loss
	model_checkpoint = ModelCheckpoint(cfg.model_weights_path,
									   save_best_only=True,
									   save_weights_only=True,
									   monitor='val_loss')

	# Decay learning rate by half every 10 epochs
	decay = decay_lr(10, 0.5)

	# Start training
	history = model.fit_generator(
					train_generator,
					samples_per_epoch=cfg.nb_train_samples,
					nb_epoch=nb_epochs,
					val_data=val_generator,
					nb_val_samples=cfg.nb_val_samples,
					callbacks=[model_checkpoint, decay])

	# Load best weights to get val data predictions
	model.load_weights(cfg.model_weights_path)

	# Need to recreate val data generator
	val_generator = val_datagen.flow_from_directory(
								cfg.val_data_dir,
								target_size=(cfg.img_height, cfg.img_width),
								batch_size=batch_size,
								class_mode=cfg.classmode,
								shuffle=False)

	# Get val data predictions
	val_pred_proba = model.predict_generator(val_generator, cfg.nb_val_samples)

	return model, history, val_pred_proba
