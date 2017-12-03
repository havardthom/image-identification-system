#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Written by Håvard Thom
"""Bottleneck model functions"""
from __future__ import print_function
import numpy as np
import os
import os.path as osp

from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils

from cnn.networks import BottleneckModel, PretrainedVGG16NoTop
from utils import load_np_array, decay_lr, save_np_array
import config as cfg

def train_bottleneck_model(nb_epochs, batch_size):
	"""Train bottleneck model"""
	# Load the training and validation bottleneck features
	train_data = load_np_array(cfg.bf_train_path)
	val_data = load_np_array(cfg.bf_val_path)

	# Get training and validation labels for bottleneck features
	# (we know the images are in sorted order)
	train_labels = []
	val_labels = []
	k = 0
	for class_name in cfg.classes:
		train_labels += [k] * len(os.listdir(osp.join(cfg.train_data_dir, class_name)))
		val_labels += [k] * len(os.listdir(osp.join(cfg.val_data_dir, class_name)))
		k += 1

	# Create custom model
	model = BottleneckModel.build(input_shape=train_data.shape[1:],
								  nb_classes=cfg.nb_classes)

	# If multiclass, encode the labels to 1-to-K binary format
	if cfg.nb_classes != 2:
		train_labels = np_utils.to_categorical(train_labels, cfg.nb_classes)
		val_labels = np_utils.to_categorical(val_labels, cfg.nb_classes)

	# Compile model
	model.compile(loss='{}_crossentropy'.format(cfg.classmode),
				  optimizer=Adam(lr=5e-5),
				  metrics=['accuracy'])

	# Print model summary
	model.summary()

	# Save weights with best val loss
	model_checkpoint = ModelCheckpoint(cfg.model_weights_path,
									   save_best_only=True,
									   save_weights_only=True,
									   monitor='val_loss')

	# Decay learning rate by half every 20 epochs
	decay = decay_lr(20, 0.5)

	# Start training
	history = model.fit(train_data,
			  			train_labels,
			  			nb_epoch=nb_epochs,
			  			batch_size=batch_size,
			  			validation_data=(val_data, val_labels),
			  			callbacks=[model_checkpoint, decay])

	# Load best weights to get val data predictions
	model.load_weights(cfg.model_weights_path)

	# Get val data predictions
	val_pred_proba = model.predict(val_data)

	return model, history, val_pred_proba

def generate_and_save_bottleneck_features(batch_size):
	"""Generate and save bottleneck features"""
	# Create pretrained VGG16 convolutional base
	model = PretrainedVGG16NoTop.build(depth=cfg.img_channels,
									   width=cfg.img_width,
									   height=cfg.img_height)

	# Initialize ImageDataGenerator and set ImageNet mean (which is subtracted from images)
	datagen = ImageDataGenerator(featurewise_center=True)
	datagen.mean = np.array([103.939, 116.779, 123.68],
							dtype=np.float32).reshape(3,1,1)

	# Generate and save bottleneck features for training data if they do not exist
	if not osp.isfile(cfg.bf_train_path):
		generator = datagen.flow_from_directory(
							cfg.train_data_dir,
							target_size=(cfg.img_width, cfg.img_height),
							batch_size=batch_size,
							class_mode=cfg.classmode,
							shuffle=False)

		print("Creating bottleneck features for training data: \n{}\n".format(cfg.bf_train_path))
		bottleneck_features_train = model.predict_generator(generator, cfg.nb_train_samples)
		save_np_array(cfg.bf_train_path, bottleneck_features_train)
	else:
		print("Using existing bottleneck features for training data: \n{}\n".format(cfg.bf_train_path))

	# Generate and save bottleneck features for validation data if they do not exist
	if not osp.isfile(cfg.bf_val_path):
		generator = datagen.flow_from_directory(
							cfg.val_data_dir,
							target_size=(cfg.img_width, cfg.img_height),
							batch_size=batch_size,
							class_mode=cfg.classmode,
							shuffle=False)

		print("Creating bottleneck features for val data: \n{}\n".format(cfg.bf_val_path))
		bottleneck_features_val = model.predict_generator(generator, cfg.nb_val_samples)
		save_np_array(cfg.bf_val_path, bottleneck_features_val)
	else:
		print("Using existing bottleneck features for val data: \n{}\n".format(cfg.bf_val_path))
