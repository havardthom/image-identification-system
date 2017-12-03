#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Written by Håvard Thom
"""Custom model functions"""
from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint

from cnn.networks import CustomModel
import config as cfg

def train_custom_model(nb_epochs, batch_size):
	"""Train custom model"""
	# Create custom model
	model = CustomModel.build(depth=cfg.img_channels,
							  width=cfg.img_width,
							  height=cfg.img_height,
							  nb_classes=cfg.nb_classes)

	# Compile model
	model.compile(loss='{}_crossentropy'.format(cfg.classmode),
				  optimizer=Adam(lr=1e-4),
				  metrics=['accuracy'])

	# Print model summary
	model.summary()

	# Initialize ImageDataGenerator for training data (with data augmentation)
	train_datagen = ImageDataGenerator(
		rescale=1./255,		   # scale from uint8 [0-255] to float32 [0,1]
		width_shift_range=0.05,  # randomly shift images horizontally (fraction of total width)
		height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
		horizontal_flip=True,	# randomly flip images horizontally
		vertical_flip=True,		 # randomly flip images vertically
		fill_mode='nearest')

	# Initialize ImageDataGenerator for val data
	val_datagen = ImageDataGenerator(rescale=1./255)

	# Create flow_from directory generators which takes the path to a directory,
	# and generates batches of augmented/normalized data
	train_generator = train_datagen.flow_from_directory(
									cfg.train_data_dir,
									target_size=(cfg.img_height, cfg.img_width),
									batch_size=batch_size,
									class_mode=cfg.classmode,
									color_mode='grayscale')

	val_generator = val_datagen.flow_from_directory(
								cfg.val_data_dir,
								target_size=(cfg.img_height, cfg.img_width),
								batch_size=batch_size,
								class_mode=cfg.classmode,
								color_mode='grayscale',
								shuffle=False)

	# Save weights with best val loss
	model_checkpoint = ModelCheckpoint(cfg.model_weights_path,
									   save_best_only=True,
									   save_weights_only=True,
									   monitor='val_loss')

	# Start training
	history = model.fit_generator(
					train_generator,
					samples_per_epoch=cfg.nb_train_samples,
					nb_epoch=nb_epochs,
					validation_data=val_generator,
					nb_val_samples=cfg.nb_val_samples,
					callbacks=[model_checkpoint])

	# Load best weights to get val data predictions
	model.load_weights(cfg.model_weights_path)

	# Need to recreate val data generator
	val_generator = val_datagen.flow_from_directory(
								cfg.val_data_dir,
								target_size=(cfg.img_height, cfg.img_width),
								batch_size=batch_size,
								class_mode=cfg.classmode,
								color_mode='grayscale',
								shuffle=False)

	# Get val data predictions
	val_pred_proba = model.predict_generator(val_generator, cfg.nb_val_samples)

	return model, history, val_pred_proba
