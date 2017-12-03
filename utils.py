#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Written by Håvard Thom
"""Utility functions"""
from __future__ import print_function
import glob
import cv2
import numpy as np
import datetime
import os
import os.path as osp
from sklearn.metrics import accuracy_score, classification_report, log_loss, f1_score
import matplotlib.pyplot as plt

from keras.models import model_from_json
from keras.callbacks import Callback
from keras import backend as k

import config as cfg

def set_cfg_values(model_name, train_dir, crop):
	# Setup image properties
	if model_name == 'custom':
		cfg.color = 0
		cfg.img_width = 32
		cfg.img_height = 32
		cfg.img_channels = 1

	if crop:
		cfg.crop_height = (16, 704)
		cfg.crop_infix = '_no_border'

	# Get classes
	class_paths = sorted(glob.glob(osp.join(train_dir, '*')))
 	if len(class_paths) < 2:
		raise Exception("Training directory: {} does not contain enough classes".format(train_dir))

	for path in class_paths:
		class_name = path.split("/")[-1]
		cfg.classes.append(class_name)

	cfg.classes = np.asarray(cfg.classes)
	print("Using the following classes: {}".format(cfg.classes))

	# Set number of classes
	cfg.nb_classes = len(cfg.classes)

	# Set classmode used in training
	if cfg.nb_classes == 2:
		cfg.classmode = 'binary'
	else:
		cfg.classmode = 'categorical'

def set_cfg_paths(model_name, train_dir):
	"""Set configuration dirs and paths"""
	dataset_name = train_dir.split("/")[-1]
	size = '{}x{}{}'.format(cfg.img_width, cfg.img_height, cfg.crop_infix)
	classes = '{}c'.format(cfg.nb_classes)

	data_dir = osp.join('data', dataset_name, size, classes)
	output_dir = osp.join('output', dataset_name, model_name, size, classes)

	# Set directory names for data
	cfg.train_data_dir = osp.join(data_dir, 'train')
	cfg.val_data_dir = osp.join(data_dir, 'val')

	# Set directory names for output
	cfg.model_weights_dir = osp.join(output_dir, 'weights')
	cfg.model_arch_dir = osp.join(output_dir, 'architectures')
	cfg.model_graphs_dir = osp.join(output_dir, 'graphs')
	cfg.model_results_dir = osp.join(output_dir, 'results')

	# Set path for model weights
	cfg.model_weights_path = osp.join(cfg.model_weights_dir,
									  '{}_weights_{}.h5'.format(model_name, classes))
	cfg.model_arch_path = osp.join(cfg.model_arch_dir,
								   '{}_arch_{}.json'.format(model_name, classes))
	# Set paths for graphs
	cfg.loss_graph_path = osp.join(cfg.model_graphs_dir,
								   '{}_loss_graph_{}.png'.format(model_name, classes))
	cfg.acc_graph_path = osp.join(cfg.model_graphs_dir,
								  '{}_acc_graph_{}.png'.format(model_name, classes))
	# Set path for val results
	cfg.val_results_path = osp.join(cfg.model_results_dir,
									'{}_val_results_{}.txt'.format(model_name, classes))
	# Set paths for bottleneck features
	cfg.bf_train_path = osp.join(data_dir, 'bottleneck_features_train_{}.npy'.format(classes))
	cfg.bf_val_path = osp.join(data_dir, 'bottleneck_features_val_{}.npy'.format(classes))

	# Create directories for model
	create_directories(cfg.model_weights_dir)
	create_directories(cfg.model_arch_dir)
	create_directories(cfg.model_graphs_dir)
	create_directories(cfg.model_results_dir)

def create_directories(directory):
	"""Create directory tree if it does not exist"""
	if not osp.exists(directory):
		os.makedirs(directory)

def write_data_directory(imgs, img_paths, data_dir, test=False):
	"""Write images to data directory"""
	for i, img_path in enumerate(img_paths):
		splt = img_path.split("/")
		img_name = splt[-1]

		if not test:
			class_name = splt[-2]
			class_dir = osp.join(data_dir, class_name)
		else:
			class_dir = osp.join(data_dir, 'test')

		cv2.imwrite(osp.join(class_dir, img_name), imgs[i])

def load_train_dir(img_dir):
	"""Load training images from directory"""
	# Get all class directories
	class_dirs = sorted(glob.glob(osp.join(img_dir, '*')))

	# Tuple of supported image formats
	formats = ('*.JPG', '*.jpg', '*.jpeg', '*.png')

	nb_imgs = 0
	# Get total number of images
	for f in formats:
		nb_imgs += len(glob.glob(osp.join(img_dir, '*', f)))

	# Initialize numpy array to hold images
	if cfg.color:
		shape = (nb_imgs, cfg.img_width, cfg.img_height, cfg.img_channels)
	else:
		shape = (nb_imgs, cfg.img_width, cfg.img_height)
	train_imgs = np.zeros(shape, dtype='uint8')

	# Initialize label and name list
	train_labels = []
	train_img_paths = []

	# Traverse through class directories
	i = 0
	for label, class_dir in enumerate(class_dirs):
		class_name = class_dir.rsplit("/", 1)[1]
		print("Loading class: {} with label: {}\n".format(class_name, label))

		# Get all image paths in class directory
		img_paths = []
		for f in formats:
			img_paths.extend(glob.glob(osp.join(class_dir, f)))
		img_paths = sorted(img_paths)

		for img_path in img_paths:
			# Read image
			img = cv2.imread(img_path, cfg.color)

			# Crop out top and bottom border
			if cfg.crop_height:
				img = img[cfg.crop_height[0]:cfg.crop_height[1]]

			# Resize image
			img = cv2.resize(img, dsize=(cfg.img_width, cfg.img_height),
			interpolation=cv2.INTER_CUBIC)

			# Add image, label and image name to their respective array/list
			train_imgs[i] = img.astype('uint8')
			train_labels.append(label)
			train_img_paths.append(img_path)
			i+=1

	# Convert labels and image names to numpy array
	train_labels = np.asarray(train_labels)
	train_img_paths = np.asarray(train_img_paths)

	return train_imgs, train_labels, train_img_paths

def load_test_dir(img_dir):
	"""Load test images from directory"""
	# Tuple of supported image formats
	formats = ('*.JPG', '*.jpg', '*.jpeg', '*.png')

	img_paths = []
	# Get all image paths in test directory
	for f in formats:
		img_paths.extend(glob.glob(osp.join(img_dir, f)))

	img_paths = sorted(img_paths)
	nb_imgs = len(img_paths)

	# Initialize numpy array to hold images
	shape = (nb_imgs, cfg.img_channels, cfg.img_width, cfg.img_height)
	test_imgs = np.zeros(shape, dtype='float32')
	test_img_paths = []

	for i, img_path in enumerate(img_paths):
		# Read test image
		img = cv2.imread(img_path, cfg.color)

		# Crop out top and bottom border
		if cfg.crop_height:
			img = img[cfg.crop_height[0]:cfg.crop_height[1]]

		# Resize image
		img = cv2.resize(img, dsize=(cfg.img_width, cfg.img_height),
		interpolation=cv2.INTER_CUBIC)

		# Reorder shape to fit theanos specifications:
		# [number of images X img channels X img width X img height]
		if cfg.color:
			img = img.transpose(2, 0, 1)
		else:
			img = np.expand_dims(img, axis=0)

		test_imgs[i] = img.astype('float32')
		test_img_paths.append(img_path)

	test_img_paths = np.asarray(test_img_paths)

	return test_imgs, test_img_paths

def get_model_files(model_arch_dir, model_weights_dir):
	"""Get model files"""
	print("Getting model files from: \n{}\n{}\n".format(
	model_arch_dir, model_weights_dir))

	model_arch_paths = sorted(glob.glob(osp.join(model_arch_dir, '*.json')))
	model_weight_paths = sorted(glob.glob(osp.join(model_weights_dir, '*.h5')))

	if len(model_arch_paths) == 0:
		raise Exception("\nNo model architecture found in: {}\nTrain model first.".format(
		model_arch_dir))
	elif len(model_weight_paths) == 0:
		raise Exception("\nNo model weights found in: {}\nTrain model first.".format(
		model_weights_dir))

	print("Using model files: \n{}\n{}\n".format(
	model_arch_paths[0], model_weight_paths[0]))

	return model_arch_paths[0], model_weight_paths[0]

def load_model(model_path, model_weights_path=None):
	"""Load a model from disk"""
	with open(model_path, 'r') as infile:
		model = model_from_json(infile.read())
	if model_weights_path:
		model.load_weights(model_weights_path)
	return model

def save_model(model, postfix):
	"""Save model architecture to disk"""
	model_arch_path = cfg.model_arch_path.replace('.json', '{}.json'.format(postfix))
	model_json = model.to_json()
	with open(model_arch_path, 'w') as outfile:
		outfile.write(model_json)

def load_np_array(np_path):
	"""Load a numpy array from disk"""
	with open(np_path, 'r') as infile:
		return np.load(infile)

def save_np_array(np_path, np_array):
	"""Save a numpy array to disk"""
	with open(np_path, 'w') as outfile:
		np.save(outfile, np_array)

def save_graphs(history, postfix):
	"""Save accuracy and loss trend graphs"""
	loss_graph_path = cfg.loss_graph_path.replace('.png', '{}.png'.format(postfix))
	acc_graph_path = cfg.acc_graph_path.replace('.png', '{}.png'.format(postfix))

	plt.xlabel('Epochs')
	plt.ylabel('Loss')
	plt.title('Model Loss Trend')
	plt.plot(history.history['loss'], 'blue', label='Training Loss')
	plt.plot(history.history['val_loss'], 'green', label='Validation Loss')
	plt.legend()
	plt.savefig(loss_graph_path, bbox_inches='tight')
	plt.close()

	plt.xlabel('Epochs')
	plt.ylabel('Accuracy')
	plt.title('Model Accuracy Trend')
	plt.plot(history.history['acc'], 'blue', label='Training Accuracy')
	plt.plot(history.history['val_acc'], 'green', label='Validation Accuracy')
	plt.legend(loc='lower right')
	plt.savefig(acc_graph_path, bbox_inches='tight')
	plt.close()

def write_val_results(postfix, val_img_paths, val_labels, val_pred_probs, val_pred_labels):
	"""Write validation results"""
	val_results_path = cfg.val_results_path.replace('.txt', '{}.txt'.format(postfix))

	# Get various metrics with sklearn library
	accuracy = accuracy_score(y_true=val_labels, y_pred=val_pred_labels)
	report = classification_report(val_labels, val_pred_labels, target_names=cfg.classes)
	loss = log_loss(val_labels, val_pred_probs)
	f1 = f1_score(val_labels, val_pred_labels, average='weighted')

	# Get index of images that are misclassified
	mc_idx = np.where(val_labels != val_pred_labels)

	# Get count of misclassified images for each class
	unique, counts = np.unique(val_labels[mc_idx], return_counts=True)
	misclassified = dict(zip(cfg.classes[unique], counts))

	# Get prediction probabilities
	probabilities = np.amax(val_pred_probs[mc_idx], axis=1)*100
	probabilities = np.around(probabilities, decimals=2)

	# Create an array of [image name, real class, predicted class, prediction probability]
	results = np.vstack((val_img_paths[mc_idx], cfg.classes[val_labels[mc_idx]],
	cfg.classes[val_pred_labels[mc_idx]], probabilities)).transpose(1,0)

	# Write val results
	with open(val_results_path, 'w') as result_file:
		result_file.write('Accuracy: {}\n'.format(accuracy))
		result_file.write('{}\n'.format(report))
		result_file.write("Number of misclassified images out of {} images: {}\n".format(
		val_labels.shape[0], len(mc_idx)))
		result_file.write("Number of misclassified images per class: {}\n".format(
		misclassified))
		result_file.write('Log loss: {}\n'.format(loss))
		result_file.write('F1 score: {}\n\n'.format(f1))
		result_file.write('{0:130} - {1:12} - {2:17} - {3:12}\n\n'.format(
		'Image name', 'Real class', 'Predicted class', 'Probability'))

		for i in results:
			result_file.write('{0:130} - {1:12} - {2:17} - {3:.2f}%\n'.format(
			i[0], i[1], i[2], i[3].astype('float32')))

def show_mc_val_images(train_dir, val_labels, val_img_paths, val_pred_probs, val_pred_labels):
	"""Show misclassified validation images"""
	mc_indices = np.where(val_labels != val_pred_labels)[0]
	mc_val_img_paths = val_img_paths[mc_indices]
	nb_mc_imgs = len(mc_val_img_paths)

	train_img_paths = []

	# Get image paths from original training directory
	for i in range(nb_mc_imgs):
		splt = mc_val_img_paths[i].split('/')
		img_name = splt[-1]
		class_name = splt[-2]
		train_img_paths.append(osp.join(train_dir, class_name, img_name))

	i = 0
	while 0 <= i < nb_mc_imgs:
		img = cv2.imread(train_img_paths[i])
		mc_idx = mc_indices[i]

		real_class = cfg.classes[val_labels[mc_idx]]
		predicted_class = cfg.classes[val_pred_labels[mc_idx]]
		probability = np.around(val_pred_probs[mc_idx][val_pred_labels[mc_idx]]*100, decimals=2)

		cv2.putText(img, "Real: {}   Prediction: {} ({}%) ({} of {})".format(
		real_class, predicted_class, probability, i+1, nb_mc_imgs),
		(20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

		cv2.imshow('Classification', img)

		# Wait for keypress
		key = cv2.waitKey(0) & 0xFF
		# b is prev img, q is quit and rest is next img
		if key == ord('q'):
			break
		elif key == ord('b'):
			i -= 1
		else:
			i += 1

def write_test_results(test_results_path, test_img_paths, test_pred_probs,
					   test_pred_labels, model_name, model_weight_name):
	"""Write test data results"""
	unique, counts = np.unique(pred_labels, return_counts=True)
	total = dict(zip(cfg.classes[unique], counts))

	probabilities = np.amax(pred_probs, axis=1)*100
	probabilities = np.around(probabilities, decimals=2)

	results = np.vstack((test_img_paths, cfg.classes[pred_labels],
	pred_labels, probabilities)).transpose(1,0)

	with open(test_results_path, 'w') as result_file:
		result_file.write('Date: {}\n'.format(datetime.date.today()))
		result_file.write('Model: {}\n'.format(model_name))
		result_file.write('Weights: {}\n\n'.format(model_weight_name))
		result_file.write('Total images per class: {}\n\n'.format(total))
		result_file.write('{0:70} {1:20} {2:10} {3}\n'.format(
		'Image name', 'Predicted class', 'Label', 'Probability'))

		for i in results:
			result_file.write('{0:70} {1:20} {2:10} {3:.2f}%\n'.format(
			i[0], i[1], i[2], i[3].astype('float32')))

def show_test_images(test_img_dir, pred_probs, pred_labels):
	"""Show test result images"""
	img_paths = sorted(glob.glob(osp.join(test_img_dir, '*')))
	nb_imgs = len(img_paths)

	i = 0
	while 0 <= i < nb_imgs:
		img = cv2.imread(img_paths[i])

		predicted_class = cfg.classes[pred_labels[i]]
		probability = np.around(pred_probs[i][pred_labels[i]]*100, decimals=2)

		cv2.putText(img, "Prediction: {} ({}%) ({} of {})".format(
		predicted_class, probability, i+1, nb_imgs),
		(20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

		cv2.imshow('Classification', img)

		# Wait for keypress
		key = cv2.waitKey(0) & 0xFF
		# b is prev img, q is quit and rest is next img
		if key == ord('q'):
			break
		elif key == ord('b'):
			i -= 1
		else:
			i += 1

def override_keras_directory_iterator_next():
	"""Override .next method of DirectoryIterator in keras to
	 reorder color channels for images from RGB to BGR"""
	from keras.preprocessing.image import DirectoryIterator
	original_next = DirectoryIterator.next
	# Do not allow to override one more time
	if 'custom_next' in str(original_next):
		return

	def custom_next(self):
		batch_x, batch_y = original_next(self)
		batch_x = batch_x[:, ::-1, :, :]
		return batch_x, batch_y

	DirectoryIterator.next = custom_next

class decay_lr(Callback):
	"""Custom callback class to decay learning rate"""
	def __init__(self, decay_epoch, decay_rate):
		super(decay_lr, self).__init__()
		self.decay_epoch = decay_epoch
		self.decay_rate = decay_rate

	def on_epoch_begin(self, epoch, logs={}):
		old_lr = self.model.optimizer.lr.get_value()
		if epoch > 1 and epoch % self.decay_epoch == 0:
			new_lr = self.decay_rate*old_lr
			k.set_value(self.model.optimizer.lr, new_lr)
		else:
			k.set_value(self.model.optimizer.lr, old_lr)
