#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Written by Håvard Thom
"""
"""
from __future__ import print_function
import argparse
import os
import os.path as osp
import numpy as np
import glob
from sklearn.model_selection import train_test_split

import config as cfg
from cnn.networks import PretrainedVGG16NoTop
from bottleneck import generate_and_save_bottleneck_features, train_bottleneck_model
from custom import train_custom_model
from fine_tune import train_fine_tune_model

from utils import set_cfg_values, set_cfg_paths, override_keras_directory_iterator_next
from utils import load_train_dir, load_test_dir, write_data_directory, create_directories
from utils import write_val_results, show_mc_val_images, write_test_results, show_test_images
from utils import get_model_files, save_model, load_model, save_graphs

seed = 7
np.random.seed(seed)

def parse_args():
    """Parse input arguments"""
    parser = argparse.ArgumentParser(
    description='',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-m', '--model',
            help='Use custom, bottleneck or finetune model',
            default='custom', type=str)
    parser.add_argument('-tr_d', '--train_dir',
            help='Directory with training images',
            default='../rodentcam_training_10k_11c', type=str)
    parser.add_argument('-te_d', '--test_dir',
            help='(Optional) Directory with test images to predict',
            default=None, type=str)
    parser.add_argument('-c', '--crop',
            help='Crop black borders in rodentcam images',
            default=False, action='store_true')
    parser.add_argument('-s_t', '--show_test',
            help='Show test images after predicting',
            default=False, action='store_true')
    parser.add_argument('-s_v', '--show_val',
            help='Show misclassified validation images after training',
            default=False, action='store_true')

    args = parser.parse_args()

    if (args.model != 'custom'
    and args.model != 'bottleneck'
    and args.model != 'finetune'):
        parser.error("Invalid model: {}. Use custom, bottleneck or finetune.".format(args.model))

    if not osp.isdir(args.train_dir):
        parser.error("Training directory: {} does not exist.".format(args.train_dir))

    if args.test_dir != None:
        if not osp.isdir(args.test_dir):
            parser.error("Test directory: {} does not exist.".format(args.test_dir))

    return args

def train_model(train_dir, model_name, show_val):
    prepare_data(train_dir)

    if model_name == 'custom':
        model, history, val_pred_probs = train_custom_model(100, 64)
    elif model_name == 'bottleneck':
        override_keras_directory_iterator_next()

        generate_and_save_bottleneck_features(64)

        model, history, val_pred_probs = train_bottleneck_model(100, 64)
    elif model_name == 'finetune':
        override_keras_directory_iterator_next()

        model, history, val_pred_probs = train_fine_tune_model(30, 32)

    min_loss, min_loss_acc = find_min_loss_acc(history)
    postfix = '_{0:.2f}l_{1:.2f}a'.format(min_loss, min_loss_acc)

    # Rename model weights
    try:
        print("Renaming model weights.\n")
        tmp_path = cfg.model_weights_path.replace('.h5', '{}.h5'.format(postfix))
        os.rename(cfg.model_weights_path, tmp_path)
    except Exception as e:
        print("Renaming model weights failed: {}".format(e))

    # Save model architecture
    try:
        print("Saving model architecture.\n")
        save_model(model, postfix)
    except Exception as e:
        print("Saving model architecture failed: {}".format(e))

    # Save loss and accuracy graphs
    try:
        print("Saving loss and accuracy graphs.\n")
        save_graphs(history, postfix)
    except Exception as e:
        print("Saving loss and accuracy graphs failed: {}".format(e))

    # Save validation results
    try:
        print("Saving validation results.\n")
        # Get validation images
        val_imgs, val_labels, val_img_paths = load_train_dir(cfg.val_data_dir)

        # Reorder predictions probabilities if binary class
        if cfg.nb_classes == 2:
            val_pred_probs = np.append(1.0-val_pred_probs, val_pred_probs, axis=1)

        # Get predictions labels
        val_pred_labels = np.argmax(val_pred_probs, axis=1)

        # Save validation results
        write_val_results(postfix, val_img_paths, val_labels, val_pred_probs, val_pred_labels)

        # Show misclassified validation images
        if show_val:
            show_mc_val_images(train_dir, val_labels, val_img_paths, val_pred_probs, val_pred_labels)
    except Exception as e:
        print("Saving validation results failed: {}".format(e))

def prepare_data(train_dir):
    """Prepare data for training"""
    # No need to create training or val data directory if they already exist
    if osp.isdir(cfg.train_data_dir) and osp.isdir(cfg.val_data_dir):
        print("Using existing data directories: \n{}\n{}\n".format(cfg.train_data_dir, cfg.val_data_dir))
        # Still need to set number of training and val images
        for class_name in cfg.classes:
            cfg.nb_train_samples += len(os.listdir(osp.join(cfg.train_data_dir, class_name)))
            cfg.nb_val_samples += len(os.listdir(osp.join(cfg.val_data_dir, class_name)))
    else:
        print("Loading training images...\n")
        # Load all training images from given directory
        imgs , _, img_paths = load_train_dir(train_dir)

        # Split into training (80%) and val (20%) sets
        train_imgs, val_imgs, train_img_paths, val_img_paths = train_test_split(
        imgs, img_paths, test_size=0.20, random_state=seed)

        # Set number of training samples and val samples
        cfg.nb_train_samples = len(train_imgs)
        cfg.nb_val_samples = len(val_imgs)

        # Create data directories for training and val data
    	for class_name in cfg.classes:
    		create_directories(osp.join(cfg.train_data_dir, class_name))
    		create_directories(osp.join(cfg.val_data_dir, class_name))

        print("Writing images to training data directory.\n")
        write_data_directory(train_imgs, train_img_paths, cfg.train_data_dir)
        print("Writing images to val data directory.\n")
        write_data_directory(val_imgs, val_img_paths, cfg.val_data_dir)

def find_min_loss_acc(history):
    """Find last occurrence of minimum val loss and corresponding accuracy"""
    min_loss = 0
    min_loss_acc = 0
    try:
        print("Getting minimum loss and corresponding accuracy")
        loss_list = history.history['val_loss']
        min_loss = min(loss_list)
        min_loss_idx = len(loss_list) - loss_list[::-1].index(min_loss) - 1
        min_loss_acc = history.history['val_acc'][min_loss_idx] * 100
    except Exception as e:
        print("Finding minimum loss failed: {}".format(e))

    return min_loss, min_loss_acc

def test_model(test_dir, model_name, show_test):
    """Test a model"""
    # Load model
    model_arch_path, model_weights_path = get_model_files(cfg.model_arch_dir, cfg.model_weights_dir)
    model = load_model(model_arch_path, model_weights_path)

    # Print model summary
    model.summary()

    # Get test images
    test_imgs, test_img_paths = load_test_dir(test_dir)

    # Get prediction probabilities for test images
    test_pred_probs = predict(model_name, test_imgs)

    # Reorder predictions probabilities if binary class
    if cfg.nb_classes == 2:
        test_pred_probs = np.append(1.0-test_pred_probs, test_pred_probs, axis=1)

    # Get prediction labels
    test_pred_labels = np.argmax(test_pred_probs, axis=1)

    # Save test results
    try:
        print("Saving test results.\n")
        write_test_results(test_dir + "_results.txt", test_img_paths,
        test_pred_probs, test_pred_labels, model_arch_path, model_weights_path)
    except Exception as e:
        print("Saving test results failed: {}".format(e))

    # Show test results
    if show_test:
        show_test_images(test_dir, test_pred_probs, test_pred_labels)

def predict(model_name, test_imgs):
    """Predict test images"""
    # Do some preprocessing
    if model_name == 'custom':
        test_imgs /= 255
    else:
        test_imgs -= np.array([123.68, 116.779, 103.939],
                              dtype=np.float32).reshape(3,1,1)

    # If bottleneck model, generate bottleneck features for test data
    if model_name == 'bottleneck':
        print("Creating bottleneck features for test data.\n")
        base_model = PretrainedVGG16NoTop.build(depth=cfg.img_channels,
                                                width=cfg.img_width,
                                                height=cfg.img_height)
        test_imgs = base_model.predict(test_imgs, verbose=1)

    # Get prediction probabilities for test images
    test_pred_probs = model.predict(test_imgs, verbose=1)
    return test_pred_probs

if __name__ == "__main__":
    args = parse_args()

    print('Called with args:')
    print(args)

    model_name = args.model
    train_dir = args.train_dir
    test_dir = args.test_dir
    crop = args.crop
    show_test = args.show_test
    show_val = args.show_val

    # Set config
    set_cfg_values(model_name, train_dir, crop)
    set_cfg_paths(model_name, train_dir)

    # Train model if test dir is not given. If it is given, test model.
    if test_dir == None:
        train_model(train_dir, model_name, show_val)
    else:
        test_model(test_dir, model_name, show_test)
