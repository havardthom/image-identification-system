#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Written by Håvard Thom
"""Configuration file"""

# Directory names for data
train_data_dir = ""
val_data_dir = ""
test_data_dir = ""

# Directory names for output
model_weights_dir = ""
model_arch_dir = ""
model_graphs_dir = ""
model_results_dir = ""

# Path for model
model_weights_path = ""
model_arch_path = ""

# Paths for bottleneck features
bf_train_path = ""
bf_val_path = ""

# Path for graphs
loss_graph_path = ""
acc_graph_path = ""

# Data properties
classes = []
nb_classes = 0
nb_train_samples = 0
nb_val_samples = 0
nb_test_samples = 0

# Image properties
color = 1
img_width = 224
img_height = 224
img_channels = 3
crop_height = None
crop_infix = ""

# Model settings
nb_epoch = 0
batch_size = 0
classmode = ""
