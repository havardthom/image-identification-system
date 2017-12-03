#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Written by Håvard Thom
"""Bottleneck model"""
from keras.models import Sequential
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.layers.core import Dropout
from keras.regularizers import l2

class BottleneckModel:
	@staticmethod
	def build(input_shape, nb_classes):
		# Build the bottleneck model
		model = Sequential()
		model.add(Flatten(input_shape=input_shape))
		model.add(Dense(4096, activation='relu', W_regularizer = l2(1e-2)))
		model.add(Dropout(0.6))
		model.add(Dense(4096, activation='relu', W_regularizer = l2(1e-2)))
		model.add(Dropout(0.6))

		if nb_classes == 2:
			model.add(Dense(1, activation='sigmoid'))
		else:
			model.add(Dense(nb_classes, activation='softmax'))

		return model
