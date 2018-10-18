#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 23:36:28 2018

@author: brandinho
"""

import numpy as np
import tensorflow as tf

class PolicyNetwork():
    def __init__(self, sess, n_features, n_actions, neurons, learning_rate):
        self.sess = sess
        self.n_features = n_features
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.neurons = neurons
        if len(self.neurons) != 2:
            raise ValueError('You either need to use 2 layers or change the configurations below!')
        
        self.population = np.random.rand(self.population_size, self.neurons[0] * (self.n_features + 1 + self.neurons[1]) + 2 * self.neurons[1] + 1)        
        
        self.inputs = tf.placeholder(shape = [None, self.n_features], dtype = tf.float32)
        
        self.hidden_weights = tf.placeholder(shape = [self.n_features, self.neurons[0]], dtype = tf.float32)  
        self.hidden_bias = tf.placeholder(shape = [1, self.neurons[0]], dtype = tf.float32)
        self.hidden_layer = tf.nn.elu(tf.matmul(self.inputs, self.hidden_weights) + self.hidden_bias)
        
        self.hidden_weights_2 = tf.placeholder(shape = [self.neurons[0], self.neurons[1]], dtype = tf.float32)  
        self.hidden_bias_2 = tf.placeholder(shape = [1, self.neurons[1]], dtype = tf.float32)
        self.hidden_layer_2 = tf.nn.elu(tf.matmul(self.hidden_layer, self.hidden_weights_2) + self.hidden_bias_2)        
        
        self.policy_weights = tf.placeholder(shape = [self.neurons[1], self.n_actions], dtype = tf.float32)
        self.policy_bias = tf.placeholder(shape = [1, self.n_actions], dtype = tf.float32)
        self.policy = tf.nn.tanh(tf.nn.relu(tf.matmul(self.hidden_layer_2, self.policy_weights) + self.policy_bias))   

