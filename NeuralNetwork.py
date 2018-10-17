#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 23:36:28 2018

@author: brandinho
"""

import tensorflow as tf

class PolicyNetwork():
    def __init__(self, sess, n_features, n_actions, learning_rate):
        self.sess = sess
        self.n_features = n_features
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.neurons = [36, 18, 9]
        
        self.inputs = tf.placeholder(shape = [None, self.n_features], dtype = tf.float32)
        
        self.weights1 = tf.Variable(tf.random_normal([self.n_features, self.neurons[0]], stddev = tf.sqrt(2/(self.n_features + self.neurons[0]))))
        self.bias1 = tf.Variable(tf.zeros([1,self.neurons[0]]) + 0.01)
        self.layer1 = tf.nn.elu(tf.matmul(self.inputs, self.weights1) + self.bias1)
        
        self.weights2 = tf.Variable(tf.random_normal([self.neurons[0], self.neurons[1]], stddev = tf.sqrt(2/(self.neurons[0] + self.neurons[1]))))
        self.bias2 = tf.Variable(tf.zeros([1,self.neurons[1]]) + 0.01)
        self.layer2 = tf.nn.elu(tf.matmul(self.layer1, self.weights2) + self.bias2)
        
        self.weights3 = tf.Variable(tf.random_normal([self.neurons[1], self.neurons[2]], stddev = tf.sqrt(2/(self.neurons[1] + self.neurons[2]))))
        self.bias3 = tf.Variable(tf.zeros([1,self.neurons[2]]) + 0.01)
        self.layer3 = tf.nn.elu(tf.matmul(self.layer2, self.weights3) + self.bias3)
        
        self.weights4 = tf.Variable(tf.random_normal([self.neurons[2], self.n_actions], stddev = tf.sqrt(2/(self.neurons[2] + self.n_actions))))
        self.bias4 = tf.Variable(tf.zeros([1,self.n_actions]) + 0.01)
        self.policy = tf.nn.softmax(tf.matmul(self.layer3, self.weights4) + self.bias4)
