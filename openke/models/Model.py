#coding:utf-8
import numpy as np
import tensorflow as tf

class Model(object):


	def get_positive_instance(self, in_batch = True):
		if in_batch:
			return [self.postive_h, self.postive_t, self.postive_r]
		else:
			B = self.batch_size
			return [self.batch_h[0:B], self.batch_t[0:B], self.batch_r[0:B]]


	def get_negative_instance(self, in_batch = True):
		if in_batch:
			return [self.negative_h, self.negative_t, self.negative_r]
		else:
			B, S = self.batch_size, self.batch_seq_size
			return [self.batch_h[B:S], self.batch_t[B:S], self.batch_r[B:S]]


	def get_all_instance(self, in_batch = False):
		if in_batch:
			N = self.negatives
			return [tf.transpose(tf.reshape(self.batch_h, [1+N,-1]), [1,0]),\
			tf.transpose(tf.reshape(self.batch_t, [1+N,-1]), [1,0]),\
			tf.transpose(tf.reshape(self.batch_r, [1+N,-1]), [1,0])]
		else:
			return [self.batch_h, self.batch_t, self.batch_r]


	def get_all_labels(self, in_batch = False):
		if in_batch:
			return tf.transpose(tf.reshape(self.batch_y, [1 + self.negatives, -1]), [1, 0])
		else:
			return self.batch_y


	def get_predict_instance(self):
		return [self.predict_h, self.predict_t, self.predict_r]


	def input_def(self):
		B, S, N = self.batch_size, self.batch_seq_size, self.negatives
		self.batch_h = tf.placeholder(tf.int64, [S])
		self.batch_t = tf.placeholder(tf.int64, [S])
		self.batch_r = tf.placeholder(tf.int64, [S])
		self.batch_y = tf.placeholder(tf.float32, [S])
		self.postive_h = tf.transpose(tf.reshape(self.batch_h[0:B], [1,-1]), [1,0])
		self.postive_t = tf.transpose(tf.reshape(self.batch_t[0:B], [1,-1]), [1,0])
		self.postive_r = tf.transpose(tf.reshape(self.batch_r[0:B], [1,-1]), [1,0])
		self.negative_h = tf.transpose(tf.reshape(self.batch_h[B:S], [N,-1]), perm=[1,0])
		self.negative_t = tf.transpose(tf.reshape(self.batch_t[B:S], [N,-1]), perm=[1,0])
		self.negative_r = tf.transpose(tf.reshape(self.batch_r[B:S], [N,-1]), perm=[1,0])
		self.predict_h = tf.placeholder(tf.int64, [None])
		self.predict_t = tf.placeholder(tf.int64, [None])
		self.predict_r = tf.placeholder(tf.int64, [None])
		self.parameter_lists = []


	def embedding_def(self):
		pass


	def loss_def(self):
		pass


	def predict_def(self):
		pass


	def __init__(self, **config):
		self.batch_size = config['batch_size']
		self.batch_seq_size = config['batch_seq_size']
		self.negatives = config['negative_ent'] + config['negative_rel']

		with tf.name_scope("input"):
			self.input_def()

		with tf.name_scope("embedding"):
			self.embedding_def()

		with tf.name_scope("loss"):
			self.loss_def()

		with tf.name_scope("predict"):
			self.predict_def()

