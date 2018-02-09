#coding:utf-8
import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer
from . import Model


class ComplEx(Model):


	def embedding_def(self):
		d, cE, cR = self.hiddensize, self.entities, self.relations
		self.ent1_embeddings = tf.get_variable("ent1_embeddings", [cE, d], initializer=xavier_initializer(uniform=True))
		self.rel1_embeddings = tf.get_variable("rel1_embeddings", [cR, d], initializer=xavier_initializer(uniform=True))
		self.ent2_embeddings = tf.get_variable("ent2_embeddings", [cE, d], initializer=xavier_initializer(uniform=True))
		self.rel2_embeddings = tf.get_variable("rel2_embeddings", [cR, d], initializer=xavier_initializer(uniform=True))
		self.parameter_lists = {
				"ent_re_embeddings":self.ent1_embeddings,
				"ent_im_embeddings":self.ent2_embeddings,
				"rel_re_embeddings":self.rel1_embeddings,
				"rel_im_embeddings":self.rel2_embeddings}


	def _calc(self, e1_h, e2_h, e1_t, e2_t, r1, r2):
		return e1_h * e1_t * r1 + e2_h * e2_t * r1 + e1_h * e2_t * r2 - e2_h * e1_t * r2


	def loss_def(self, _lambda=.0):
		#Obtaining the initial configuration of the model
		#To get positive triples and negative triples for training
		#To get labels for the triples, positive triples as 1 and negative triples as -1
		#The shapes of h, t, r, y are (batch_size, 1 + negative_ent + negative_rel)
		h, t, r = self.get_all_instance()
		y = self.get_all_labels()
		#Embedding entities and relations of triples
		e1_h = tf.nn.embedding_lookup(self.ent1_embeddings, h)
		e2_h = tf.nn.embedding_lookup(self.ent2_embeddings, h)
		e1_t = tf.nn.embedding_lookup(self.ent1_embeddings, t)
		e2_t = tf.nn.embedding_lookup(self.ent2_embeddings, t)
		r1 = tf.nn.embedding_lookup(self.rel1_embeddings, r)
		r2 = tf.nn.embedding_lookup(self.rel2_embeddings, r)
		#Calculating score functions for all positive triples and negative triples
		res = tf.reduce_sum(self._calc(e1_h, e2_h, e1_t, e2_t, r1, r2), 1, keep_dims = False)
		loss_func = tf.reduce_mean(tf.nn.softplus(- y * res), 0, keep_dims = False)
		regul_func = tf.reduce_mean(e1_h ** 2) + tf.reduce_mean(e1_t ** 2) + tf.reduce_mean(e2_h ** 2) + tf.reduce_mean(e2_t ** 2) + tf.reduce_mean(r1 ** 2) + tf.reduce_mean(r2 ** 2)
		#Calculating loss to get what the framework will optimize
		self.loss =  loss_func + _lambda * regul_func


	def predict_def(self):
		h, t, r = self.get_predict_instance()
		h0 = tf.nn.embedding_lookup(self.ent1_embeddings, h)
		t0 = tf.nn.embedding_lookup(self.ent1_embeddings, t)
		r0 = tf.nn.embedding_lookup(self.rel1_embeddings, r)
		h1 = tf.nn.embedding_lookup(self.ent2_embeddings, h)
		t1 = tf.nn.embedding_lookup(self.ent2_embeddings, t)
		r1 = tf.nn.embedding_lookup(self.rel2_embeddings, r)
		self.predict = -tf.reduce_sum(self._calc(h0, h1, t0, t1, r0, r1), 1, keep_dims=True)


	def __init__(self, **config):
		self.entities = config['entTotal']
		self.relations = config['relTotal']
		self._lambda = config['lmbda']
		self.hiddensize = config['hidden_size']
		super().__init__(config['batch_size'], config['negatives'])
