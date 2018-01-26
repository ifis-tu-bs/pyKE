#coding:utf-8
from tensorflow import (get_variable as var,
                        reduce_sum as sum,
                        reduce_mean as mean,
                        maximum as max,
                        nn)
from tensorflow.contrib.layers import xavier_initializer as xavier
at = nn.embedding_lookup
from . import Model

class TransH(Model):


	def _transfer(self, e, n):
		return e - sum(e * n, 1, keep_dims=True) * n


	def _calc(self, h, t, r):
		return abs(h + r - t)


	def embedding_def(self):

		cE, cR, d = self.entities, self.relations, self.hiddensize

		#Defining required parameters of the model, including embeddings of entities and relations, and normal vectors of planes
		self.ent_embeddings = var("ent_embeddings", [cE, d],
				initializer=xavier(uniform=False))
		self.rel_embeddings = var("rel_embeddings", [cR, d],
				initializer=xavier(uniform=False))
		self.normal_vectors = var("normal_vectors", [cR, d],
				initializer=xavier(uniform=False))

		self.parameter_lists = {
				"ent_embeddings":self.ent_embeddings,
				"rel_embeddings":self.rel_embeddings,
				"normal_vectors":self.normal_vectors}


	def loss_def(self):
		#To get positive triples and negative triples for training
		#The shapes of pos_h, pos_t, pos_r are (batch_size)
		#The shapes of neg_h, neg_t, neg_r are ((negative_ent + negative_rel) × batch_size)
		pos_h, pos_t, pos_r = self.get_positive_instance(in_batch = False)
		neg_h, neg_t, neg_r = self.get_negative_instance(in_batch = False)
		#Embedding entities and relations of triples, e.g. pos_h_e, pos_t_e and pos_r_e are embeddings for positive triples
		pos_h_e = tf.nn.embedding_lookup(self.ent_embeddings, pos_h)
		pos_t_e = tf.nn.embedding_lookup(self.ent_embeddings, pos_t)
		pos_r_e = tf.nn.embedding_lookup(self.rel_embeddings, pos_r)
		neg_h_e = tf.nn.embedding_lookup(self.ent_embeddings, neg_h)
		neg_t_e = tf.nn.embedding_lookup(self.ent_embeddings, neg_t)
		neg_r_e = tf.nn.embedding_lookup(self.rel_embeddings, neg_r)
		#Getting the required normal vectors of planes to transfer entity embeddings
		pos_norm = tf.nn.embedding_lookup(self.normal_vectors, pos_r)
		neg_norm = tf.nn.embedding_lookup(self.normal_vectors, neg_r)
		#Calculating score functions for all positive triples and negative triples
		p_h = self._transfer(pos_h_e, pos_norm)
		p_t = self._transfer(pos_t_e, pos_norm)
		p_r = pos_r_e
		n_h = self._transfer(neg_h_e, neg_norm)
		n_t = self._transfer(neg_t_e, neg_norm)
		n_r = neg_r_e
		#Calculating score functions for all positive triples and negative triples
		#The shape of _p_score is (1, batch_size, hidden_size)
		#The shape of _n_score is (negative_ent + negative_rel, batch_size, hidden_size)
		_p_score = self._calc(p_h, p_t, p_r)
		_n_score = self._calc(n_h, n_t, n_r)
		#The shape of p_score is (batch_size, 1)
		#The shape of n_score is (batch_size, 1)
		p_score =  sum(mean(_p_score, 0, keep_dims=False), 1, keep_dims=True)
		n_score =  sum(mean(_n_score, 0, keep_dims=False), 1, keep_dims=True)
		#Calculating loss to get what the framework will optimize
		self.loss = sum(max(p_score - n_score + self.margin, 0))

	def predict_def(self):
		predict_h, predict_t, predict_r = self.get_predict_instance()
		predict_h_e = tf.nn.embedding_lookup(self.ent_embeddings, predict_h)
		predict_t_e = tf.nn.embedding_lookup(self.ent_embeddings, predict_t)
		predict_r_e = tf.nn.embedding_lookup(self.rel_embeddings, predict_r)
		predict_norm = tf.nn.embedding_lookup(self.normal_vectors, predict_r)
		h_e = self._transfer(predict_h_e, predict_norm)
		t_e = self._transfer(predict_t_e, predict_norm)
		r_e = predict_r_e
		self.predict = tf.reduce_sum(self._calc(h_e, t_e, r_e), 1, keep_dims = True)


	def __init__(self, config):
		super().__init__(config)
		self.entities = config.entTotal
		self.relations = config.relTotal
		self.margin = config.margin
		self.hiddensize
		self.entitysize
		self.relationsize
