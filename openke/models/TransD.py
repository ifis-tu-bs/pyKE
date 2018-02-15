#coding:utf-8
from tensorflow import (get_variable as var,
                        reduce_sum as sum,
                        reduce_mean as mean,
                        maximum as max,
                        nn)
from tensorflow.contrib.layers import xavier_initializer as xavier
at = nn.embedding_lookup
from . import Model


class TransD(Model):


	def _embeddings(self, h, t, r):
		'''The term to embed triples.'''

		he = at(self.ent_embeddings, h) # [.,D]
		te = at(self.ent_embeddings, t) # [.,D]
		re = at(self.rel_embeddings, r) # [.,D]
		ht = at(self.ent_transfer, h) # [.,D]
		tt = at(self.ent_transfer, t) # [.,D]
		rt = at(self.rel_transfer, r) # [.,D]

		def transfer(e, t):
			return e + sum(e * t, 1, keep_dims=True) * rt
		h = transfer(he, ht) # [.,D]
		t = transfer(te, tt) # [.,D]

		return h + re - t


	def embedding_def(self):

		E, R, D = self.entities, self.relations, self.hiddensize

		self.ent_embeddings = var("ent_embeddings", [E, D],
				initializer=xavier(uniform=False))
		self.rel_embeddings = var("rel_embeddings", [R, D],
				initializer=xavier(uniform=False))
		self.ent_transfer = var("ent_transfer", [E, D],
				initializer=xavier(uniform=False))
		self.rel_transfer = var("rel_transfer", [R, D],
				initializer=xavier(uniform=False))
		self.parameter_lists = {
				"ent_embeddings": self.ent_embeddings,
				"rel_embeddings": self.rel_embeddings,
				"ent_transfer": self.ent_transfer,
				"rel_transfer": self.rel_transfer}


	def loss_def(self):
		'''Initializes the loss function.'''

		def scores(h, t, r):
			e = self._embeddings(h, t, r) # [B,N,D]
			return sum(mean(abs(p), 1), 1, keep_dims=True) # [B]

		p = scores(*self.get_positive_instance(in_batch=True)) # [B]
		n = scores(*self.get_negative_instance(in_batch=True)) # [B]

		self.loss = sum(max(p - n + self.margin, 0)) # []


	def predict_def(self):
		'''Initializes the prediction function.'''

		self.embed = self._embeddings(*self.get_predict_instance()) # [B,D]

		self.predict = sum(abs(self.embed), 1, keep_dims=True) # [B]


	def __init__(self, **config):
		self.entities = config['entTotal']
		self.relations = config['relTotal']
		self.hiddensize = config['hidden_size']
		self.margin = config['margin']
		super().__init__(**config)
