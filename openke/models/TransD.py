#coding:utf-8
from tensorflow import (get_variable as var,
                        reduce_sum as sum,
                        reduce_mean as mean,
                        maximum as max,
                        nn)
from tensorflow.contrib.layers import xavier_initializer as xavier
at = nn.embedding_lookup
from .Base import ModelClass


class TransD(ModelClass):


	def _embeddings(self, h, t, r):
		'''The term to embed triples.'''

		he = at(self.ent_embeddings, h) # [.,d]
		te = at(self.ent_embeddings, t) # [.,d]
		re = at(self.rel_embeddings, r) # [.,d]
		ht = at(self.ent_transfer, h) # [.,d]
		tt = at(self.ent_transfer, t) # [.,d]
		rt = at(self.rel_transfer, r) # [.,d]

		def transfer(e, t):
			return e + sum(e * t, 1, keep_dims=True) * rt
		h = transfer(he, ht) # [.,d]
		t = transfer(te, tt) # [.,d]

		return h + re - t # [.,d]


	def embedding_def(self):
		'''Initializes the variables of the model.'''

		e, r, d = self.entities, self.relations, self.dimensions

		self.ent_embeddings = var("ent_embeddings", [e, d],
				initializer=xavier(uniform=False))
		self.rel_embeddings = var("rel_embeddings", [r, d],
				initializer=xavier(uniform=False))
		self.ent_transfer = var("ent_transfer", [e, d],
				initializer=xavier(uniform=False))
		self.rel_transfer = var("rel_transfer", [r, d],
				initializer=xavier(uniform=False))
		self.parameter_lists = {
				"ent_embeddings": self.ent_embeddings,
				"rel_embeddings": self.rel_embeddings,
				"ent_transfer": self.ent_transfer,
				"rel_transfer": self.rel_transfer}


	def loss_def(self):
		'''Initializes the loss function.'''

		def scores(h, t, r):
			e = self._embeddings(h, t, r) # [b,n,d]
			return mean(sum(abs(e), 2), 1) # [b]

		p = scores(*self.get_positive_instance(in_batch=True)) # [b]
		n = scores(*self.get_negative_instance(in_batch=True)) # [b]

		self.loss = sum(max(p - n + self.margin, 0)) # []


	def predict_def(self):
		'''Initializes the prediction function.'''

		self.embed = self._embeddings(*self.get_predict_instance()) # [b,d]

		self.predict = sum(abs(self.embed), 1, keep_dims=True) # [b]


	def __init__(self, **config):
		self.entities = config['entTotal']
		self.relations = config['relTotal']
		self.dimensions = config['hidden_size']
		self.margin = config['margin']
		super().__init__(**config)
