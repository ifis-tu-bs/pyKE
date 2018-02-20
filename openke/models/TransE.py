#coding:utf-8
from tensorflow import (get_variable as var,
                        reduce_sum as sum,
                        reduce_mean as mean,
                        maximum as max,
                        nn)
from tensorflow.contrib.layers import xavier_initializer as xavier
at = nn.embedding_lookup
from .Base import ModelClass


class TransE(ModelClass):


	def _embeddings(self, h, t, r):
		'''The term to embed triples.'''

		h = at(self.ent_embeddings, h) # [.,d]
		t = at(self.ent_embeddings, t) # [.,d]
		r = at(self.rel_embeddings, r) # [.,d]

		return h - t + r # [.,d]


	def embedding_def(self):
		'''Initializes the variables of the model.'''

		e, r, d = self.entities, self.relations, self.dimensions

		self.ent_embeddings = var("ent_embeddings", [e, d],
				initializer=xavier(uniform=False))
		self.rel_embeddings = var("rel_embeddings", [r, d],
				initializer=xavier(uniform=False))
		self.parameter_lists = {
				"ent_embeddings": self.ent_embeddings,
				"rel_embeddings": self.rel_embeddings}


	def loss_def(self):
		'''Initializes the loss function.'''

		def scores(h, t, r):
			e = self._embeddings(h, t, r) # [b,n,d]
			return sum(mean(abs(e), 1), 1, keep_dims=True) # [b]

		p = scores(*self.get_positive_instance(in_batch=True)) # [b]
		n = scores(*self.get_negative_instance(in_batch=True)) # [b]

		self.loss = sum(max(p - n + self.margin, 0)) # []


	def predict_def(self):
		'''Initializes the prediction function.'''

		self.embed = self._embeddings(*self.get_predict_instance()) # [b,d]

		self.predict = mean(abs(self.embed), 1) # [b]


	def __init__(self, **config):
		self.entities = config['entTotal']
		self.relations = config['relTotal']
		self.dimensions = config['hidden_size']
		self.margin = config['margin']
		super().__init__(**config)
