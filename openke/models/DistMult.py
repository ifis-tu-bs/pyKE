#coding:utf-8
from tensorflow import (get_variable as var,
                        reduce_sum as sum,
                        reduce_mean as mean,
                        nn)
from tensorflow.contrib.layers import xavier_initializer as xavier
at, softplus = nn.embedding_lookup, nn.softplus
from .Base import ModelClass


class DistMult(ModelClass):


	def _lookup(self, h, t, r):

		h = at(self.ent_embeddings, h) # [.,d]
		t = at(self.ent_embeddings, t) # [.,d]
		r = at(self.rel_embeddings, r) # [.,d]
		return h, t, r # [.,d]


	def _embeddings(self, h, t, r):
		'''The term to embed triples.'''

		h, t, r = _lookup(h, t, r) # [.,d]
		return h * r * t # [.,d]


	def embedding_def(self):
		'''Initializes the variables of the model.'''

		e, r, d = self.entities, self.relations, self.dimensions

		self.ent_embeddings = var('ent_embeddings', [e, d],
				initializer=xavier(uniform=True))
		self.rel_embeddings = var('rel_embeddings', [r, d],
				initializer=xavier(uniform=True))
		self.parameter_lists = {
				'ent_embeddings': self.ent_embeddings,
				'rel_embeddings': self.rel_embeddings}


	def loss_def(self):
		'''Initializes the loss function.'''

		h, t, r = self._lookup(*self.get_all_instance()) # [bp+bn,d]
		y = self.get_all_labels() # [bp+bn]

		res = sum(h * r * t, 1) # [bp+bn]
		loss_func = mean(softplus(-y * res)) # []
		regul_func = mean(h ** 2) + mean(t ** 2) + mean(r ** 2) # []

		self.loss =  loss_func + self._lambda * regul_func # []


	def predict_def(self):
		'''Initializes the prediction function.'''

		self.embed = self._embeddings(*self.get_predict_instance()) # [n,d]

		self.predict = -sum(self.embed, 1) # [n]


	def __init__(self, **config):
		self.entities = config['entTotal']
		self.relations = config['relTotal']
		self._lambda = config['lmbda']
		self.dimensions = config['hidden_size']
		super().__init__(config['batch_size'], config['negatives'])
