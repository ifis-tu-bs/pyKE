#coding:utf-8
from tensorflow import (get_variable as var,
                        reduce_sum as sum,
                        reduce_mean as mean,
                        maximum as max,
                        nn, sigmoid, ifft, fft, real, cast, conj, complex64)
from tensorflow.contrib.layers import xavier_initializer as xavier
at, norm = nn.embedding_lookup, nn.l2_normalize
from .Base import ModelClass


class HolE(ModelClass):


	def _embeddings(self, h, t, r):
		'''The term to embed triples.'''

		def transform(x):
			return fft(cast(at(self.ent_embeddings, x), complex64)) # [.,d]
		e = real(ifft(conj(transform(h)) * transform(t))) # [.,d]
		r = norm(at(self.rel_embeddings, r), 1) # [.,d]

		return r * e # [.,d]


	def embedding_def(self):
		'''Initializes the variables of the model.'''

		e, r, d = self.entities, self.relations, self.dimensions

		self.ent_embeddings = var('ent_embeddings', [e, d],
				initializer=xavier(uniform=False))
		self.rel_embeddings = var('rel_embeddings', [r, d],
				initializer=xavier(uniform=False))
		self.parameter_lists = {
				'ent_embeddings': self.ent_embeddings,
				'rel_embeddings': self.rel_embeddings}


	def loss_def(self):
		'''Initializes the loss function.'''

		def scores(h, t, r):
			e = self._embeddings(h, t, r) # [b,n,d]
			return mean(-sigmoid(sum(e, 1)), 1) # [b,n]

		p = scores(*self.get_positive_instance(in_batch=True)) # [b,n]
		n = scores(*self.get_negative_instance(in_batch=True)) # [b,n]

		self.loss = sum(max(p - n + self.margin, 0)) # []


	def predict_def(self):
		'''Initializes the prediction function.'''

		self.embed = self._embeddings(*self.get_predict_instance()) # [n,d]

		self.predict = sum(sigmoid(sum(self.embed, 1)), 1) # [n]


	def __init__(self, **config):
		self.entities = config['entTotal']
		self.relations = config['relTotal']
		self.dimensions = config['hidden_size']
		self.margin = config['margin']
		super().__init__(**config)
