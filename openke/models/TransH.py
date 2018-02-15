#coding:utf-8
from tensorflow import (get_variable as var,
                        reduce_sum as sum,
                        reduce_mean as mean,
                        maximum as max,
                        nn, reshape)
from tensorflow.contrib.layers import xavier_initializer as xavier
at = nn.embedding_lookup
from . import Model


class TransH(Model):


	def embedding_def(self):
		'''Initializes the variables of the model.'''

		E, R, D = self.entities, self.relations, self.hiddensize

		#Defining required parameters of the model, including embeddings of entities and relations, and normal vectors of planes
		self.ent_embeddings = var("ent_embeddings", [E, D],
				initializer=xavier(uniform=False))
		self.rel_embeddings = var("rel_embeddings", [R, D],
				initializer=xavier(uniform=False))
		self.normal_vectors = var("normal_vectors", [R, D],
				initializer=xavier(uniform=False))

		self.parameter_lists = {
				"ent_embeddings": self.ent_embeddings,
				"rel_embeddings": self.rel_embeddings,
				"normal_vectors": self.normal_vectors}


	def _embeddings(self, h, t, r):
		'''The term to embed triples.'''

		n = at(self.normal_vectors, r) # [.,D]
		def transfer(e):
			return e - sum(e * n, 1, keep_dims=True) * n

		h = transfer(at(self.ent_embeddings, h)) # [.,D]
		t = transfer(at(self.ent_embeddings, t)) # [.,D]
		r = at(self.rel_embeddings, r) # [.,D]

		return h + r - t # [.,D]


	def loss_def(self):
		'''Initializes the loss function.'''

		def scores(h, t, r):
			e = self._embeddings(h, t, r) # [B,N,D]
			return sum(mean(abs(e), 1), 1, keep_dims=True) # [B]

		p = scores(*self.get_positive_instance(in_batch=True)) # [B]
		n = scores(*self.get_negative_instance(in_batch=True)) # [B]

		self.loss = sum(max(p - n + self.margin, 0)) # []


	def predict_def(self):
		'''Initializes the prediction function.'''

		s = self._embeddings(*self.get_predict_instance()) # [B,D]

		self.predict = sum(abs(s), 1) # [B]


	def __init__(self, **config):
		self.entities = config['entTotal']
		self.relations = config['relTotal']
		self.hiddensize = config['hidden_size']
		self.margin = config['margin']
		super().__init__(**config)
