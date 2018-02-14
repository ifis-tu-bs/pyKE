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


	def _scores(self, h, t, r):

		n = at(self.normal_vectors, r)

		def transfer(e):
			return e - sum(e * n, 1, keep_dims=True) * n

		h = transfer(at(self.ent_embeddings, h))
		t = transfer(at(self.ent_embeddings, t))
		r = at(self.rel_embeddings, r)

		return abs(h + r - t)


	def loss_def(self):

		B = self.batchsize
		def fit(x):
			return reshape(x, [B, -1])

		def score(h, t, r):
			s = self._scores(fit(h), fit(t), fit(r))
			return sum(mean(s, 1, keep_dims=False), 1, keep_dims=True)

		p = score(*self.get_positive_instance(in_batch=False))
		n = score(*self.get_negative_instance(in_batch=False))

		self.loss = sum(max(p - n + self.margin, 0))


	def predict_def(self):
		s = self._scores(*self.get_predict_instance())
		self.predict = sum(s, 1, keep_dims=True)


	def __init__(self, **config):
		self.entities = config['entTotal']
		self.relations = config['relTotal']
		self.margin = config['margin']
		self.hiddensize = config['hidden_size']
		super().__init__(**config)
