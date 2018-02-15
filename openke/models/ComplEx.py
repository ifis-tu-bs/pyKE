#coding:utf-8
from tensorflow import (get_variable as var,
                        reduce_sum as sum,
                        reduce_mean as mean,
                        maximum as max,
                        nn)
from tensorflow.contrib.layers import xavier_initializer as xavier
at, softplus = nn.embedding_lookup, nn.softplus
from . import Model


class ComplEx(Model):


	def _lookup(self, h, t, r):
		'''Gets the variables concerning a fact.'''
		assert h.shape == t.shape == r.shape
		hre = at(self.real_entity_embeddings, h)
		tre = at(self.real_entity_embeddings, t)
		rre = at(self.real_relation_embeddings, r)
		him = at(self.imaginary_entity_embeddings, h)
		tim = at(self.imaginary_entity_embeddings, t)
		rim = at(self.imaginary_relation_embeddings, r)
		return hre, him, tre, tim, rre, rim


	def _term(self, hre, him, tre, tim, rre, rim):
		'''Returns the real part of the ComplEx embedding of a fact described by
three complex numbers.'''
		return hre * tre * rre + him * tim * rre + hre * tim * rim - him * tre * rim


	def _embeddings(self, h, t, r):
		'''The term to embed triples.'''
		return self._term(*self._lookup(h, t, r))


	def embedding_def(self):
		'''Initializes the variables.'''

		E, R, D = self.entities, self.relations, self.hiddensize

		self.real_entity_embeddings = var("ent_re_embeddings", [E, D],
				initializer=xavier(uniform=True))
		self.real_relation_embeddings = var("rel_re_embeddings", [R, D],
				initializer=xavier(uniform=True))
		self.imaginary_entity_embeddings = var("ent_im_embeddings", [E, D],
				initializer=xavier(uniform=True))
		self.imaginary_relation_embeddings = var("rel_im_embeddings", [R, D],
				initializer=xavier(uniform=True))
		self.parameter_lists = {
				"ent_re_embeddings": self.real_entity_embeddings,
				"ent_im_embeddings": self.imaginary_entity_embeddings,
				"rel_re_embeddings": self.real_relation_embeddings,
				"rel_im_embeddings": self.imaginary_relation_embeddings}


	def loss_def(self):
		'''Initializes the loss function.'''

		hre, him, tre, tim, rre, rim = self._lookup(*self.get_all_instance())
		y = self.get_all_labels()

		e = self._term(hre, him, tre, tim, rre, rim)
		res = sum(e, 1)
		loss_func = mean(softplus(- y * res), 0)
		regul_func = mean(hre ** 2) + mean(tre ** 2) + mean(him ** 2) + mean(tim ** 2) + mean(rre ** 2) + mean(rim ** 2)

		self.loss =  loss_func + self._lambda * regul_func


	def predict_def(self):
		'''Initializes the prediction function.'''

		self.embed = self._embeddings(*self.get_predict_instance())

		self.predict = -sum(self.embed, 1, keep_dims=True)


	def __init__(self, **config):
		self.entities = config['entTotal']
		self.relations = config['relTotal']
		self._lambda = config['lmbda']
		self.hiddensize = config['hidden_size']
		super().__init__(config['batch_size'], config['negatives'])
