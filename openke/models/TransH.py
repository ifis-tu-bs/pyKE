#coding:utf-8
from tensorflow import (get_variable as var,
                        reduce_sum as sum,
                        reduce_mean as mean,
                        maximum as max,
                        nn, reshape)
at = nn.embedding_lookup
from .Base import ModelClass


def _score(h, t, r):
	'''The term to score triples.'''

	n = at(var('normal_vectors'), r) # [.,d]
	def transfer(e):
		return e - sum(e * n, 1, keepdims=True) * n

	h = transfer(at(var('ent_embeddings'), h)) # [.,d]
	t = transfer(at(var('ent_embeddings'), t)) # [.,d]
	r = at(var('rel_embeddings'), r) # [.,d]

	return sum(abs(h + r - t), -1) # [.,d]


class TransH(ModelClass):


	def _embedding_def(self):
		'''Initializes the variables of the model.'''

		e, r, d = self.base[0], self.base[1], self.dimension[0]

		#Defining required parameters of the model, including embeddings of entities and relations, and normal vectors of planes
		ent = var("ent_embeddings", [e, d])
		rel = var("rel_embeddings", [r, d])
		nor = var("normal_vectors", [r, d])

		yield 'ent_embeddings', ent
		yield 'rel_embeddings', rel
		yield 'normal_vectors', nor

		self._entity = at(ent, self.predict_h)
		self._relation = at(rel, self.predict_l)


	def _loss_def(self):
		'''Initializes the loss function.'''

		def scores(h, t, r):
			s = _score(h, t, r) # [b,n]
			return mean(s, 1) # [b]

		p = scores(*self._positive_instance(in_batch=True)) # [b]
		n = scores(*self._negative_instance(in_batch=True)) # [b]

		return sum(max(p - n + self.margin, 0)) # []


	def _predict_def(self):
		'''Initializes the prediction function.'''

		return _score(*self._predict_instance()) # [b]


	def __init__(self, **config):
		self.dimension = config['hidden_size'],
		self.margin = config['margin']
		super().__init__(**config)


	def __str__(self):
		return '{}-{}'.format(type(self).__name__, self.dimension[0])
