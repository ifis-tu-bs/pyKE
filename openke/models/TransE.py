#coding:utf-8
from tensorflow import (get_variable as var,
                        reduce_sum as sum,
                        reduce_mean as mean,
                        maximum as max,
                        nn)
at = nn.embedding_lookup
from .Base import ModelClass


def _score(h, t, r):
	'''The term to score triples.'''

	h = at(var('ent_embeddings'), h) # [.,d]
	t = at(var('ent_embeddings'), t) # [.,d]
	r = at(var('rel_embeddings'), r) # [.,d]

	return sum(abs(h - t + r), -1) # [.]


class TransE(ModelClass):


	def _embedding_def(self):
		'''Initializes the variables of the model.'''

		e, r, d = self.base[0], self.base[1], self.dimension[0]

		ent = var('ent_embeddings', [e, d])
		rel = var('rel_embeddings', [r, d])

		yield 'ent_embeddings', ent
		yield 'rel_embeddings', rel

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
