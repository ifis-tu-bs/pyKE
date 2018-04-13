#coding:utf-8
from tensorflow import (get_variable as var,
                        reduce_sum as sum,
                        reduce_mean as mean,
                        maximum as max,
                        nn, sigmoid, ifft, fft, real, cast, conj, complex64)
at, norm = nn.embedding_lookup, nn.l2_normalize
from .Base import ModelClass


def _lookup(h, t, l):

	ent = var('ent_embeddings')
	rel = var('rel_embeddings')

	return at(ent, h), at(ent, t), at(rel, l)


def _term(h, t, l):

	def transform(x):
		return fft(cast(x, complex64)) # [.,d]

	return norm(r, 1) * real(ifft(conj(transform(h)) * transform(t)))


class HolE(ModelClass):


	def _score(self, h, t, l):
		'''The term to embed triples.'''

		return sigmoid(self._norm(_term(*_lookup(h, t, l))))


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
			return mean(-s, 1) # [b]

		p = scores(*self._positive_instance(in_batch=True)) # [b]
		n = scores(*self._negative_instance(in_batch=True)) # [b]

		return sum(max(p - n + self.margin, 0)) # []


	def _predict_def(self):
		'''Initializes the prediction function.'''

		return _score(*self._predict_instance()) # [b]


	def __init__(self, dimension, margin, baseshape, batchshape, optimizer=None):
		self.dimension = dimension,
		self.margin = margin
		super().__init__(baseshape, batchshape, optimizer=optimizer)


	def __str__(self):
		return '{}-{}'.format(type(self).__name__, self.dimension[0])
