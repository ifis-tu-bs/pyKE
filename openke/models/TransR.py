#coding:utf-8
from tensorflow import (get_variable as var,
                        reduce_sum as sum,
                        reduce_mean as mean,
                        maximum as max,
                        matmul, squeeze, nn, rank)
at = nn.embedding_lookup
from .Base import ModelClass


def _score(h, t, r):
	'''The term to score triples.'''

	ent = var('ent_embeddings')
	m = at(var('transfer_matrix'), r) # [.,k,d]
	r = at(var('rel_embeddings'), r) # [.,k,1]
	h = matmul(m, at(ent, h)) # [.,k,1]
	t = matmul(m, at(ent, t)) # [.,k,1]

	return sum(squeeze(h + r - t, [-1]), -1) # [.]


class TransR(ModelClass):


	def _embedding_def(self):
		'''Initializes the variables of the model.'''

		e, r = self.base
		d, k = self.dimension

		ent = var('ent_embeddings', [e, d, 1])
		rel = var('rel_embeddings', [r, k, 1])
		mat = var('transfer_matrix', [r, k, d])

		yield 'ent_embeddings', ent
		yield 'rel_embeddings', rel
		yield 'transfer_matrix', mat

		self._entity = at(ent, self.predict_h)
		self._relation = at(rel, self.predict_l)


	def _loss_def(self):
		'''Initializes the loss function.'''

		def scores(h, t, r):
			s = _score(h, t, r) # [b,n,k]
			return mean(s, 1) # [b]

		p = scores(*self._positive_instance(in_batch=True)) # [b]
		n = scores(*self._negative_instance(in_batch=True)) # [b]

		return sum(max(p - n + self.margin, 0)) # []


	def _predict_def(self):
		'''Initializes the prediction function.'''

		return _score(*self._predict_instance()) # [b]


	def __init__(self, edimension, rdimension, margin, baseshape, batchshape,\
			optimizer=None):
		self.dimension = edimension, rdimension
		self.margin = margin
		super().__init__(baseshape, batchshape, optimizer=optimizer)


	def __str__(self):
		return '{}-{}-{}'.format(type(self).__name__, self.dimension[0],\
				self.dimension[1])
