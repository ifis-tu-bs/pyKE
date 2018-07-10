# coding:utf-8
from tensorflow import (get_variable as var,
                        reduce_sum as sum,
                        reduce_mean as mean,
                        maximum as max,
                        matmul, squeeze, nn)

at = nn.embedding_lookup
from .Base import ModelClass


def _lookup(h, t, l):
    ent = var('ent_embeddings')
    mat = var('transfer_matrix')
    rel = var('rel_embeddings')

    return at(ent, h), at(ent, t), at(mat, l), at(rel, l)


def _term(h, t, m, l):
    return squeeze(matmul(m, h) + l - matmul(m, t), [-1])  # [.]


class TransR(ModelClass):

    def _score(self, h, t, l):
        '''The term to score triples.'''

        return self._norm(_term(*_lookup(h, t, l)))

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
            s = self._score(h, t, r)  # [b,n,k]
            return mean(s, 1)  # [b]

        p = scores(*self.get_positive_instance(in_batch=True))  # [b]
        n = scores(*self.get_negative_instance(in_batch=True))  # [b]

        return sum(max(p - n + self.margin, 0))  # []

    def _predict_def(self):
        '''Initializes the prediction function.'''

        return self._score(*self.get_predict_instance())  # [b]

    def __init__(self, edimension, rdimension, margin, baseshape, batchshape=None, \
                 optimizer=None):
        self.dimension = edimension, rdimension
        self.margin = margin
        super().__init__(baseshape, batchshape=batchshape, optimizer=optimizer)

    def __str__(self):
        return '{}-{}-{}'.format(type(self).__name__, self.dimension[0], \
                                 self.dimension[1])
