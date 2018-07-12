# coding:utf-8
from tensorflow import (get_variable as var,
                        reduce_sum as sum,
                        reduce_mean as mean,
                        maximum as max,
                        squeeze, matmul, nn)

at = nn.embedding_lookup
from .base import BaseModel


# FIXME replace matmul with PEP465 @-operator when upgrading to Python 3.5


def _lookup(h, t, l):
    ent = var('ent_embeddings')
    rel = var('rel_matrices')

    return at(ent, h), at(ent, t), at(rel, l)


def _term(h, t, m):
    return squeeze(h * matmul(m, t), [-1])


class RESCAL(BaseModel):

    def _score(h, t, l):
        '''The term to score triples.'''

        return self._norm(_term(*_lookup(h, t, l)))  # [.]

    def _embedding_def(self):
        '''Initializes the variables of the model.'''

        e, r, d = self.base[0], self.base[1], self.dimension[0]

        ent = var('ent_embeddings', [e, d, 1])
        rel = var('rel_matrices', [r, d, d])

        yield 'ent_embeddings', ent
        yield 'rel_matrices', rel

        self._entity = squeeze(at(ent, self.predict_h), [-1])

    def _loss_def(self):
        '''Initializes the loss function.'''

        def scores(h, t, l):
            s = self._score(h, t, l)  # [b,n]
            return mean(s, 1)  # [b]

        p = scores(*self.get_positive_instance(in_batch=True))  # [b]
        n = scores(*self.get_negative_instance(in_batch=True))  # [b]

        return sum(max(p - n + self.margin, 0))  # []

    def _predict_def(self):
        '''Initializes the prediction function.'''

        return self._score(*self.get_predict_instance())  # [b]

    def __init__(self, dimension, margin, baseshape, batchshape=None, \
                 optimizer=None):
        self.dimension = dimension,
        self.margin = margin
        super().__init__(baseshape, batchshape=batchshape, optimizer=optimizer)

    def __str__(self):
        return '{}-{}'.format(type(self).__name__, self.dimension[0])
