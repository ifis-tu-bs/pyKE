# coding:utf-8
from tensorflow import (get_variable as var,
                        reduce_sum as sum,
                        reduce_mean as mean,
                        maximum as max,
                        nn)

at = nn.embedding_lookup
from .base import BaseModel


def _score(h, t, l):
    '''The term to embed triples.'''

    ent = var('ent_embeddings')
    rel = var('rel_embeddings')
    etr = var('ent_transfer')
    rtr = var('rel_transfer')

    return at(ent, h), at(etr, h), at(ent, t), at(etr, t), at(rel, l), at(rtr, l)


def _term(he, ht, te, tt, le, lt):
    def transfer(e, t):
        return e + sum(e * t, -1, keepdims=True) * lt

    return transfer(he, ht) + le - transfer(te, tt)


class TransD(BaseModel):

    def _score(self, h, t, l):
        return self._norm(_term(*_lookup(h, t, l)))

    def _embedding_def(self):
        '''Initializes the variables of the model.'''

        e, r, d = self.base[0], self.base[1], self.dimension[0]

        ent = var("ent_embeddings", [e, d])
        rel = var("rel_embeddings", [r, d])
        etr = var("ent_transfer", [e, d])
        rtr = var("rel_transfer", [r, d])

        yield 'ent_embeddings', ent
        yield 'rel_embeddings', rel
        yield 'ent_transfer', etr
        yield 'rel_transfer', rtr

        self._entity = at(ent, self.predict_h)
        self._relation = at(rel, self.predict_l)

    def _loss_def(self):
        '''Initializes the loss function.'''

        def scores(h, t, l):
            s = _score(h, t, l)  # [b,n]
            return mean(s, 1)  # [b]

        p = scores(*self.get_positive_instance(in_batch=True))  # [b]
        n = scores(*self.get_negative_instance(in_batch=True))  # [b]

        return sum(max(p - n + self.margin, 0))  # []

    def _predict_def(self):
        '''Initializes the prediction function.'''

        return _score(*self.get_predict_instance())  # [b]

    def __init__(self, dimension, margin, baseshape, batchshape=None, \
                 optimizer=None):
        self.dimension = dimension,
        self.margin = margin
        super().__init__(baseshape, batchshape=batchshape, optimizer=optimizer)

    def __str__(self):
        return '{}-{}'.format(type(self).__name__, self.dimension[0])
