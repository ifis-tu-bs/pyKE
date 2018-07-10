# coding:utf-8
from tensorflow import (get_variable as var,
                        reduce_mean as mean,
                        nn)

at, softplus = nn.embedding_lookup, nn.softplus
from .Base import ModelClass


def _lookup(h, t, l):
    ent = var('ent_embeddings')
    rel = var('rel_embeddings')

    return at(ent, h), at(ent, t), at(rel, l)  # [.,d]


def _term(h, t, l):
    return h * l * t


class DistMult(ModelClass):

    def _score(self, h, t, l):
        '''The term to score triples.'''

        return self._norm(_term(*_lookup(h, t, l)))  # [.]

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

        h, t, l = _lookup(*self.get_all_instance())  # [bp+bn,d]
        y = self.get_all_labels()  # [bp+bn]

        s = self._norm(_term(h, t, l))  # [bp+bn]
        loss = mean(softplus(y * s))  # []
        reg = mean(h ** 2) + mean(t ** 2) + mean(l ** 2)  # []

        return loss + self.weight * reg  # []

    def _predict_def(self):
        '''Initializes the prediction function.'''

        return self._score(*self.get_predict_instance())  # [b]

    def __init__(self, dimension, weight, baseshape, batchshape=None, \
                 optimizer=None):
        self.dimension = dimension,
        self.weight = weight
        super().__init__(baseshape, batchshape=batchshape, optimizer=optimizer)

    def __str__(self):
        return '{}-{}'.format(type(self).__name__, self.dimension)
