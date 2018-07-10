# coding:utf-8
from tensorflow import (get_variable as var,
                        reduce_mean as mean,
                        nn)

at, softplus = nn.embedding_lookup, nn.softplus
from .Base import ModelClass


def _term(hRe, hIm, tRe, tIm, rRe, rIm):
    '''Returns the real part of the ComplEx embedding of a fact described by
three complex numbers.'''

    return hRe * tRe * rRe + hIm * tIm * rRe + hRe * tIm * rIm - hIm * tRe * rIm


def _lookup(h, t, r):
    '''Gets the variables concerning a fact.'''

    eRe = var('ent_re_embeddings')
    rRe = var('rel_re_embeddings')
    eIm = var('ent_im_embeddings')
    rIm = var('rel_im_embeddings')

    return at(eRe, h), at(eRe, t), at(rRe, r), at(eIm, h), at(eIm, t), at(rIm, r)


class ComplEx(ModelClass):

    def _score(self, h, t, r):
        '''The term to embed triples.'''

        return self._norm(_term(*_lookup(h, t, r)))  # [.]

    def _embedding_def(self):
        '''Initializes the variables.'''

        e, r, d = self.base[0], self.base[1], self.dimension[0]

        eRe = var("ent_re_embeddings", [e, d])
        eIm = var("ent_im_embeddings", [e, d])
        rRe = var("rel_re_embeddings", [r, d])
        rIm = var("rel_im_embeddings", [r, d])

        yield 'ent_re_embeddings', eRe
        yield 'ent_im_embeddings', eIm
        yield 'rel_re_embeddings', rRe
        yield 'rel_im_embeddings', rIm

        self._entity = at(eRe, self.predict_h)
        self._relation = at(rRe, self.predict_l)

    def _loss_def(self):
        '''Initializes the loss function.'''

        hRe, hIm, tRe, tIm, lRe, lIm = _lookup(*self.get_all_instance())  # [b,d]
        y = self.get_all_labels()  # [b]

        s = self._norm(_term(hRe, hIm, tRe, tIm, lRe, lIm))  # [b]
        loss = mean(softplus(- y * s), 0)  # []
        reg = mean(hRe ** 2) + mean(hIm ** 2) + mean(tRe ** 2) + mean(tIm ** 2) + mean(lRe ** 2) + mean(lIm ** 2)

        return loss + self.weight * reg  # []

    def _predict_def(self):
        '''Initializes the prediction function.'''

        return self._score(*self.get_predict_instance())

    def __init__(self, dimension, weight, baseshape, batchshape=None, \
                 optimizer=None):
        self.dimension = dimension,
        self.weight = weight
        super().__init__(baseshape, batchshape=batchshape, optimizer=optimizer)

    def __str__(self):
        return '{}-{}'.format(type(self).__name__, self.dimension)
