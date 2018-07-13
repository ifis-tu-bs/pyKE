# coding:utf-8

import tensorflow as tf

from pyke.models import BaseModel

_at = tf.nn.embedding_lookup
_norm = tf.nn.l2_normalize


def _lookup(h, t, l):
    ent = tf.get_variable('ent_embeddings')
    rel = tf.get_variable('rel_embeddings')

    return _at(ent, h), _at(ent, t), _at(rel, l)


def _term(h, t, l):
    def transform(x):
        return tf.fft(tf.cast(x, tf.complex64))  # [.,d]

    return _norm(l, 1) * tf.real(tf.ifft(tf.conj(transform(h)) * transform(t)))


class HolE(BaseModel):
    def __init__(self, dimension, margin, **kwargs):
        self.dimension = dimension
        self.margin = margin
        super().__init__(**kwargs)

    def __str__(self):
        return '{}-{}'.format(type(self).__name__, self.dimension)

    def _score(self, h, t, l):
        """The term to embed triples."""
        return tf.sigmoid(self._norm(_term(*_lookup(h, t, l))))

    def _embedding_def(self):
        """Initializes the variables of the model."""
        ent = tf.get_variable('ent_embeddings', [self.ent_count, self.dimension])
        rel = tf.get_variable('rel_embeddings', [self.rel_count, self.dimension])

        yield 'ent_embeddings', ent
        yield 'rel_embeddings', rel

        self._entity = _at(ent, self.predict_h)
        self._relation = _at(rel, self.predict_l)

    def _loss_def(self):
        """Initializes the loss function."""

        def scores(h, t, r):
            s = self._score(h, t, r)  # [b,n]
            return tf.reduce_mean(-s, 1)  # [b]

        p = scores(*self.get_positive_instance(in_batch=True))  # [b]
        n = scores(*self.get_negative_instance(in_batch=True))  # [b]

        return tf.reduce_sum(tf.maximum(p - n + self.margin, 0))  # []

    def _predict_def(self):
        """Initializes the prediction function."""

        return self._score(*self.get_predict_instance())  # [b]
