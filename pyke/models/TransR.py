# coding:utf-8
import tensorflow as tf

from pyke.models import BaseModel

_at = tf.nn.embedding_lookup


def _lookup(h, t, l):
    ent = tf.get_variable('ent_embeddings')
    mat = tf.get_variable('transfer_matrix')
    rel = tf.get_variable('rel_embeddings')

    return _at(ent, h), _at(ent, t), _at(mat, l), _at(rel, l)


def _term(h, t, m, l):
    return tf.squeeze(tf.matmul(m, h) + l - tf.matmul(m, t), [-1])  # [.]


class TransR(BaseModel):
    def __init__(self, edimension, rdimension, margin, **kwargs):
        self.dimension = edimension, rdimension
        self.margin = margin
        super().__init__(**kwargs)

    def __str__(self):
        return '{}-{}-{}'.format(type(self).__name__, self.dimension[0], self.dimension[1])

    def _score(self, h, t, l):
        """The term to score triples."""

        return self._norm(_term(*_lookup(h, t, l)))

    def _embedding_def(self):
        """Initializes the variables of the model."""
        ent = tf.get_variable('ent_embeddings', [self.ent_count, self.dimension[0], 1])
        rel = tf.get_variable('rel_embeddings', [self.rel_count, self.dimension[1], 1])
        mat = tf.get_variable('transfer_matrix', [self.rel_count, self.dimension[1], self.dimension[0]])

        yield 'ent_embeddings', ent
        yield 'rel_embeddings', rel
        yield 'transfer_matrix', mat

        self._entity = _at(ent, self.predict_h)
        self._relation = _at(rel, self.predict_l)

    def _loss_def(self):
        """Initializes the loss function."""

        def scores(h, t, r):
            s = self._score(h, t, r)  # [b,n,k]
            return tf.reduce_mean(s, 1)  # [b]

        p = scores(*self.get_positive_instance(in_batch=True))  # [b]
        n = scores(*self.get_negative_instance(in_batch=True))  # [b]

        return tf.reduce_sum(tf.maximum(p - n + self.margin, 0))  # []

    def _predict_def(self):
        """Initializes the prediction function."""

        return self._score(*self.get_predict_instance())  # [b]
