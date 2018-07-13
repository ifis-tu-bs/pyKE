# coding:utf-8
import tensorflow as tf

from pyke.models import BaseModel

_at = tf.nn.embedding_lookup


def _lookup(h, t, l):
    ent = tf.get_variable('ent_embeddings')
    rel = tf.get_variable('rel_matrices')
    return _at(ent, h), _at(ent, t), _at(rel, l)


def _term(h, t, m):
    # TODO: Replace matmul with PEP465 @-operator when upgrading to Python 3.5
    return tf.squeeze(h * tf.matmul(m, t), [-1])


class RESCAL(BaseModel):
    # FIXME: RESCAL loss doesn't decrease
    def __init__(self, dimension, margin, **kwargs):
        self.dimension = dimension
        self.margin = margin
        super().__init__(**kwargs)

    def __str__(self):
        return '{}-{}'.format(type(self).__name__, self.dimension)

    def _score(self, h, t, l):
        """The term to score triples."""
        # TODO: Check whether the norm should appear here
        return self._norm(_term(*_lookup(h, t, l)))  # [.]

    def _embedding_def(self):
        """Initializes the variables of the model."""
        ent = tf.get_variable('ent_embeddings', [self.ent_count, self.dimension, 1])
        rel = tf.get_variable('rel_matrices', [self.rel_count, self.dimension, self.dimension])
        yield 'ent_embeddings', ent
        yield 'rel_matrices', rel
        self._entity = tf.squeeze(_at(ent, self.predict_h), [-1])

    def _loss_def(self):
        """Initializes the loss function."""

        def scores(h, t, l):
            s = self._score(h, t, l)  # [b,n]
            return tf.reduce_mean(s, 1)  # [b]

        p = scores(*self.get_positive_instance(in_batch=True))  # [b]
        n = scores(*self.get_negative_instance(in_batch=True))  # [b]

        return tf.reduce_sum(tf.maximum(p - n + self.margin, 0))  # []

    def _predict_def(self):
        """Initializes the prediction function."""
        return self._score(*self.get_predict_instance())  # [b]
