# coding:utf-8
import tensorflow as tf

from pyke.models import BaseModel

_at = tf.nn.embedding_lookup


def _score(h, t, l):
    """The term to embed triples."""
    ent = tf.get_variable('ent_embeddings')
    rel = tf.get_variable('rel_embeddings')
    etr = tf.get_variable('ent_transfer')
    rtr = tf.get_variable('rel_transfer')

    return _at(ent, h), _at(etr, h), _at(ent, t), _at(etr, t), _at(rel, l), _at(rtr, l)


def _term(he, ht, te, tt, le, lt):
    def transfer(e, t):
        return e + tf.reduce_sum(e * t, -1, keepdims=True) * lt

    return transfer(he, ht) + le - transfer(te, tt)


class TransD(BaseModel):
    def __init__(self, dimension, margin, **kwargs):
        self.dimension = dimension
        self.margin = margin
        super().__init__(**kwargs)

    def __str__(self):
        return '{}-{}'.format(type(self).__name__, self.dimension)

    def _score(self, h, t, l):
        # TODO: Implement lookup
        return self._norm(_term(*_lookup(h, t, l)))

    def _embedding_def(self):
        """Initializes the variables of the model."""

        ent = tf.get_variable("ent_embeddings", [self.ent_count, self.dimension])
        rel = tf.get_variable("rel_embeddings", [self.rel_count, self.dimension])
        etr = tf.get_variable("ent_transfer", [self.ent_count, self.dimension])
        rtr = tf.get_variable("rel_transfer", [self.rel_count, self.dimension])

        yield 'ent_embeddings', ent
        yield 'rel_embeddings', rel
        yield 'ent_transfer', etr
        yield 'rel_transfer', rtr

        self._entity = _at(ent, self.predict_h)
        self._relation = _at(rel, self.predict_l)

    def _loss_def(self):
        """Initializes the loss function."""

        def scores(h, t, l):
            s = _score(h, t, l)  # [b,n]
            return tf.reduce_mean(s, 1)  # [b]

        p = scores(*self.get_positive_instance(in_batch=True))  # [b]
        n = scores(*self.get_negative_instance(in_batch=True))  # [b]

        return tf.reduce_sum(tf.maximum(p - n + self.margin, 0))  # []

    def _predict_def(self):
        """Initializes the prediction function."""

        return _score(*self.get_predict_instance())  # [b]
