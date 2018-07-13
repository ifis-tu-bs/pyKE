# coding:utf-8
import tensorflow as tf

from pyke.models import BaseModel

_at = tf.nn.embedding_lookup
_softplus = tf.nn.softplus


def _term(h_re, h_im, t_re, t_im, r_re, r_im):
    """
    Returns the real part of the ComplEx embedding of a fact described by
    three complex numbers.
    """
    return h_re * t_re * r_re + h_im * t_im * r_re + h_re * t_im * r_im - h_im * t_re * r_im


def _lookup(h, t, r):
    """Gets the variables concerning a fact."""

    ent_re = tf.get_variable('ent_re_embeddings')
    rel_re = tf.get_variable('rel_re_embeddings')
    ent_im = tf.get_variable('ent_im_embeddings')
    rel_im = tf.get_variable('rel_im_embeddings')

    return _at(ent_re, h), _at(ent_re, t), _at(rel_re, r), _at(ent_im, h), _at(ent_im, t), _at(rel_im, r)


class ComplEx(BaseModel):
    def __init__(self, dimension, weight, **kwargs):
        self.dimension = dimension
        self.weight = weight
        super().__init__(**kwargs)

    def __str__(self):
        return '{}-{}'.format(type(self).__name__, self.dimension)

    def _score(self, h, t, r):
        """The term to embed triples."""

        return self._norm(_term(*_lookup(h, t, r)))  # [.]

    def _embedding_def(self):
        """Initializes the variables."""
        ent_re = tf.get_variable("ent_re_embeddings", [self.ent_count, self.dimension])
        ent_im = tf.get_variable("ent_im_embeddings", [self.ent_count, self.dimension])
        rel_re = tf.get_variable("rel_re_embeddings", [self.rel_count, self.dimension])
        rel_im = tf.get_variable("rel_im_embeddings", [self.rel_count, self.dimension])

        yield 'ent_re_embeddings', ent_re
        yield 'ent_im_embeddings', ent_im
        yield 'rel_re_embeddings', rel_re
        yield 'rel_im_embeddings', rel_im

        self._entity = _at(ent_re, self.predict_h)
        self._relation = _at(rel_re, self.predict_l)

    def _loss_def(self):
        """Initializes the loss function."""

        h_re, h_im, t_re, t_im, l_re, l_im = _lookup(*self.get_all_instance())  # [b,d]
        y = self.get_all_labels()  # [b]

        s = self._norm(_term(h_re, h_im, t_re, t_im, l_re, l_im))  # [b]
        loss = tf.reduce_mean(_softplus(- y * s), 0)  # []
        reg = tf.reduce_mean(h_re ** 2) + tf.reduce_mean(h_im ** 2) + tf.reduce_mean(t_re ** 2) + tf.reduce_mean(
            t_im ** 2) + tf.reduce_mean(l_re ** 2) + tf.reduce_mean(l_im ** 2)

        return loss + self.weight * reg  # []

    def _predict_def(self):
        """Initializes the prediction function."""
        return self._score(*self.get_predict_instance())
