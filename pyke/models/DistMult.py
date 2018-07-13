# coding:utf-8
import tensorflow as tf

from pyke.models import BaseModel

_at = tf.nn.embedding_lookup
_softplus = tf.nn.softplus


def _lookup(h, t, l):
    ent = tf.get_variable('ent_embeddings')
    rel = tf.get_variable('rel_embeddings')

    return _at(ent, h), _at(ent, t), _at(rel, l)  # [.,d]


def _term(h, t, l):
    return h * l * t


class DistMult(BaseModel):
    def __init__(self, dimension, weight, **kwargs):
        self.dimension = dimension
        self.weight = weight
        super().__init__(**kwargs)

    def __str__(self):
        return '{}-{}'.format(type(self).__name__, self.dimension)

    def _score(self, h, t, l):
        """The term to score triples."""

        return self._norm(_term(*_lookup(h, t, l)))  # [.]

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
        h, t, r = _lookup(*self.get_all_instance())  # [bp+bn,d]
        y = self.get_all_labels()  # [bp+bn]

        s = self._norm(_term(h, t, r))  # [bp+bn]
        loss = tf.reduce_mean(_softplus(y * s))  # []
        reg = tf.reduce_mean(h ** 2) + tf.reduce_mean(t ** 2) + tf.reduce_mean(r ** 2)  # []

        return loss + self.weight * reg  # []

    def _predict_def(self):
        """Initializes the prediction function."""

        return self._score(*self.get_predict_instance())  # [b]
