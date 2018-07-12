# coding:utf-8
import tensorflow as tf

from pyke.models import BaseModel

_at = tf.nn.embedding_lookup


class TransE(BaseModel):
    # TODO: Add **kwargs
    def __init__(self, dimension, margin, ent_count, rel_count, batch_size=0, variants=0, optimizer=None):
        self.dimension = dimension
        self.margin = margin
        super().__init__(ent_count, rel_count, batch_size, variants, optimizer=optimizer)

    def __str__(self):
        return f"{type(self).__name__}-{self.dimension}"

    @staticmethod
    def term(h, t, l):
        return h - t + l

    @staticmethod
    def lookup(h, t, l):
        """
        Returns the vectors for the head, tail and label.

        :param h: head id(s)
        :param t: tail id(s)
        :param l: label id(s)
        :return: head, tail and label vectors
        """
        ent = tf.get_variable('ent_embedding')
        rel = tf.get_variable('rel_embedding')
        return _at(ent, h), _at(ent, t), _at(rel, l)

    def get_triple_score(self, h, t, l):
        """The term to score triples."""
        return self._norm(self.term(*self.lookup(h, t, l)))  # [.]

    def _embedding_def(self):
        """Initializes the variables of the model."""
        ent = tf.get_variable('ent_embedding', [self.ent_count, self.dimension])
        rel = tf.get_variable('rel_embedding', [self.rel_count, self.dimension])
        yield 'ent_embedding', ent
        yield 'rel_embedding', rel
        self._entity = _at(ent, self.predict_h)
        self._relation = _at(rel, self.predict_l)

    def _loss_def(self):
        """Initializes the loss function."""

        def scores(h, t, l):
            s = self.get_triple_score(h, t, l)  # [b,n]
            return tf.reduce_mean(s, 1)  # [b]

        p = scores(*self.get_positive_instance(in_batch=True))  # [b]
        n = scores(*self.get_negative_instance(in_batch=True))  # [b]
        loss = tf.reduce_sum(tf.maximum(p - n + self.margin, 0.0))  # []
        return loss

    def _predict_def(self):
        """Initializes the prediction function."""

        return self.get_triple_score(*self.get_predict_instance())  # [b]
