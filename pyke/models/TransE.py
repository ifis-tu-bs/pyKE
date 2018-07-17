# coding:utf-8
import tensorflow as tf

from pyke.models import BaseModel

_at = tf.nn.embedding_lookup


class TransE(BaseModel):
    def __init__(self, dimension, margin, **kwargs):
        self.dimension = dimension
        self.margin = margin
        super().__init__(**kwargs)

    def __str__(self):
        return f"{type(self).__name__}-{self.dimension}"

    @staticmethod
    def _calc(h, t, l):
        return abs(h - t + l)

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

    # def get_triple_score(self, h, t, l):
    #     """
    #     Calculates the score for a triple.
    #
    #     :param h: head ids
    #     :param t: tail ids
    #     :param l: label ids
    #     :return: vector with size of head-vector with the norms
    #     """
    #     hv, tv, lv = self.lookup(h, t, l)  # head-vector, tail-vector, label-vector
    #     result = self._calc(hv, tv, lv)  # result-vector
    #     return self._norm(result)  # [n]

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
        pos_h, pos_t, pos_r = self.get_positive_instance(in_batch=True)
        pos_h_v, pos_t_v, pos_r_v = self.lookup(pos_h, pos_t, pos_r)
        _p_score = self._calc(pos_h_v, pos_t_v, pos_r_v)
        p_score = tf.reduce_sum(tf.reduce_mean(_p_score, 1, keepdims=False), 1, keepdims=True)

        neg_h, neg_t, neg_r = self.get_negative_instance(in_batch=True)
        neg_h_v, neg_t_v, neg_r_v = self.lookup(neg_h, neg_t, neg_r)
        _n_score = self._calc(neg_h_v, neg_t_v, neg_r_v)
        n_score = tf.reduce_sum(tf.reduce_mean(_n_score, 1, keepdims=False), 1, keepdims=True)

        loss = tf.reduce_sum(tf.maximum(p_score - n_score + self.margin, 0.0))
        return loss

    def _predict_def(self):
        """Initializes the prediction function."""
        pred_h, pred_t, pred_r = self.get_predict_instance()
        pred_h_e, pred_t_e, pred_r_e = self.lookup(pred_h, pred_t, pred_r)
        return tf.reduce_sum(self._calc(pred_h_e, pred_t_e, pred_r_e), 1)
