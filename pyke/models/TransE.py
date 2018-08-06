# coding:utf-8
import logging

import tensorflow as tf

from pyke.models import BaseModel

_at = tf.nn.embedding_lookup
logger = logging.getLogger("pyke")


class TransE(BaseModel):
    def __init__(self, dimension, margin, **kwargs):
        self.dimension = dimension
        self.margin = margin
        super().__init__(**kwargs)

    def __str__(self):
        return f"{type(self).__name__}-{self.dimension}"

    @staticmethod
    def _calc(h, t, l):
        return abs(h + l - t)

    @staticmethod
    def lookup(h, t, l):
        """
        Returns the vectors for the head, tail and label.

        :param h: head id(s)
        :param t: tail id(s)
        :param l: label id(s)
        :return: head, tail and label vectors
        """
        ent = tf.get_variable('ent_embeddings')
        rel = tf.get_variable('rel_embeddings')
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
        ent = tf.get_variable('ent_embeddings', [self.ent_count, self.dimension])
        rel = tf.get_variable('rel_embeddings', [self.rel_count, self.dimension])
        yield 'ent_embeddings', ent
        yield 'rel_embeddings', rel
        self._entity = _at(ent, self.predict_h)
        self._relation = _at(rel, self.predict_l)

    def _loss_def(self):
        """Initializes the loss function."""
        pos_h, pos_t, pos_r = self.get_positive_instance(in_batch=True)  # shape (b,1)
        pos_h_v, pos_t_v, pos_r_v = self.lookup(pos_h, pos_t, pos_r)  # shape (b,1,dim)
        _p_score = self._calc(pos_h_v, pos_t_v, pos_r_v)  # shape (b,1,dim)
        p_score = tf.reduce_sum(_p_score, -1, keepdims=True)  # shape (b,1,1); L1 norm

        neg_h, neg_t, neg_r = self.get_negative_instance(in_batch=True)  # shape (b,neg)
        neg_h_v, neg_t_v, neg_r_v = self.lookup(neg_h, neg_t, neg_r)  # shape (b,neg,dim)
        _n_score = self._calc(neg_h_v, neg_t_v, neg_r_v)  # shape (b,neg,dim)
        n_score_sum = tf.reduce_sum(_n_score, -1, keepdims=True)  # shape (b,neg,1)
        n_score = tf.reduce_mean(n_score_sum, 1, keepdims=True)  # shape (b,1,1)

        logger.debug(f"LOSS function pos_h shape {pos_h.shape}")
        logger.debug(f"LOSS function pos_h_v shape {pos_h_v.shape}")
        logger.debug(f"LOSS function _p_score shape {_p_score.shape}")
        logger.debug(f"LOSS function p_score shape {p_score.shape}")
        logger.debug(f"LOSS function neg_h shape {neg_h.shape}")
        logger.debug(f"LOSS function neg_h_v shape {neg_h_v.shape}")
        logger.debug(f"LOSS function _n_score shape {_n_score.shape}")
        logger.debug(f"LOSS function n_score_sum shape {n_score_sum.shape}")
        logger.debug(f"LOSS function n_score shape {n_score.shape}")

        loss = tf.reduce_sum(tf.maximum(p_score - n_score + self.margin, 0.0))
        return loss

    def _predict_def(self):
        """Initializes the prediction function."""
        pred_h, pred_t, pred_r = self.get_predict_instance()
        pred_h_e, pred_t_e, pred_r_e = self.lookup(pred_h, pred_t, pred_r)
        _pred_score = self._calc(pred_h_e, pred_t_e, pred_r_e)
        # pred_score = tf.reduce_sum(_pred_score, 1, keepdims=False)
        pred_score = tf.reduce_mean(_pred_score, 1, keepdims=False)  # Divides by dimension

        logger.debug(f"PREDICT function pred_h shape {pred_h.shape}")
        logger.debug(f"PREDICT function pred_h_e shape {pred_h_e.shape}")
        logger.debug(f"PREDICT function _pred_score shape {_pred_score.shape}")
        logger.debug(f"PREDICT function pred_score shape {pred_score.shape}")

        return pred_score
