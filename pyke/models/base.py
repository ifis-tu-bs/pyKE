# -*- coding: utf-8 -*-
import json
import logging

import numpy as np
import tensorflow as tf

from pyke import norm

logger = logging.getLogger("pyke")


class BaseModel(object):
    """Properties and behaviour that different embedding models share."""

    def __init__(self, ent_count=None, rel_count=None, batch_size=0, variants=0, optimizer=None, norm_func=norm.l1,
                 per_process_gpu_memory_fraction=0.5, learning_rate=0.01):
        """
        Creates a new model.

            baseshape
        A pair of numbers describing the amount of entities and relations.

            batchshape
        A pair of numbers describing the amount of training statements per
        iteration and the amount of variants per statement.
        The first variant is considered true while the rest is considered false.
        default: Model not intended for training

            optimizer
        The optimization algorithm used to approximate the optimal model in each iteration.
        default: Stochastic Gradient Descent with learning factor of 1%.

            norm
        The used vector norm to compute a scalar score from the model's prediction.
        default: L1 norm (sum of absolute features).

            per_process_gpu_memory_fraction
        Fraction of GPU memory to be used by tensorflow.
        """
        self._relation = None
        self._rel_embeddings = None
        self._ent_embeddings = None
        self._entity = None

        self.ent_count = ent_count
        self.rel_count = rel_count

        self.batch_size = batch_size
        self.neg_total = variants - 1

        self.__parameters = dict()

        if optimizer is None:
            optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        self._norm = norm_func

        shape = self.batch_size * (self.neg_total + 1)
        self.__graph = tf.Graph()
        with self.__graph.as_default():
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=per_process_gpu_memory_fraction)
            self.__session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
            with self.__session.as_default():
                # TODO: Does it matter that TransX uses uniform=False?
                initializer = tf.contrib.layers.xavier_initializer(uniform=True)
                with tf.variable_scope('model', reuse=tf.AUTO_REUSE, initializer=initializer):
                    with tf.name_scope('input'):
                        self.batch_h = tf.placeholder(tf.int64, [shape])
                        self.batch_t = tf.placeholder(tf.int64, [shape])
                        self.batch_l = tf.placeholder(tf.int64, [shape])
                        self.batch_y = tf.placeholder(tf.float32, [shape])
                        self.all_h = tf.transpose(tf.reshape(self.batch_h, [1 + self.neg_total, -1]), [1, 0])
                        self.all_t = tf.transpose(tf.reshape(self.batch_t, [1 + self.neg_total, -1]), [1, 0])
                        self.all_l = tf.transpose(tf.reshape(self.batch_l, [1 + self.neg_total, -1]), [1, 0])
                        self.all_y = tf.transpose(tf.reshape(self.batch_y, [1 + self.neg_total, -1]), [1, 0])
                        self.postive_h = tf.transpose(tf.reshape(self.batch_h[:self.batch_size], [1, -1]), [1, 0])
                        self.postive_t = tf.transpose(tf.reshape(self.batch_t[:self.batch_size], [1, -1]), [1, 0])
                        self.postive_l = tf.transpose(tf.reshape(self.batch_l[:self.batch_size], [1, -1]), [1, 0])
                        self.negative_h = tf.transpose(tf.reshape(self.batch_h[self.batch_size:], [self.neg_total, -1]),
                                                       [1, 0])
                        self.negative_t = tf.transpose(tf.reshape(self.batch_t[self.batch_size:], [self.neg_total, -1]),
                                                       [1, 0])
                        self.negative_l = tf.transpose(tf.reshape(self.batch_l[self.batch_size:], [self.neg_total, -1]),
                                                       [1, 0])
                        self.predict_h = tf.placeholder(tf.int64, [None])
                        self.predict_t = tf.placeholder(tf.int64, [None])
                        self.predict_l = tf.placeholder(tf.int64, [None])
                    with tf.name_scope('embedding'):
                        for k, v in self._embedding_def():
                            self.__parameters[k] = v
                    with tf.name_scope('loss'):
                        self.__loss = self._loss_def()
                    with tf.name_scope('predict'):
                        self.__prediction = self._predict_def()
                    grads_and_vars = optimizer.compute_gradients(self.__loss)
                    self.__training = optimizer.apply_gradients(grads_and_vars)
                self.__saver = tf.train.Saver()
                self.__session.run(tf.global_variables_initializer())

        logger.debug(f"batch_h shape: {self.batch_h.shape}")
        logger.debug(f"postive_h shape: {self.postive_h.shape}")
        logger.debug(f"negative_h shape: {self.negative_h.shape}")
        logger.debug(f"all_h shape: {self.all_h.shape}")

    def __iter__(self):
        """Iterates all parameter fields of the model."""
        return iter(self.__parameters)

    def __getitem__(self, key):
        """Retrieves the values of a parameter field."""
        return self.__session.run(self.__parameters[key])

    def __setitem__(self, key, value):
        """Updates the values of a parameter field."""
        with self.__graph.as_default():
            with self.__session.as_default():
                self.__parameters[key].assign(value).eval()

    def __del__(self):
        self.__session.close()

    def fit(self, head, tail, label, score):
        """Trains the model on a batch of weighted statements."""
        feed = {
            self.batch_h: head,
            self.batch_t: tail,
            self.batch_l: label,
            self.batch_y: score,
        }
        _, loss = self.__session.run([self.__training, self.__loss], feed)
        return loss

    def predict(self, head_id, tail_id, label_id):
        """Evaluates the model's scores on a batch of statements."""
        # transform wildcard parameters
        # require otherwise scalar parameters
        heads, tails, labels = [head_id], [tail_id], [label_id]
        if head_id is None:
            if tail_id is None:
                if label_id is None:
                    raise NotImplementedError('universal prediction')
                raise NotImplementedError('full-relation prediction')
            elif label_id is None:
                raise NotImplementedError('full-tail prediction')
            heads, tails, labels = np.arange(self.ent_count), np.full([self.ent_count], tail_id), np.full(
                [self.ent_count],
                label_id)
        elif tail_id is None:
            if label_id is None:
                raise NotImplementedError('full-head prediction')
            heads, tails, labels = np.full([self.ent_count], head_id), np.arange(self.ent_count), np.full(
                [self.ent_count],
                label_id)
        elif label_id is None:
            heads, tails, labels = np.full([self.rel_count], head_id), np.full([self.rel_count], tail_id), np.arange(
                self.rel_count)

        # perform prediction
        feed = {
            self.predict_h: heads,
            self.predict_t: tails,
            self.predict_l: labels,
        }
        return self.__session.run(self.__prediction, feed)

    def relation(self, label=None):
        """Embeds a batch of predicates."""
        if label is None:
            with self.__graph.as_default():
                with self.__session.as_default():
                    return self.__session.run(self._rel_embeddings)
        feed = {
            self.predict_l: label,
        }
        return self.__session.run(self._relation, feed)

    def entity(self, head=None):
        """Embeds a batch of subjects."""
        if head is None:
            with self.__session.as_default():
                return self.__session.run(self._ent_embeddings)
        feed = {
            self.predict_h: head,
        }
        return self.__session.run(self._entity, feed)

    def save(self, prefix: str, step: int = None):
        """
        Save the model to filesystem.

        :param prefix: File prefix for the model
        :param step: Step of the model (appended to prefix)
        """
        if step:
            self.__saver.save(self.__session, prefix, global_step=step)
        else:
            self.__saver.save(self.__session, prefix)

    def save_to_json(self, filename: str):
        """
        Save the embedding as JSON file. The JSON file contains the embedding parameters (e.g. entity and relation
        matrices). These parameters depend on the model.

        :param filename: Filename for the output JSON file
        """
        content = {}
        for var_name in self.__parameters:
            with self.__graph.as_default():
                with self.__session.as_default():
                    content[var_name] = self.__session.run(self.__parameters[var_name]).tolist()
        with open(filename, "w") as f:
            f.write(json.dumps(content))

    def restore(self, prefix: str):
        """
        Reads a model from filesystem.

        :param prefix: Model prefix of the model to laod
        """
        self.__saver.restore(self.__session, prefix)

    def get_positive_instance(self, in_batch=True):
        if in_batch:
            return self.postive_h, self.postive_t, self.postive_l
        else:
            return self.batch_h[:self.batch_size], self.batch_t[:self.batch_size], self.batch_l[:self.batch_size]

    def get_negative_instance(self, in_batch=True):
        if in_batch:
            return self.negative_h, self.negative_t, self.negative_l
        else:
            return self.batch_h[self.batch_size:], self.batch_t[self.batch_size:], self.batch_l[self.batch_size:]

    def get_all_instance(self, in_batch=False):
        if in_batch:
            return self.all_h, self.all_t, self.all_l
        return self.batch_h, self.batch_t, self.batch_l

    def get_all_labels(self, in_batch=False):
        if in_batch:
            return self.all_y
        return self.batch_y

    def get_predict_instance(self):
        return [self.predict_h, self.predict_t, self.predict_l]

    def _embedding_def(self):
        raise NotImplementedError

    def _loss_def(self):
        raise NotImplementedError

    def _predict_def(self):
        raise NotImplementedError
