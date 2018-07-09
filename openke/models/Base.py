# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
from tensorflow.python.training.saver import Saver


class ModelClass(object):
    """Properties and behaviour that different embedding models share."""

    def __init__(self, baseshape, batchshape=None, optimizer=None, norm=None, per_process_gpu_memory_fraction=0.5):
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
        self.base = baseshape
        if batchshape is None:
            batchshape = 0, 0
        self.batchsize = batchshape[0]
        self.negatives = batchshape[1] - 1
        self.__parameters = dict()
        if optimizer is None:
            import tensorflow
            optimizer = tensorflow.train.GradientDescentOptimizer(.01)
        if norm is None:
            from .norm import l1
            norm = l1
        self._norm = norm
        batch_size, negatives = self.batchsize, self.negatives
        shape = batch_size * (negatives + 1)
        self.__graph = tf.Graph()
        with self.__graph.as_default():
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=per_process_gpu_memory_fraction)
            self.__session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
            with self.__session.as_default():
                initializer = tf.contrib.layers.xavier_initializer(uniform=True)
                with tf.variable_scope('model', reuse=tf.AUTO_REUSE, initializer=initializer):
                    with tf.name_scope('input'):
                        self.batch_h = tf.placeholder(tf.int64, [shape])
                        self.batch_t = tf.placeholder(tf.int64, [shape])
                        self.batch_l = tf.placeholder(tf.int64, [shape])
                        self.batch_y = tf.placeholder(tf.float32, [shape])
                        self.all_h = tf.transpose(tf.reshape(self.batch_h, [1 + negatives, -1]), [1, 0])
                        self.all_t = tf.transpose(tf.reshape(self.batch_t, [1 + negatives, -1]), [1, 0])
                        self.all_l = tf.transpose(tf.reshape(self.batch_l, [1 + negatives, -1]), [1, 0])
                        self.all_y = tf.transpose(tf.reshape(self.batch_y, [1 + negatives, -1]), [1, 0])
                        self.postive_h = tf.transpose(tf.reshape(self.batch_h[:batch_size], [1, -1]), [1, 0])
                        self.postive_t = tf.transpose(tf.reshape(self.batch_t[:batch_size], [1, -1]), [1, 0])
                        self.postive_l = tf.transpose(tf.reshape(self.batch_l[:batch_size], [1, -1]), [1, 0])
                        self.negative_h = tf.transpose(tf.reshape(self.batch_h[batch_size:], [negatives, -1]), [1, 0])
                        self.negative_t = tf.transpose(tf.reshape(self.batch_t[batch_size:], [negatives, -1]), [1, 0])
                        self.negative_l = tf.transpose(tf.reshape(self.batch_l[batch_size:], [negatives, -1]), [1, 0])
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
                self.__saver = Saver()
                self.__session.run(tf.global_variables_initializer())

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

    def embedding_def(self):
        pass

    def loss_def(self):
        raise NotImplementedError('loss impossible without model')

    def predict_def(self):
        raise NotImplementedError('prediction impossible without model')

    def fit(self, head, tail, label, score):
        """Trains the model on a batch of weighted statements."""
        feed = {
            self.batch_h: head,
            self.batch_t: tail,
            self.batch_l: label,
            self.batch_y: score,
        }
        return self.__session.run([self.__training, self.__loss], feed)[1]

    def predict(self, head, tail, label):
        """Evaluates the model's scores on a batch of statements."""
        # transform wildcard parameters
        # require otherwise scalar parameters
        ent_count, rel_count = self.base
        if head is None:
            if tail is None:
                if label is None:
                    raise NotImplementedError('universal prediction')
                raise NotImplementedError('full-relation prediction')
            elif label is None:
                raise NotImplementedError('full-tail prediction')
            head, tail, label = np.arange(ent_count), np.full([ent_count], tail), np.full([ent_count], label)
        elif tail is None:
            if label is None:
                raise NotImplementedError('full-head prediction')
            head, tail, label = np.full([ent_count], head), np.arange(ent_count), np.full([ent_count], label)
        elif label is None:
            head, tail, label = np.full([rel_count], head), np.full([rel_count], tail), np.arange(rel_count)

        # perform prediction
        feed = {
            self.predict_h: head,
            self.predict_t: tail,
            self.predict_l: label}
        return self.__session.run(self.__prediction, feed)

    def relation(self, label=None):
        """Embeds a batch of predicates."""
        if label is None:
            with self.__graph.as_default():
                with self.__session.as_default():
                    return self.__session.run(self._rel_embeddings)
        feed = {
            self.predict_l: label}
        return self.__session.run(self._relation, feed)

    def entity(self, head=None):
        """Embeds a batch of subjects."""
        if head is None:
            with self.__session.as_default():
                return self.__session.run(self._ent_embeddings)
        feed = {
            self.predict_h: head}
        return self.__session.run(self._entity, feed)

    def save(self, fileprefix):
        """Writes the model's state into persistent memory."""
        self.__saver.save(self.__session, fileprefix)

    def restore(self, fileprefix):
        """Reads a model from persistent memory."""
        self.__saver.restore(self.__session, fileprefix)

    def _positive_instance(self, in_batch=True):
        if in_batch:
            return self.postive_h, self.postive_t, self.postive_l
        batch_size = self.batchsize
        return self.batch_h[:batch_size], self.batch_t[:batch_size], self.batch_l[:batch_size]

    def _negative_instance(self, in_batch=True):
        if in_batch:
            return self.negative_h, self.negative_t, self.negative_l
        batch_size = self.batchsize
        return self.batch_h[batch_size:], self.batch_t[batch_size:], self.batch_l[batch_size:]

    def _all_instance(self, in_batch=False):
        if in_batch:
            return self.all_h, self.all_t, self.all_l
        return self.batch_h, self.batch_t, self.batch_l

    def _all_labels(self, in_batch=False):
        if in_batch:
            return self.all_y
        return self.batch_y

    def _predict_instance(self):
        return [self.predict_h, self.predict_t, self.predict_l]

    def _embedding_def(self):
        raise NotImplementedError

    def _loss_def(self):
        raise NotImplementedError

    def _predict_def(self):
        raise NotImplementedError
