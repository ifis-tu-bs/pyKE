#coding:utf-8
from tensorflow import (Session, Graph,
                        name_scope, variable_scope,
                        transpose, reshape, placeholder,
                        int64, float32,
                        global_variables_initializer, nn
at = nn.embedding_lookup
from tensorflow.contrib.layers import xavier_initializer as xavier
from tensorflow.python.training.saver import Saver


class Model(object):


	def train(self, head, tail, label, objective):
		'''Performs one unit of training on a batch of graded triples.'''
		feed_dict = {
				self.batch_h: head,
				self.batch_t: tail,
				self.batch_r: label,
				self.batch_y: objective}
		with self._graph.as_default():
			with self._session.as_default():
				return self._session.run([self._train, self.loss], feed_dict)


	def predict(self, head, tail, label):
		'''Evaluates each triple.'''
		feed_dict = {
				self.predict_h: head,
				self.predict_t: tail,
				self.predict_r: label}
		with self._graph.as_default():
			with self._session.as_default():
				return self._session.run(self.predict, feed_dict)


	def rel_embed(self, head, tail, label):
		'''Embeds each triple into semantic space.'''
		feed_dict = {
				self.predict_h: head,
				self.predict_t: tail,
				self.predict_r: label}
		with self._graph.as_default():
			with self._session.as_default():
				return self._session.run(self.embed, feed_dict)


	def ent_embed(self, entities):
		'''Embeds each entity into semantic space.'''
		feed_dict = {
				self.predict_h: entities}
		with self._graph.as_default():
			with self._session.as_default():
				e = at(self.ent_embeddings, self.predict_h)
				return self._session.run(e, feed_dict)


	def get_positive_instance(self, in_batch=True):
		if in_batch:
			return [self.postive_h, self.postive_t, self.postive_r]
		B = self.batchsize
		return [self.batch_h[0:B], self.batch_t[0:B], self.batch_r[0:B]]


	def get_negative_instance(self, in_batch=True):
		if in_batch:
			return [self.negative_h, self.negative_t, self.negative_r]
		B = self.batchsize
		return [self.batch_h[B:], self.batch_t[B:], self.batch_r[B:]]


	def get_all_instance(self, in_batch=False):
		if not in_batch:
			return [self.batch_h, self.batch_t, self.batch_r]
		N = self.negatives
		return [transpose(reshape(self.batch_h, [1+N,-1]), [1,0]),\
				transpose(reshape(self.batch_t, [1+N,-1]), [1,0]),\
				transpose(reshape(self.batch_r, [1+N,-1]), [1,0])]


	def get_all_labels(self, in_batch=False):
		if not in_batch:
			return self.batch_y
		return transpose(reshape(self.batch_y, [1+self.negatives, -1]), [1,0])


	def get_predict_instance(self):
		return [self.predict_h, self.predict_t, self.predict_r]


	def input_def(self):
		B, N = self.batchsize, self.negatives
		S = B * (N + 1)
		self.batch_h = placeholder(int64, [S])
		self.batch_t = placeholder(int64, [S])
		self.batch_r = placeholder(int64, [S])
		self.batch_y = placeholder(float32, [S])
		self.postive_h = transpose(reshape(self.batch_h[0:B], [1,-1]), [1,0])
		self.postive_t = transpose(reshape(self.batch_t[0:B], [1,-1]), [1,0])
		self.postive_r = transpose(reshape(self.batch_r[0:B], [1,-1]), [1,0])
		self.negative_h = transpose(reshape(self.batch_h[B:S], [N,-1]), perm=[1,0])
		self.negative_t = transpose(reshape(self.batch_t[B:S], [N,-1]), perm=[1,0])
		self.negative_r = transpose(reshape(self.batch_r[B:S], [N,-1]), perm=[1,0])
		self.predict_h = placeholder(int64, [None])
		self.predict_t = placeholder(int64, [None])
		self.predict_r = placeholder(int64, [None])
		self.parameter_lists = []


	def embedding_def(self):
		pass


	def loss_def(self):
		raise NotImplementedError('loss impossible without model')


	def predict_def(self):
		raise NotImplementedError('prediction impossible without model')


	def __getitem__(self, name):
		'''Returns one of the object's parameters.'''
		if name not in self._parameters:
			raise KeyError('model has no such parameter')
		with self._graph.as_default():
			with self._session.as_default():
				return self._session.run(self._parameters[name])


	def __setitem__(self, name, tensor):
		'''Updates one of the object's parameters.'''
		if name not in self._parameters:
			raise KeyError('model has no such parameter')
		with self._graph.as_default():
			with self._session.as_default():
				self.parameter_lists[name].assign(tensor).eval()


	def save(self, filename=None):
		'''Attempts to persistently store the model's parameters.'''
		if filename is not None:
			self._filename = filename
		with self._graph.as_default():
			with self._session.as_default():
				self._saver.save(self._session, self._filename)


	def restore(self, filename=None):
		'''Loads the model's parameters from a file.'''
		if filename is not None:
			self._filename = filename
		with self._graph.as_default():
			with self._session.as_default():
				self._saver.restore(self._session, self._filename)


	def __init__(self, **config):
		self.batchsize = config['batch_size']
		self.negatives = config['negative_ent'] + config['negative_rel']
		self._optimizer = config['optimizer']
		grads_and_vars = self._optimizer.compute_gradients(self._m.loss)
		self._train = self._optimizer.apply_gradients(grads_and_vars)
		self._graph = Graph()
		with self._graph.as_default():
			self._session = Session()
			with self._session.as_default():
				with variable_scope('model', reuse=None,
						initializer=xavier(uniform=True)):
					with name_scope("input"):
						self.input_def()
					with name_scope("embedding"):
						self.embedding_def()
					with name_scope("loss"):
						self.loss_def()
					with name_scope("predict"):
						self.predict_def()
				self._saver = Saver()
				self._session.run(global_variables_initializer())
