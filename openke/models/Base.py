#coding:utf-8
from tensorflow import name_scope, transpose, reshape, placeholder, int64, float32
from tensorflow import Session, Graph, global_variables_initializer, variable_scope, nn, AUTO_REUSE
from tensorflow.contrib.layers import xavier_initializer
from tensorflow.python.training.saver import Saver



class ModelClass(object):


	def fit(self, head, tail, label, score):
		'''Trains the model on a batch of weighted statements.'''
		feed = {
				self.batch_h:head,
				self.batch_t:tail,
				self.batch_l:label,
				self.batch_y:score}
		with self.__graph.as_default():
			with self.__session.as_default():
				return self.__session.run([self.__training, self.__loss], feed)[1]


	def predict(self, head, tail, label):
		'''Evaluates the model's scores on a batch of statements.'''
		feed = {
				self.predict_h:head,
				self.predict_t:tail,
				self.predict_l:label}
		with self.__graph.as_default():
			with self.__session.as_default():
				return self.__session.run(self.__prediction, feed)


	def relation(self, label):
		'''Embeds a batch of predicates.'''
		feed = {
				self.predict_l:label}
		with self.__graph.as_default():
			with self.__session.as_default():
				return self.__session.run(self._relation, feed)


	def entity(self, head):
		'''Embeds a batch of subjects.'''
		feed = {
				self.predict_h:head}
		with self.__graph.as_default():
			with self.__session.as_default():
				return self.__session.run(self._entity, feed)


	def save(self, fileprefix):
		'''Writes the model's state into persistent memory.'''
		with self.__graph.as_default():
			self.__saver.save(self.__session, fileprefix)


	def restore(self, fileprefix):
		'''Reads a model from persistent memory.'''
		with self.__graph.as_default():
			self.__saver.restore(self.__session, fileprefix)


	def __iter__(self):
		'''Iterates all parameter fields of the model.'''
		return iter(self.__parameters)


	def __getitem__(self, key):
		'''Retrieves the values of a parameter field.'''
		with self.__graph.as_default():
			with self.__session.as_default():
				return self.__session.run(self.__parameters[key])


	def __setitem__(self, key, value):
		'''Updates the values of a parameter field.'''
		with self.__graph.as_default():
			with self.__session.as_default():
				self.__parameters[key].assign(value).eval()


	def embedding_def(self):
		pass


	def loss_def(self):
		raise NotImplementedError('loss impossible without model')


	def predict_def(self):
		raise NotImplementedError('prediction impossible without model')


	def __init__(self, baseshape, batchshape, optimizer=None):
		'''Creates a new model.

	baseshape
A pair of numbers describing the amount of entities and relations.

	batchshape
A pair of numbers describing the amount of training statements per iteration and the amount of variants per statement.
The first variant is considered true while the rest is considered false.

	optimizer
The optimization algorithm used to approximate the optimal model in each iteration.
default: Stochastic Gradient Descent with learning factor of 1%.'''

		self.base = baseshape
		self.batchsize = batchshape[0]
		self.negatives = batchshape[1] - 1
		self.__parameters = dict()
		if optimizer is None:
			import tensorflow
			optimizer = tensorflow.train.GradientDescentOptimizer(.01)
		B, N = self.batchsize, self.negatives
		S = B * (N + 1)
		self.__graph = Graph()
		with self.__graph.as_default():
			self.__session = Session()
			with self.__session.as_default():
				initializer = xavier_initializer(uniform=True)
				with variable_scope('model', reuse=AUTO_REUSE, initializer=initializer):
					with name_scope('input'):
						self.batch_h = placeholder(int64, [S])
						self.batch_t = placeholder(int64, [S])
						self.batch_l = placeholder(int64, [S])
						self.batch_y = placeholder(float32, [S])
						self.all_h = transpose(reshape(self.batch_h, [1+N,-1]), [1,0])
						self.all_t = transpose(reshape(self.batch_t, [1+N,-1]), [1,0])
						self.all_l = transpose(reshape(self.batch_l, [1+N,-1]), [1,0])
						self.all_y = transpose(reshape(self.batch_y, [1+N,-1]), [1,0])
						self.postive_h = transpose(reshape(self.batch_h[:B], [1,-1]), [1,0])
						self.postive_t = transpose(reshape(self.batch_t[:B], [1,-1]), [1,0])
						self.postive_l = transpose(reshape(self.batch_l[:B], [1,-1]), [1,0])
						self.negative_h = transpose(reshape(self.batch_h[B:], [N,-1]), [1,0])
						self.negative_t = transpose(reshape(self.batch_t[B:], [N,-1]), [1,0])
						self.negative_l = transpose(reshape(self.batch_l[B:], [N,-1]), [1,0])
						self.predict_h = placeholder(int64, [None])
						self.predict_t = placeholder(int64, [None])
						self.predict_l = placeholder(int64, [None])
					with name_scope('embedding'):
						for k, v in self._embedding_def():
							self.__parameters[k] = v
					with name_scope('loss'):
						self.__loss = self._loss_def()
					with name_scope('predict'):
						self.__prediction = self._predict_def()
					grads_and_vars = optimizer.compute_gradients(self.__loss)
					self.__training = optimizer.apply_gradients(grads_and_vars)
				self.__saver = Saver()
				self.__session.run(global_variables_initializer())


	def _positive_instance(self, in_batch=True):
		if in_batch:
			return self.postive_h, self.postive_t, self.postive_l
		B = self.batchsize
		return self.batch_h[:B], self.batch_t[:B], self.batch_l[:B]


	def _negative_instance(self, in_batch=True):
		if in_batch:
			return self.negative_h, self.negative_t, self.negative_l
		B = self.batchsize
		return self.batch_h[B:], self.batch_t[B:], self.batch_l[B:]


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
