#coding:utf-8
from tensorflow import name_scope, transpose, reshape, placeholder, int64, float32


class Model(object):


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
		pass


	def predict_def(self):
		pass


	def __init__(self, **config):
		self.batchsize = config['batch_size']
		self.negatives = config['negative_ent'] + config['negative_rel']
		with name_scope("input"):
			self.input_def()
		with name_scope("embedding"):
			self.embedding_def()
		with name_scope("loss"):
			self.loss_def()
		with name_scope("predict"):
			self.predict_def()
