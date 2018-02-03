#coding:utf-8
from tensorflow import Session, Graph, initialize_all_variables, variable_scope
from tensorflow.contrib.layers import xavier_initializer
import tensorflow as tf
Saver = tf.train.Saver
from numpy import zeros, int64, float32
import os
import time
import datetime
from ctypes import c_void_p, c_int64, c_char_p, cdll, CFUNCTYPE, c_int
from json import dumps
def c_str(s):
	return c_char_p(bytes(s, 'utf-8'))



class Config(object):


	def __init__(self, library='./release/Base.so'):
		self._l = cdll.LoadLibrary(library)
		self._l.sampling.argtypes = [c_void_p, c_void_p, c_void_p, c_void_p, c_int64, c_int64, c_int64, c_int64]
		self._l.bernSampling.argtypes = self._l.sampling.argtypes
		self._l.getHeadBatch.argtypes = [c_void_p, c_void_p, c_void_p]
		self._l.getTailBatch.argtypes = [c_void_p, c_void_p, c_void_p]
		self._l.testHead.argtypes = [c_void_p]
		self._l.testTail.argtypes = [c_void_p]
		self.logcall = CFUNCTYPE(c_int, c_int)
		self._l.importTrainFiles.argtypes = [c_void_p, c_int64, c_int64, self.logcall]
		self._l.importTestFiles.argtypes = [c_void_p, c_void_p, c_void_p]
		self._l.randReset.argtypes = [c_int64]
		self.export_filename = None
		self.import_filename = None
		self.export_steps = 0


	def init(self, filename, entities, relations, batch_count=1,
			negative_entities=0, negative_relations=0, seed=1):
		self.negative_ent, self.negative_rel = negative_entities, negative_relations
		self._m = None
		self._l.randReset(seed)
		self.entTotal, self.relTotal = entities, relations
		self._l.importTrainFiles(c_str(filename), entities, relations,
				self.logcall(lambda x: print(x) or 0))
		self.trainTotal = self._l.getTrainTotal()
		self.nbatches = batch_count
		self.batch_size = self.trainTotal // batch_count

		self.batch_seq_size = self.batch_size * (1 + self.negative_ent + self.negative_rel)
		self.batch_h = zeros(self.batch_seq_size, dtype=int64)
		self.batch_t = zeros(self.batch_seq_size, dtype=int64)
		self.batch_r = zeros(self.batch_seq_size, dtype=int64)
		self.batch_y = zeros(self.batch_seq_size, dtype=float32)
		self.batch_h_addr = self.batch_h.__array_interface__['data'][0]
		self.batch_t_addr = self.batch_t.__array_interface__['data'][0]
		self.batch_r_addr = self.batch_r.__array_interface__['data'][0]
		self.batch_y_addr = self.batch_y.__array_interface__['data'][0]


	def inittest(self, testname, trainname, validname):
		self._l.importTestFiles(c_str(testname), c_str(trainname), c_str(validname))
		self.test_h = zeros(self.entTotal, dtype=int64)
		self.test_t = zeros(self.entTotal, dtype=int64)
		self.test_r = zeros(self.entTotal, dtype=int64)
		self.test_h_addr = self.test_h.__array_interface__['data'][0]
		self.test_t_addr = self.test_t.__array_interface__['data'][0]
		self.test_r_addr = self.test_r.__array_interface__['data'][0]


	def set_export(self, graphname, parametername, steps=0):
		self.graphname = graphname
		self.parametername = parametername
		self.export_steps = steps


	def _save(self, path):
		self.saver.save(self._s, path)


	def save(self, path=None):
		if not path and not self.graphname:
			return
		with self._g.as_default():
			with self._s.as_default():
				self._save(path or self.graphname)


	def _restore(self, path):
		self.saver.restore(self._s, path)


	def restore(self, path=None):
		if not path and not self.parametername:
			return
		with self._g.as_default():
			with self._s.as_default():
				self._restore(path or self.parametername)


	def get_parameter_lists(self):
		return self._m.parameter_lists


	def get_parameters_by_name(self, var_name):
		with self._g.as_default():
			with self._s.as_default():
				if var_name in self._m.parameter_lists:
					return self._s.run(self._m.parameter_lists[var_name])
				else:
					return None


	def get_parameters(self, mode="numpy"):
		res = {}
		lists = self.get_parameter_lists()
		for var_name in lists:
			if mode == "numpy":
				res[var_name] = self.get_parameters_by_name(var_name)
			else:
				res[var_name] = self.get_parameters_by_name(var_name).tolist()
		return res


	def save_parameters(self, path=None):
		f = open(path, "w")
		f.write(dumps(self.get_parameters("list")))
		f.close()


	def set_parameters_by_name(self, var_name, tensor):
		with self._g.as_default():
			with self._s.as_default():
				if var_name in self._m.parameter_lists:
					self._m.parameter_lists[var_name].assign(tensor).eval()


	def set_parameters(self, lists):
		for i in lists:
			self.set_parameters_by_name(i, lists[i])


	def set_model(self, model, optimizer, **kwargs):
		if 'batch_size' not in kwargs:
			kwargs['batch_size'] = self.batch_size
		if 'batch_seq_size' not in kwargs:
			kwargs['batch_seq_size'] = self.batch_seq_size
		if 'negative_ent' not in kwargs:
			kwargs['negative_ent'] = self.negative_ent
		if 'negative_rel' not in kwargs:
			kwargs['negative_rel'] = self.negative_rel
		self._g = Graph()
		with self._g.as_default():
			self._s = Session()
			with self._s.as_default():
				initializer = xavier_initializer(uniform=True)
				with variable_scope("model", reuse=None, initializer=initializer):
					self._m = model(**kwargs)
					grads_and_vars = optimizer.compute_gradients(self._m.loss)
					self.train_op = optimizer.apply_gradients(grads_and_vars)
				self.saver = Saver()
				self._s.run(initialize_all_variables())


	def train(self, times=1, bern=True, log=None, workers=1):
		feed_dict = {
			self._m.batch_h: self.batch_h,
			self._m.batch_t: self.batch_t,
			self._m.batch_r: self.batch_r,
			self._m.batch_y: self.batch_y
		}
		sampling = self._l.bernSampling if bern else self._l.sampling
		with self._g.as_default():
			with self._s.as_default():
				self.parametername and self._restore(self.parametername)
				for t in range(times):
					loss = 0.0
					for batch in range(self.nbatches):
						sampling(self.batch_h_addr, self.batch_t_addr,
								self.batch_r_addr, self.batch_y_addr, self.batch_size,
								self.negative_ent, self.negative_rel, workers)
						loss += self._s.run([self.train_op, self._m.loss], feed_dict)[1]
					log and log(t, loss)
					if self.export_steps and t % self.export_steps == 0:
						self.graphname and self._save(self.graphname)
				self.graphname and self._save(self.graphname)
				print(self.batch_h, self.batch_t, self.batch_r, self.batch_y)


	def test(self, log=None):
		def step(h, t, r):
			feed_dict = {
				self._m.predict_h: h,
				self._m.predict_t: t,
				self._m.predict_r: r,
			}
			return self._s.run(self._m.predict, feed_dict)
		with self._g.as_default():
			with self._s.as_default():
				self.parametername and self._restore(self.parametername)
				total = self._l.getTestTotal()
				for t in range(total):
					self._l.getHeadBatch(self.test_h_addr, self.test_t_addr, self.test_r_addr)
					res = step(self.test_h, self.test_t, self.test_r)
					self._l.testHead(res.__array_interface__['data'][0])

					self._l.getTailBatch(self.test_h_addr, self.test_t_addr, self.test_r_addr)
					res = step(self.test_h, self.test_t, self.test_r)
					self._l.testTail(res.__array_interface__['data'][0])
					log and log(t, total)
				self._l.test()
