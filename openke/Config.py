#coding:utf-8
from numpy import zeros, int64, float32, bool_
from ctypes import c_void_p, c_int64, c_char_p, cdll, CFUNCTYPE, c_int
def c_str(s):
	return c_char_p(bytes(s, 'utf-8'))
def c_array(a):
	return a.__array_interface__['data'][0]


class Config(object):


	def __init__(self, library='./libopenke.so'):
		self._l = cdll.LoadLibrary(library)
		self._l.sampling.argtypes = [c_void_p, c_void_p, c_void_p, c_void_p, c_int64, c_int64, c_int64, c_int64]
		self._l.bernSampling.argtypes = self._l.sampling.argtypes
		self._l.query_head.argtypes = [c_void_p, c_int64, c_int64]
		self._l.query_tail.argtypes = [c_int64, c_void_p, c_int64]
		self._l.query_rel.argtypes = [c_int64, c_int64, c_void_p]
		self._l.importTrainFiles.argtypes = [c_void_p, c_int64, c_int64]
		self._l.randReset.argtypes = [c_int64, c_int64]
		self.export_steps = 0


	def init(self, filename, entities, relations, batch_count=1,
			negative_entities=0, negative_relations=0):
		self.negative_ent, self.negative_rel = negative_entities, negative_relations
		self.entTotal, self.relTotal = entities, relations
		self._l.importTrainFiles(c_str(filename), entities, relations)
		self.trainTotal = self._l.getTrainTotal()
		self.nbatches = batch_count
		self.batch_size = self.trainTotal // batch_count

		self.batch_seq_size = self.batch_size * (1 + self.negative_ent + self.negative_rel)
		self.batch_h = zeros(self.batch_seq_size, dtype=int64)
		self.batch_t = zeros(self.batch_seq_size, dtype=int64)
		self.batch_r = zeros(self.batch_seq_size, dtype=int64)
		self.batch_y = zeros(self.batch_seq_size, dtype=float32)
		self.batch_h_addr = c_array(self.batch_h)
		self.batch_t_addr = c_array(self.batch_t)
		self.batch_r_addr = c_array(self.batch_r)
		self.batch_y_addr = c_array(self.batch_y)


	def set_model(self, model, optimizer, **kwargs):
		'''Prepares the model by adding knowledgebase-based hyperparameters.'''
		if 'batch_size' not in kwargs:
			kwargs['batch_size'] = self.batch_size
		if 'batch_seq_size' not in kwargs:
			kwargs['batch_seq_size'] = self.batch_seq_size
		if 'negative_ent' not in kwargs:
			kwargs['negative_ent'] = self.negative_ent
		if 'negative_rel' not in kwargs:
			kwargs['negative_rel'] = self.negative_rel
		return = model(**kwargs)


	def batch(self, seed=1, bern=True, workers=1):
		'''Provides the knowledge in batches.
Includes negative sampling if set.'''
		self._l.randReset(workers, seed)
		sampling = self._l.bernSampling if bern else self._l.sampling
		for batch in range(self.nbatches)
			sampling(self.batch_h_addr, self.batch_t_addr,
					self.batch_r_addr, self.batch_y_addr, self.batch_size,
					self.negative_ent, self.negative_rel, workers)
			yield self.batch_h, self.batch_t, self.batch_r, self.batch_y


	def train(self, model, epochs=1, save_steps=0, seed=1, bern=True, workers=1, log=None):
		for epoch in range(epochs):
			loss = 0.0
			for h, t, l, y in self.batch(seed, bern, workers):
				loss += model.train(h, t, l, y)[1]
			log and log(epoch, loss)
			if save_steps and epoch % save_steps == 0:
				model.save()
		model.save()


	def open(self, filename):
		'''Replaces the currently loaded knowledge with another file's content.'''
		self._l.importTrainFiles(c_str(filename), self.entTotal, self.relTotal)


	def query(self, head, tail, relation):
		if head is None:
			if tail is None:
				if relation is None:
					raise NotImplementedError('querying everything')
				raise NotImplementedError('querying full relation')
			if relation is None:
				raise NotImplementedError('querying full head')
			heads = zeros(self.entTotal, bool_)
			self._l.query_head(c_array(heads), tail, relation)
			return heads
		if tail is None:
			if relation is None:
				raise NotImplementedError('querying full tail')
			tails = zeros(self.entTotal, bool_)
			self._l.query_tail(head, c_array(tails), relation)
			return tails
		if relation is None:
			relations = zeros(self.relTotal, bool_)
			self._l.query_rel(head, tail, c_array(relations))
			return relations
		raise NotImplementedError('querying single facts')
