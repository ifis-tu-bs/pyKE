#coding:utf-8
from numpy import zeros, int64, float32, bool_
from ctypes import c_void_p, c_int64, c_char_p, cdll, CFUNCTYPE, c_int
from json import dumps
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


	def init(self, filename,
			entities, relations,
			batchcount=None, batchsize=None,
			negative_entities=0, negative_relations=0):

		self.entTotal, self.relTotal = entities, relations
		self.open(filename)
		triples = self._l.getTrainTotal()
		self.trainTotal = triples
		self.baseshape = entities, relations

		if batchcount is not None:
			assert batchsize is None
			batchsize = triples // batchcount
		elif batchsize is not None:
			from math import ceil
			batchcount = ceil(triples / batchsize)
		else:
			batchsize, batchcount = 1, triples
		self.batchcount = batchcount

		N = negative_entities + negative_relations
		self.negative_ent, self.negative_rel = negative_entities, negative_relations
		self.batchshape = batchsize, 1 + N

		S = batchsize * (1 + N)
		self.batch_h = zeros(S, dtype=int64)
		self.batch_t = zeros(S, dtype=int64)
		self.batch_r = zeros(S, dtype=int64)
		self.batch_y = zeros(S, dtype=float32)
		self.batch_h_addr = c_array(self.batch_h)
		self.batch_t_addr = c_array(self.batch_t)
		self.batch_r_addr = c_array(self.batch_r)
		self.batch_y_addr = c_array(self.batch_y)


	def batch(self, bern=True, workers=1, seed=1):
		self._l.randReset(workers, seed)
		h, t, l, y = [c_array(x) for x in [self.batch_h, self.batch_t, self.batch_r, self.batch_y]]
		sampling = self._l.bernSampling if bern else self._l.sampling
		for _ in range(self.batchcount):
			sampling(h, t, l, y, self.batchshape[0], self.negative_ent, self.negative_rel, workers)
			yield self.batch_h, self.batch_t, self.batch_r, self.batch_y


	def train(self, model, epochs=1, bern=True, workers=1, seed=1,
			eachepoch=None, eachbatch=None):
		for epoch in range(epochs):
			loss = 0
			for i, batch in enumerate(self.batch(bern=bern, workers=workers, seed=seed)):
				loss += model.fit(*batch)
				eachbatch and eachbatch(batch, loss)
			eachepoch and eachepoch(epoch, loss)


	def open(self, filename):
		self._l.importTrainFiles(c_char_p(bytes(filename, 'utf-8')),
				self.entTotal, self.relTotal)


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
