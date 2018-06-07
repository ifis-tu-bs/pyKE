#coding:utf-8
class Dataset(object):
	''' Dataset'''


	def __init__(self, filename,
			entities, relations,
			library='./libopenke.so'):
		''' Creates a new dataset from a table in a file.
At the moment, no two datasets can be open at the same time!
		'''

		global _l
		from ctypes import c_char_p
		self._l = _l[library]

		self._l.importTrainFiles(c_char_p(bytes(filename, 'utf-8')),
				entities, relations)
		self.size = self._l.getTrainTotal()
		self.shape = entities, relations


	def batch(self, count,
			negatives=(0,0), bern=True, workers=1, seed=1):
		''' Separates the dataset into nearly equal parts.
Iterates over all parts, each time yielding four arrays.
This method can be used for cross-validation as shown below:
```
assert numpy, base, folds, epochs, Model

for i in range(folds):

	model = Model()

	for _ in range(epochs):
		for j, b in enumerate(base.batch(folds)):
			if i != j:
				model.fit(*b)

	for j, b in enumerate(base.batch(folds)):
		if i == j:
			break
	score = model.predict(*b[:3])
```
		'''

		from numpy import float32, int64, zeros

		size = self.size // count
		S = size * (1 + sum(negatives[:2]))
		types = [int64, int64, int64, float32]
		batch = [zeros(S, dtype=t) for t in types]
		h, t, l, y = [_carray(x) for x in batch]

		self._l.randReset(workers, seed)
		sampling = self._l.bernSampling if bern else self._l.sampling
		for _ in range(count):
			print(h,t,l,y,S)
			sampling(h, t, l, y, size, negatives[0], negatives[1], workers)
			yield batch


	def train(self, model, epochs=1, bern=True, workers=1, seed=1,
			eachepoch=None, eachbatch=None):
		''' A simple training algorithm over the whole set. '''

		for epoch in range(epochs):
			loss = 0
			for i, batch in enumerate(self.batch(bern=bern, workers=workers, seed=seed)):
				loss += model.fit(*batch)
				eachbatch and eachbatch(batch, loss)
			eachepoch and eachepoch(epoch, loss)


	def meanrank(self, model, head=True, tail=True, label=True):
		''' Computes the mean rank of link prediction of its entire content.
Returns floating values between 0 and size-1 where lower results denote better models.
The return value consists of up to three values, one for each of the columns 'head', 'tail' and 'label'.
	Arguments:
model - The to-be-tested embedding model.
head, tail, label - Truthvalues denoting whether or not the respecting column should be tested.
	Note:
This test filters only 'false' facts that evaluate better than the question.
See `openke.meanrank` for the unfiltered, or 'raw', version.'''
		h,t,l,_ = next(self.batch())
		def rank(x, h, t, l):
			y, z = self.query(h, t, l), model.predict(h, t, l)
			return sum(1 for i in range(self.shape[0]) if z[i] < z[x] and not y[i])
		ranks = [
				(rank(h[i], None, t[i], l[i]) for i in range(self.size)) if head else None,
				(rank(t[i], h[i], None, l[i]) for i in range(self.size)) if tail else None,
				(rank(l[i], h[i], t[i], None) for i in range(self.size)) if label else None]
		return [sum(i) / self.size for i in ranks if i is not None]


	def query(self, head, tail, relation):
		''' Checks which facts are stored in the entire dataset. '''
		from numpy import bool_, zeros
		if head is None:
			if tail is None:
				if relation is None:
					raise NotImplementedError('querying everything')
				raise NotImplementedError('querying full relation')
			if relation is None:
				raise NotImplementedError('querying full head')
			heads = zeros(self.shape[0], bool_)
			self._l.query_head(_carray(heads), tail, relation)
			return heads
		if tail is None:
			if relation is None:
				raise NotImplementedError('querying full tail')
			tails = zeros(self.shape[0], bool_)
			self._l.query_tail(head, _carray(tails), relation)
			return tails
		if relation is None:
			relations = zeros(self.shape[1], bool_)
			self._l.query_rel(head, tail, _carray(relations))
			return relations
		raise NotImplementedError('querying single facts')
# FIXME backwards-compatibility
Config = Dataset


def meanrank(h, t, l, model, head=True, tail=True, label=True):
	''' Computes the mean rank of link prediction of its entire content.
Returns a value between 0 and size-1 where lower results denote better models.
The return value consists of up to three values, one for each of the columns 'head', 'tail' and 'label'.
	Arguments:
h, t, l - Integral arrays of equal shape describing relational questions.
model - The to-be-tested embedding model.
head, tail, label - Truthvalues denoting whether or not the respecting column should be tested.
	Note:
This test ignores whether or not statements that evaluate better than the question are known truths or not.
See `openke.Database.meanrank` for the filtered version.'''
	def rank(x, h, t, l):
		z = model.predict(h, t, l)
		return sum(1 for i in range(self.shape[0]) if z[i] < z[x])
	ranks = [
			(rank(h[i], None, t[i], l[i]) for i in range(len(h))) if head else None,
			(rank(t[i], h[i], None, l[i]) for i in range(len(t))) if tail else None,
			(rank(l[i], h[i], t[i], None) for i in range(len(l))) if label else None]
	return [sum(i) / self.size for i in ranks if i is not None]


def _carray(a):
	return a.__array_interface__['data'][0]


class _Library:
	''' Manages the connection to the library. '''
	def __init__(self):
		self.__dict = dict()
	def __getitem__(self, key):
		if key in self.__dict:
			return self.__dict[key]
		from ctypes import c_void_p as p, c_int64 as i, cdll
		l = cdll.LoadLibrary(key)
		l.sampling.argtypes = [p, p, p, p, i, i, i, i]
		l.bernSampling.argtypes = l.sampling.argtypes
		l.query_head.argtypes = [p, i, i]
		l.query_tail.argtypes = [i, p, i]
		l.query_rel.argtypes = [i, i, p]
		l.importTrainFiles.argtypes = [p, i, i]
		l.randReset.argtypes = [i, i]
		self.__dict[key] = l
		return l
_l = _Library()
