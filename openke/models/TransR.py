#coding:utf-8
from tensorflow import (get_variable as var,
                        reduce_sum as sum,
                        reduce_mean as mean,
                        maximum as max,
                        matmul, squeeze, nn, rank)
from tensorflow.contrib.layers import xavier_initializer as xavier
at = nn.embedding_lookup
from .Base import ModelClass

class TransR(ModelClass):


	def _embeddings(self, h, t, r):
		'''The term to embed triples.'''

		h = at(self.ent_embeddings, h) # [.,d,1]
		t = at(self.ent_embeddings, t) # [.,d,1]
		m = at(self.transfer_matrix, r) # [.,k,d]
		r = at(self.rel_embeddings, r) # [.,k,1]
		h = matmul(m, h) # [.,k,1]
		t = matmul(m, t) # [.,k,1]
		return squeeze(h + r - t, [-1]) # [.,k]


	def embedding_def(self):
		'''Initializes the variables of the model.'''

		e, r = self.entities, self.relations
		d, k = self.entitysize, self.relationsize

		self.ent_embeddings = var("ent_embeddings", [e, d, 1],
				initializer=xavier(uniform=False))
		self.rel_embeddings = var("rel_embeddings", [r, k, 1],
				initializer=xavier(uniform=False))
		self.transfer_matrix = var("transfer_matrix", [r, k, d],
				initializer=xavier(uniform=False))
		self.parameter_lists = {
				"ent_embeddings": self.ent_embeddings,
				"rel_embeddings": self.rel_embeddings,
				"transfer_matrix": self.transfer_matrix}


	def loss_def(self):
		'''Initializes the loss function.'''

		def scores(h, t, r):
			e = self._embeddings(h, t, r) # [b,n,k]
			return mean(sum(e, 2), 1) # [b]

		p = scores(*self.get_positive_instance(in_batch=True)) # [b]
		n = scores(*self.get_negative_instance(in_batch=True)) # [b]

		self.loss = sum(max(p - n + self.margin, 0)) # []


	def predict_def(self):
		'''Initializes the prediction function.'''

		self.embed = self._embeddings(*self.get_predict_instance()) # [b,k]

		self.predict = sum(self.embed, 1) # [b]


	def __init__(self, **config):
		self.entities = config['entTotal']
		self.relations = config['relTotal']
		self.entitysize = config['ent_size']
		self.relationsize = config['rel_size']
		self.margin = config['margin']
		super().__init__(**config)
