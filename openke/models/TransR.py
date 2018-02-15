#coding:utf-8
from tensorflow import (get_variable as var,
                        reduce_sum as sum,
                        reduce_mean as mean,
                        maximum as max,
                        matmul, squeeze, nn, rank)
from tensorflow.contrib.layers import xavier_initializer as xavier
at = nn.embedding_lookup
from . import Model

class TransR(Model):


	def _embeddings(self, h, t, r):
		'''The term to embed triples.'''

		h = at(self.ent_embeddings, h) # [.,D,1]
		t = at(self.ent_embeddings, t) # [.,D,1]
		m = at(self.transfer_matrix, r) # [.,K,D]
		r = at(self.rel_embeddings, r) # [.,K,1]
		h = matmul(m, h) # [.,K,1]
		t = matmul(m, t) # [.,K,1]
		return squeeze(h + r - t, [rank(r)-1]) # [.,K]


	def embedding_def(self):
		'''Initializes the variables of the model.'''

		E, R = self.entities, self.relations
		D, K = self.entitysize, self.relationsize

		self.ent_embeddings = var("ent_embeddings", [E, D, 1],
				initializer=xavier(uniform=False))
		self.rel_embeddings = var("rel_embeddings", [R, K, 1],
				initializer=xavier(uniform=False))
		self.transfer_matrix = var("transfer_matrix", [R, K, D],
				initializer=xavier(uniform=False))
		self.parameter_lists = {
				"ent_embeddings": self.ent_embeddings,
				"rel_embeddings": self.rel_embeddings,
				"transfer_matrix": self.transfer_matrix}


	def loss_def(self):
		'''Initializes the loss function.'''

		def scores(h, t, r):
			e = self._embeddings(h, t, r) # [B,N,K]
			return sum(mean(p, 1), 1) # [B]

		p = scores(*self.get_positive_instance(in_batch=True)) # [B]
		n = scores(*self.get_negative_instance(in_batch=True)) # [B]

		self.loss = sum(max(ps - ns + self.margin, 0)) # []


	def predict_def(self):
		'''Initializes the prediction function.'''

		self.embed = self._embeddings(*self.get_predict_instance()) # [B,K]

		self.predict = sum(self.embed, 1) # [B]


	def __init__(self, **config):
		self.entities = config['entTotal']
		self.relations = config['relTotal']
		self.entitysize = config['ent_size']
		self.relationsize = config['rel_size']
		self.margin = config['margin']
		super().__init__(**config)
