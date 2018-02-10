#coding:utf-8
from tensorflow import (get_variable as var,
                        reduce_sum as sum,
                        reduce_mean as mean,
                        maximum as max,
                        matmul, squeeze, nn)
from tensorflow.contrib.layers import xavier_initializer as xavier
at = nn.embedding_lookup
from . import Model

class TransR(Model):


	def embedding_def(self):

		E, R = self.entities, self.relations
		dE, dR = self.entitysize, self.relationsize

		self.ent_embeddings = var("ent_embeddings", [E, dE, 1],
				initializer=xavier(uniform=False))
		self.rel_embeddings = var("rel_embeddings", [R, dR, 1],
				initializer=xavier(uniform=False))
		self.transfer_matrix = var("transfer_matrix", [R, dR, dE],
				initializer=xavier(uniform=False))
		self.parameter_lists = {
				"ent_embeddings": self.ent_embeddings,
				"rel_embeddings": self.rel_embeddings,
				"transfer_matrix": self.transfer_matrix}


	def loss_def(self):

		dE, dR = self.entitysize, self.relationsize

		ph, pt, pr = self.get_positive_instance(in_batch=True) # [B, 1]
		nh, nt, nr = self.get_negative_instance(in_batch=True) # [B, N]

		ph = at(self.ent_embeddings, ph) # [B, 1, dE, 1]
		pt = at(self.ent_embeddings, pt) # [B, 1, dE, 1]
		pm = at(self.transfer_matrix, pr) # [B, 1, dR, dE]
		pr = at(self.rel_embeddings, pr) # [B, 1, dR, 1]
		nh = at(self.ent_embeddings, nh) # [B, N, dE, 1]
		nt = at(self.ent_embeddings, nt) # [B, N, dE, 1]
		nm = at(self.transfer_matrix, nr) # [B, N, dR, dE]
		nr = at(self.rel_embeddings, nr) # [B, N, dR, 1]

		ph = matmul(pm, ph) # [B, 1, dR, 1]
		pt = matmul(pm, pt) # [B, 1, dR, 1]
		nh = matmul(nm, nh) # [B, N, dR, 1]
		nt = matmul(nm, nt) # [B, N, dR, 1]

		ps = squeeze(ph + pr - pt, [1,3]) # [B, dR]
		ns = squeeze(nh + nr - nt, [3]) # [B, N, dR]

		ps = sum(ps, 1) # [dR]
		ns = sum(mean(ns, 1), 1) # [dR]

		self.loss = sum(max(ps - ns + self.margin, 0)) # []


	def predict_def(self):

		dE, dR = self.entitysize, self.relationsize

		h, t, r = self.get_predict_instance() # [B]

		h = at(self.ent_embeddings, h) # [B, dE, 1]
		t = at(self.ent_embeddings, t) # [B, dE, 1]
		M = at(self.transfer_matrix, r) # [B, dR, dE]
		r = at(self.rel_embeddings, r) # [B, dR, 1]

		h = matmul(M, h) # [B, dR, 1]
		t = matmul(M, t) # [B, dR, 1]

		self.predict = sum(squeeze(h + r - t, [2]), 1, keep_dims=True) # [B, 1]


	def __init__(self, **config):
		self.entities = config['entTotal']
		self.relations = config['relTotal']
		self.entitysize = config['ent_size']
		self.relationsize = config['rel_size']
		self.margin = config['margin']
		super().__init__(**config)
