#coding:utf-8
from tensorflow import (get_variable as var,
                        reduce_sum as sum,
                        reduce_mean as mean,
                        maximum as max,
                        nn)
from tensorflow.contrib.layers import xavier_initializer as xavier
at = nn.embedding_lookup
from . import Model


class TransE(Model):


	def embedding_def(self):

		d, dE, dR = self.hiddensize, self.entities, self.relations

		self.ent_embeddings = var("ent_embeddings", [dE, d],
				initializer=xavier(uniform = False))
		self.rel_embeddings = var("rel_embeddings", [dR, d],
				initializer=xavier(uniform = False))
		self.parameter_lists = {
				"ent_embeddings": self.ent_embeddings,
				"rel_embeddings": self.rel_embeddings}


	def loss_def(self):

		#To get positive triples and negative triples for training
		ph, pt, pr = self.get_positive_instance(in_batch=True) # [B, 1]
		nh, nt, nr = self.get_negative_instance(in_batch=True) # [B, N]

		#Embedding entities and relations of triples
		ph = at(self.ent_embeddings, ph)
		pt = at(self.ent_embeddings, pt)
		pr = at(self.rel_embeddings, pr)
		nh = at(self.ent_embeddings, nh)
		nt = at(self.ent_embeddings, nt)
		nr = at(self.rel_embeddings, nr)

		#Calculating score functions for all positive triples and negative triples
		sp = sum(mean(abs(ph - pt + pr), 1, keep_dims=False), 1, keep_dims=True)
		sn = sum(mean(abs(nh - nt + nr), 1, keep_dims=False), 1, keep_dims=True)

		#Calculating loss to get what the framework will optimize
		self.loss = sum(max(sp - sn + self.margin, 0))


	def predict_def(self):

		h, t, r = self.get_predict_instance()

		eh = at(self.ent_embeddings, h)
		et = at(self.ent_embeddings, t)
		er = at(self.rel_embeddings, r)

		self.predict = mean(abs(eh - et + er), 1, keep_dims=False)


	def __init__(self, **config):
		self.entities = config['entTotal']
		self.relations = config['relTotal']
		self.hiddensize = config['hidden_size']
		self.margin = config['margin']
		super().__init__(**config)
