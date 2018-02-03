#coding:utf-8
from tensorflow import (get_variable as var,
                        reduce_sum as sum,
                        reduce_mean as mean,
                        maximum as max,
                        matmul,
                        nn)
from tensorflow.contrib.layers import xavier_initializer as xavier
at = nn.embedding_lookup
from . import Model
#FIXME replace matmul with PEP465 @-operator when upgrading to Python 3.5


class RESCAL(Model):


	def embedding_def(self):
		E, R, D = self.entities, self.relations, self.hiddensize

		self.ent_embeddings = var("ent_embeddings", [E, D],
				initializer=xavier(uniform=False))
		self.rel_matrices = var("rel_matrices", [R, D * D],
				initializer=xavier(uniform=False))
		self.parameter_lists = {
			"ent_embeddings": self.ent_embeddings,
			"rel_matrices": self.rel_matrices}


	def loss_def(self):
		#	each (batchsize,1)
		ph, pt, pr = self.get_positive_instance(in_batch=True)
		#	each (batchsize,negatives)
		nh, nt, nr = self.get_negative_instance(in_batch=True)

		ph = at(self.ent_embeddings, ph)
		pt = at(self.ent_embeddings, pt)
		pr = at(self.rel_matrices, pr)
		nh = at(self.ent_embeddings, nh)
		nt = at(self.ent_embeddings, nt)
		nr = at(self.rel_matrices, nr)

		#	(batchsize,1,hiddensize)
		sp = ph * matmul(pr, pt)
		#	(batchsize,negatives,hiddensize)
		sn = nh * matmul(nr, nt)

		#	(batch_size, 1)
		sp = sum(mean(sp, 1, keep_dims=False), 1, keep_dims=True)
		#	(batch_size, 1)
		sn = sum(mean(sn, 1, keep_dims=False), 1, keep_dims=True)

		#Calculating loss to get what the framework will optimize
		self.loss = sum(max(0, sn - sp + self.margin))
	
	def predict_def(self):
		h, t, r = self.get_predict_instance()

		h = at(self.ent_embeddings, h)
		t = at(self.ent_embeddings, t)
		r = at(self.rel_matrices, r)

		self.predict = -sum(h * matmul(r, t), 1, keep_dims=False)


	def __init__(self, **config):
		self.entities = config['entTotal']
		self.relations = config['relTotal']
		self.hiddensize = config['hidden_size']
		self.margin = config['margin']
		super().__init__(**config)
