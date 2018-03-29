from openke import Config
from openke.models import TransE

c = Config()

#   Input training files from benchmarks/FB15K/ folder.
with open("./benchmarks/FB15K/entity2id.txt") as f:
	E = int(f.readline())
with open("./benchmarks/FB15K/relation2id.txt") as f:
	R = int(f.readline())
c.init("./benchmarks/FB15K/train2id.txt", E, R, batch_count=100, negative_entities=1)

#   Models will be exported via tf.Saver() automatically.
c.set_export("./res/model.vec.tf", None, 1)

#   Set the knowledge embedding model
from tensorflow.python.training.gradient_descent import GradientDescentOptimizer as SGD
c.set_model(TransE, SGD(.001), hidden_size=50, margin=1.0)
#   Train the model.
c.train(10, bern=False, workers=4)

