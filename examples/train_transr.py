from openke import Dataset
from pyke.models import TransR as Model

#   Input training files from benchmarks/FB15K/ folder.
with open("./benchmarks/FB15K/entity2id.txt") as f:
    E = int(f.readline())
with open("./benchmarks/FB15K/relation2id.txt") as f:
    R = int(f.readline())

#   Read the dataset.
base = Dataset("./benchmarks/FB15K/train2id.txt", E, R)

#   Set the knowledge embedding model class.
from tensorflow.python.training.gradient_descent import GradientDescentOptimizer as optimizer
model = lambda: Model(50, 10, 1., base.shape, batchshape=(len(base) // 20, 2),
		optimizer=optimizer(.0001))

#   Train the model.
def ee(epoch, loss):
	from math import isnan
	if isnan(loss):
		raise TypeError(loss)
	print(epoch, loss)
model, record = base.train(model, folds=20, epochs=20,
                           batchkwargs={'negatives':(1,0), 'bern':False, 'workers':4},
                           post_epoch=ee, prefix="./result")
print(record)

#   Input testing files from benchmarks/FB15K/.
#   Perform a test
test = Dataset("./benchmarks/FB15K/test2id.txt", E, R)
print(test.meanrank(model, head=False, label=False))
