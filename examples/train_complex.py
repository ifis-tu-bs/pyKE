from openke import Dataset
from pyke.models import ComplEx as Model

#   Input training files from benchmarks/FB15K/ folder.
with open("./benchmarks/FB15K/entity2id.txt") as f:
    E = int(f.readline())
with open("./benchmarks/FB15K/relation2id.txt") as f:
    R = int(f.readline())

#   Read the dataset.
base = Dataset("./benchmarks/FB15K/train2id.txt", E, R)

#   Set the knowledge embedding model class.
model = lambda: Model(50, .0001, base.shape, batchshape=(len(base) // 20, 2))

#   Train the model.
model, record = base.train(model, folds=20, epochs=50,
                           batchkwargs={'negatives':(1,0), 'bern':False, 'workers':4},
                           post_epoch=print, prefix="./result")
print(record)

#   Input testing files from benchmarks/FB15K/.
test = Dataset("./benchmarks/FB15K/test2id.txt")

#   Perform a test.
print(test.meanrank(model, head=False, label=False))
