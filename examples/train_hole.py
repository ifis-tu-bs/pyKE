from openke import Dataset
from openke.models import HolE as Model

#   Input training files from benchmarks/FB15K/ folder.
with open("./benchmarks/FB15K/entity2id.txt") as f:
    E = int(f.readline())
with open("./benchmarks/FB15K/relation2id.txt") as f:
    R = int(f.readline())

#   Read the dataset.
base = Dataset("./benchmarks/FB15K/train2id.txt", E, R)

#   Set the knowledge embedding model class.
model = Model(50, 1.0, base.shape, batchshape=(len(base) // 500, 2))

#   Train the model.
base.train(500, model, count=100, negatives=(1,0), bern=False, workers=4)

#   Input testing files from benchmarks/FB15K/.
test = Dataset("./benchmarks/FB15K/test2id.txt")

#   Perform a test.
print(test.meanrank(model, head=False, label=False))
