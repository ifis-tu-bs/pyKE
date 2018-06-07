from openke import Dataset
from openke.models import restore

#   Input training files from benchmarks/FB15K/.
with open("./benchmarks/FB15K/entity2id.txt") as f:
    E = int(f.readline())
with open("./benchmarks/FB15K/relation2id.txt") as f:
    R = int(f.readline())

#   Read the dataset.
base = Dataset("./benchmarks/FB15K/train2id.txt", E, R)

#   Set the knowledge embedding model class.
model = TransE(50, 1.0, base.shape)
model.restore("./result")

#   Input testing files from benchmarks/FB15K/.
test = Dataset("./benchmarks/FB15K/test2id.txt")

print(test.meanrank(model, head=False, label=False))
