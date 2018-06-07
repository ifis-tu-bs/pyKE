from openke import Dataset
from openke.models import TransE

#   Input training files from benchmarks/FB15K/ folder.
with open("./benchmarks/FB15K/entity2id.txt") as f:
    E = int(f.readline())
with open("./benchmarks/FB15K/relation2id.txt") as f:
    R = int(f.readline())

#   Read the dataset.
base = Dataset("./benchmarks/FB15K/train2id.txt", E, R)

#   Set the knowledge embedding model class.
model = TransE(50, 1.0, base.shape)

#   Train the model.
base.train(500, model, count=100, negatives=(1,0), bern=False, workers=4)

#   Save the result.
model.save("./result")
