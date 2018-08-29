from pyke.dataset import Dataset
from pyke.embedding import Embedding
from pyke.models import TransE

# Read the dataset
dataset = Dataset("./benchmarks/fb15k.nt")
embedding = Embedding(
    dataset,
    TransE,
    folds=100,
    epochs=1000,
    neg_ent=1,
    neg_rel=0,
    bern=False,
    workers=8,
    dimension=100,
    margin=1.0,
    learning_rate=0.001,
)

embedding.restore("./TransEOpenKE")

print(embedding.meanrank())
