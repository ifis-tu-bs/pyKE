"""
This module is for testing purposes only.
It compared specific prediction values of the original OpenKE to the predictions made by this library.
It required an model trained with the original OpenKE.
"""
from pyke.dataset import Dataset
from pyke.embedding import Embedding
from pyke.models import TransE

dataset = Dataset("./benchmarks/fb15k.nt", generate_valid_test=True)
embedding = Embedding(
    dataset,
    TransE,
    folds=20,
    epochs=1,
    neg_ent=1,
    neg_rel=0,
    bern=False,
    workers=8,
    dimension=100,
    margin=1.0,
    learning_rate=0.01,
)

embedding.restore(prefix="./openke/model.vec.tf")  # Load the original model trained by OpenKE

print(embedding.meanrank())
