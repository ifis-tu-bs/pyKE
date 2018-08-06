"""
This module is for testing purposes only.
It compared specific prediction values of the original OpenKE to the predictions made by this library.
It required an model trained with the original OpenKE.
"""
from pyke.dataset import Dataset
from pyke.embedding import Embedding
from pyke.models import TransE

dataset = Dataset("./benchmarks/fb15k.nt")
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

train_triples = [
    ([0, 1, 0], 0.07093484),
    ([2, 3, 1], 0.07182518),
    ([4, 5, 2], 0.11428239),
    ([10, 11, 5], 0.07594997),
]
test_triples = [
    ([453, 1347, 37], 0.09935094),
    ([147, 307, 60], 0.07316422),
]
incorrect_triples = [
    ([10, 1500, 90], 0.12321988),
    ([20, 1700, 60], 0.12572554),
]

triples = {
    "Train triples": train_triples,
    "Test triples": test_triples,
    "Incorrect triples": incorrect_triples,
}

for k, v in triples.items():
    print(f"{10 * '*'} {k.upper()} {10 * '*'}")
    print(f"Triple\t\tScore (pyke)\tScore (OpenKE)\tDeviation")
    for t in v:
        predict = embedding.predict_single(t[0][0], t[0][1], t[0][2])[0]
        original = t[1]
        deviation = abs(predict - original) * 100.0 / original
        print(f"{t[0]}\t{predict:.4f}\t\t{original:.4f}\t\t\t{deviation:.2f} %")
