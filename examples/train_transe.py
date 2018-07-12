from pyke.dataset import Dataset
from pyke.embedding import Embedding
from pyke.models import TransE

# Configure parameters
folds = 20
neg_ent = 2
neg_rel = 0

# Read the dataset
ds = Dataset("./benchmarks/fb15k.nt")
em = Embedding(
    ds,
    TransE,
    folds=folds,
    epochs=20,
    neg_ent=neg_ent,
    neg_rel=neg_rel,
    bern=False,
    workers=4,
)


# Set the knowledge embedding model class.
def model():
    return TransE(50, 1.0, ds.ent_count, ds.rel_count, batch_size=ds.size // folds, variants=1 + neg_rel + neg_ent)


# Train the model. It is saved in the process.
em.train(
    model,
    post_epoch=print,
    prefix="./TransE",
)

# Save the embedding to a JSON file
em.save_to_json("TransE.json")
