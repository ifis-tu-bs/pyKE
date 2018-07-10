from openke import Dataset
from openke.models import TransE

# Read the dataset
ds = Dataset("./benchmarks/fb15k_tiny.nt")

# Configure parameters
folds = 20
neg_ent = 2
neg_rel = 0


# Set the knowledge embedding model class.
def model():
    return TransE(50, 1.0, ds.ent_count, ds.rel_count, batch_size=ds.size // folds, variants=1 + neg_rel + neg_ent)


# Train the model. It is saved in the process.
model, records = ds.train(
    model,
    folds=folds,
    epochs=20,
    batchkwargs=dict(
        neg_ent=neg_ent,
        neg_rel=neg_rel,
        bern=False,
        workers=4,
    ),
    post_epoch=print,
    prefix="./result",
)
print(records)

# test = Dataset("./benchmarks/FB15K/test2id.txt", E, R)
# print(test.meanrank(model, head=False, label=False))
