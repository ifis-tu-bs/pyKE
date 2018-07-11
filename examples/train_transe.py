from openke import Dataset
from openke.models import TransE

# Read the dataset
ds = Dataset("./benchmarks/fb15k.nt")

# Configure parameters
folds = 20
neg_ent = 2
neg_rel = 0


# Set the knowledge embedding model class.
def model():
    return TransE(50, 1.0, ds.ent_count, ds.rel_count, batch_size=ds.size // folds, variants=1 + neg_rel + neg_ent)


# Train the model. It is saved in the process.
model = ds.train(
    model,
    folds=folds,
    epochs=20,
    post_epoch=print,
    prefix="./TransE",
    neg_ent=neg_ent,
    neg_rel=neg_rel,
    bern=False,
    workers=4,
)

# Save the embedding to a JSON file
model.save_to_json("TransE.json")
