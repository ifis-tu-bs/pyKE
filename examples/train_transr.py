from pyke.dataset import Dataset
from pyke.embedding import Embedding
from pyke.models import TransR

# Read the dataset
dataset = Dataset("./benchmarks/fb15k.nt")
embedding = Embedding(
    dataset,
    TransR,
    folds=20,
    epochs=20,
    neg_ent=1,
    neg_rel=0,
    bern=False,
    workers=4,
    ent_dim=50,  # TransR-specific
    rel_dim=10,  # TransR-specific
    margin=1.0,  # TransR-specific
)

# Train the model. It is saved in the process.
# TODO: Currently not working
embedding.train(prefix="./TransR")

# Save the embedding to a JSON file
embedding.save_to_json("TransR.json")
