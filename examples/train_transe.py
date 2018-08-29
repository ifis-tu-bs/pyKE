from pyke.dataset import Dataset
from pyke.embedding import Embedding
from pyke.models import TransE

# Read the dataset
dataset = Dataset("./benchmarks/fb15k.nt")
embedding = Embedding(
    dataset,
    TransE,
    folds=20,
    epochs=20,
    neg_ent=1,
    neg_rel=0,
    bern=False,
    workers=4,
    dimension=50,  # TransE-specific
    margin=1.0,  # TransE-specific
)

# Train the model. It is saved in the process.
embedding.train(prefix="./TransE")

# Save the embedding to a JSON file
embedding.save_to_json("TransE.json")
