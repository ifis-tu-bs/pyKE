from pyke.dataset import Dataset
from pyke.embedding import Embedding
from pyke.models import TransH

# Read the dataset
dataset = Dataset("./benchmarks/fb15k.nt")
embedding = Embedding(
    dataset,
    TransH,
    folds=20,
    epochs=20,
    neg_ent=1,
    neg_rel=0,
    bern=False,
    workers=4,
    dimension=50,  # TransH-specific
    margin=1.0,  # TransH-specific
)

# Train the model. It is saved in the process.
embedding.train(prefix="./TransH")

# Save the embedding to a JSON file
embedding.save_to_json("TransH.json")
