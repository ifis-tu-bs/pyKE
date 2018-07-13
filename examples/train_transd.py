from pyke.dataset import Dataset
from pyke.embedding import Embedding
from pyke.models import TransD

# Read the dataset
dataset = Dataset("./benchmarks/fb15k.nt")
embedding = Embedding(
    dataset,
    TransD,
    folds=20,
    epochs=20,
    neg_ent=1,
    neg_rel=0,
    bern=False,
    workers=4,
    dimension=50,  # TransD-specific
    margin=1.0,  # TransD-specific
)

# Train the model. It is saved in the process.
# TODO: Currently not working
embedding.train(prefix="./TransD", post_epoch=print)

# Save the embedding to a JSON file
embedding.save_to_json("TransD.json")
