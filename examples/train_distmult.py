from pyke.dataset import Dataset
from pyke.embedding import Embedding
from pyke.models import DistMult

# Read the dataset
dataset = Dataset("./benchmarks/fb15k.nt")
embedding = Embedding(
    dataset,
    DistMult,
    folds=100,
    epochs=500,
    neg_ent=1,
    neg_rel=0,
    bern=False,
    workers=4,
    dimension=50,  # DistMult-specific
    weight=0.0001,  # DistMult-specific
    learning_rate=0.1,
    optimizer="Adagrad",
)

# Train the model. It is saved in the process.
# TODO: Currently not working. Loss does not decrease
embedding.train(prefix="./DistMult", continue_training=False)

# Save the embedding to a JSON file
embedding.save_to_json("DistMult.json")
