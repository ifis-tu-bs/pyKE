from pyke.dataset import Dataset
from pyke.embedding import Embedding
from pyke.models import DistMult

# Read the dataset
dataset = Dataset("./benchmarks/fb15k.nt")
embedding = Embedding(
    dataset,
    DistMult,
    folds=20,
    epochs=20,
    neg_ent=1,
    neg_rel=0,
    bern=False,
    workers=4,
    dimension=50,  # DistMult-specific
    weight=0.0001,  # DistMult-specific
)

# Train the model. It is saved in the process.
# TODO: Currently not working
embedding.train(prefix="./DistMult", post_epoch=print)

# Save the embedding to a JSON file
embedding.save_to_json("DistMult.json")
