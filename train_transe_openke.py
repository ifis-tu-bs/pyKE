from pyke.dataset import Dataset
from pyke.embedding import Embedding
from pyke.models import TransE

# Read the dataset
dataset = Dataset("./benchmarks/fb15k.nt")
embedding = Embedding(
    dataset,
    TransE,
    folds=100,
    epochs=1000,
    neg_ent=1,
    neg_rel=0,
    bern=False,
    workers=8,
    dimension=100,
    margin=1.0,
    learning_rate=0.001,
)

# Train the model. It is saved in the process.
embedding.train(prefix="./TransEOpenKE", post_epoch=print)

# Save the embedding to a JSON file
embedding.save_to_json("TransEOpenKE.json")
