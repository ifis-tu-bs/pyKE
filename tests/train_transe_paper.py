from pyke.dataset import Dataset
from pyke.embedding import Embedding
from pyke.models import TransE

# Read the dataset
dataset = Dataset("./benchmarks/fb15k.nt")
embedding = Embedding(
    dataset,
    TransE,
    folds=20,
    epochs=50,
    neg_ent=1,
    neg_rel=0,
    bern=False,
    workers=8,
    dimension=50,  # TransE paper
    margin=1.0,  # TransE paper
    learning_rate=0.01,  # TransE paper
)

# Train the model. It is saved in the process.
embedding.train(prefix="./TransE", post_epoch=print)

# Save the embedding to a JSON file
embedding.save_to_json("TransE.json")

meanrank = embedding.meanrank(batch_count=1)
print(meanrank)
