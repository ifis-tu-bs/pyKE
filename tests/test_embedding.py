from unittest import TestCase

import numpy as np

from pyke.dataset import Dataset
from pyke.embedding import Embedding
from pyke.models import TransE


class TestEmbedding(TestCase):
    def setUp(self):
        self.nt_filename = "tests/resources/test.nt"
        self.dataset = Dataset(self.nt_filename, temp_dir="tests/tmp")

    def test_init_transe(self):
        Embedding(self.dataset, TransE)

    def test_train_transe(self):
        em = Embedding(self.dataset, TransE)
        em.train("./tests/tmp/TransE", continue_training=False)

    def test_get_ent_embeddings(self):
        em = Embedding(self.dataset, TransE)
        em.train("./tests/tmp/TransE", continue_training=False)
        embedding_values = em.get_ent_embeddings()
        self.assertIsInstance(embedding_values, np.ndarray)
        self.assertEqual(embedding_values.shape, (11, 50))
