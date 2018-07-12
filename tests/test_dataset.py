import os
from unittest import TestCase

from pyke.dataset import Dataset


class TestDataset(TestCase):
    def setUp(self):
        self.nt_filename = "tests/resources/test.nt"

    def test_init(self):
        ds = Dataset(self.nt_filename, temp_dir="tests/tmp")
        tempdir = os.path.join(os.curdir, "tests/tmp", "da8c86a9fd1a62dc6bb6979203498a31")

        self.assertTrue(os.path.exists(os.path.join(tempdir, "entity2id.txt")))
        self.assertTrue(os.path.exists(os.path.join(tempdir, "relation2id.txt")))
        self.assertTrue(os.path.exists(os.path.join(tempdir, "train2id.txt")))
        self.assertEqual(ds.size, 8)
        self.assertEqual(len(ds), 8)
        self.assertEqual(ds.ent_count, 11)
        self.assertEqual(ds.rel_count, 6)
        self.assertEqual(ds.shape, (11, 6))
