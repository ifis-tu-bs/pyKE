import os
from unittest import TestCase

from openke import Dataset


class TestDataset(TestCase):
    def setUp(self):
        self.nt_filename = "tests/resources/test.nt"

    def test_init(self):
        print(os.curdir)
        d = Dataset(self.nt_filename)
