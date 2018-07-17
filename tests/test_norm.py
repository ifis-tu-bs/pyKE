from unittest import TestCase

import tensorflow as tf

from pyke.norm import l1, l2


class TestNorm(TestCase):
    def setUp(self):
        self.session = tf.Session()

    def test_l1(self):
        x = tf.constant([-1, 2, 3])
        y = tf.constant([-1, 2, 3, 8, 4.5, -10])
        norm_x = l1(x)
        norm_y = l1(y)
        self.assertEqual(self.session.run(norm_x), 6)
        self.assertEqual(self.session.run(norm_y), 28.5)

    def test_l1_multi(self):
        x = tf.constant([[-1, 2, -3], [-7, 8, 9]])
        norm_x = l1(x)
        norm = self.session.run(norm_x)
        self.assertEqual(norm[0], 6)
        self.assertEqual(norm[1], 24)

    def test_l2(self):
        x = tf.constant([-1, 2, 3], tf.float64)
        y = tf.constant([-1, 2, 3, 8, 4.5, -10], tf.float64)
        norm_x = l2(x)
        norm_y = l2(y)
        self.assertAlmostEqual(self.session.run(norm_x), 3.741657387)
        self.assertAlmostEqual(self.session.run(norm_y), 14.08012784)
