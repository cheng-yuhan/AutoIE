import unittest
from autoie.pipeline import OneHotEmbedding,blocks
from autoie.searcher.core.hyperparameters import HyperParameters
import tensorflow as tf
import numpy as np

t1=tf.constant([[[1,2,3],[1,2,3],[2,3,4]]])
t2=tf.constant([[[1.1,1.2,1.3],[2.1,2.2,2.3]]])
print(t1.shape)
hp = HyperParameters()

class test_block (unittest.TestCase):
    def setUp(self) -> None:
        print("setup")

    def tearDown(self) -> None:
        print("teardown")

    def test_OneHotEmbedding(self):
        onehot = OneHotEmbedding(id_num=20,
                 embedding_dim=50)
        outputnode =onehot.build(hp= hp, inputs= t1)
        self.assertEqual(outputnode.shape[-1] ,50)

    def test_BiLSTM(self):
        bilstm = blocks.BiLSTM(units=10)
        outputnode = bilstm.build(hp=hp, inputs=t2)

        self.assertEqual(outputnode.shape[-1], 20)

    def test_Dense(self):
        dense = blocks.Dense(units = 50)
        outputnode = dense.build(hp=hp, inputs= t2)
        self.assertEqual(outputnode.shape[-1], 50)



if __name__ == '__main__':
    unittest.main()