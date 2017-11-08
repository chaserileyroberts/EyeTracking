import pytest
import tensorflow as tf
import numpy as np
from GAN.GAN_Trainer import GAN_Trainer as Trainer

def setup():
  tf.reset_default_graph()
  tf.set_random_seed(0)
  np.random.seed(0)

def test_sanity():
  trainer = Trainer(['data.mat'],1, None)

def test_single_step():
  trainer = Trainer(['data.mat'],1, None)
  trainer.train(1)