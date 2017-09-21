import tensorflow as tf
import Trainer


def setup_function(fnc):
  tf.reset_default_graph()

def test_sanity_trainer_build():
  Trainer.Trainer(tf.Session(), './data_kang/day01/Center/data.mat')

def test_training_single_step_sanity():
  trainer = Trainer.Trainer(tf.Session(), './data_kang/day01/Center/data.mat')
  trainer.train(1)
