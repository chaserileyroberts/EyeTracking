import tensorflow as tf
import Trainer

kang_day_1 = './data_kang/day01/Center/data.mat'
kang_day_2 = './data_kang/day02/Center/data.mat'

def setup_function(fnc):
  tf.reset_default_graph()

def test_sanity_trainer_build():
  Trainer.Trainer(tf.Session(), [kang_day_1], batch_size=2)

def test_training_single_step_sanity():
  trainer = Trainer.Trainer(tf.Session(), [kang_day_1], batch_size=2)
  trainer.train(1)

def test_training_step_with_saving(tmpdir):
  trainer = Trainer.Trainer(tf.Session(), [kang_day_1], 
                            batch_size=2,
                            save_dest=str(tmpdir.join("test_model")))
  trainer.train(1)

def test_multiple_files():
  trainer = Trainer.Trainer(tf.Session(), [kang_day_1, kang_day_2], 
                            batch_size=2)
  trainer.train(1)

def test_restoring(tmpdir):
  # TODO(Chase): Test this
  pass
