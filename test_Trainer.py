import tensorflow as tf
import Trainer

kang_day_1 = './kang/day01/Center/data.mat'
kang_day_2 = './kang/day02/Center/data.mat'


def setup_function(fnc):
    tf.reset_default_graph()


def test_sanity_trainer_build():
    Trainer.Trainer([kang_day_1], batch_size=1)


def test_training_single_step_sanity():
    trainer = Trainer.Trainer([kang_day_1], batch_size=1)
    trainer.train(1)


def test_training_step_with_saving(tmpdir):
    trainer = Trainer.Trainer([kang_day_1],
                              batch_size=1,
                              save_dest=str(tmpdir.join("test_model")))
    trainer.train(1)


def test_multiple_files():
    trainer = Trainer.Trainer([kang_day_1, kang_day_2],
                              batch_size=1)
    trainer.train(1)


def test_mulitple_training_steps():
    trainer = Trainer.Trainer([kang_day_1], batch_size=1)
    trainer.train(5)


def test_eval_loop(tmpdir):
    trainer = Trainer.Trainer([kang_day_1], batch_size=1,
                              eval_loop=True, save_dest=str(tmpdir.join("test_model")))
    trainer.evaluate(num_evals=1,
                     eval_secs=1,
                     timeout=1)

