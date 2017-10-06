import Trainer
import tensorflow as tf
import argparse

parser = argparse.ArgumentParser(
    description='Run Eye Tracking Convnet Training.')
parser.add_argument('-evaluate', '-e', action='store_true', default=False)
args = parser.parse_args()

test_data = [
    'kang/day01/Center/data.mat',
  'kang/day01/Left/data.mat',
  'kang/day01/Right/data.mat'
]
kang_center_files = [
    'kang/day02/Center/data.mat',
  'kang/day03/Center/data.mat',
  'kang/day04/Center/data.mat',
  'kang/day05/Center/data.mat',
  'kang/day06/Center/data.mat'
]
kang_left_files = [

    'kang/day02/Left/data.mat',
  'kang/day03/Left/data.mat',
  'kang/day04/Left/data.mat',
  'kang/day05/Left/data.mat',
]
kang_right_files = [

    'kang/day02/Right/data.mat',
  'kang/day03/Right/data.mat',
  'kang/day04/Right/data.mat',
  'kang/day05/Right/data.mat',
  'kang/day06/Right/data.mat'
]

all_kang_data = kang_center_files + kang_left_files + kang_right_files

if __name__ == '__main__':
  path = "/media/roberc4/kang/final_dataset/"
  data_paths = [path + x for x in all_kang_data]
  batch_size = 32
  trainer = Trainer.Trainer(
      data_paths, 
      batch_size=batch_size, 
      save_dest='/media/roberc4/kang/models/',
      eval_loop=args.evaluate)
  if args.evaluate:
    trainer.evaluate(num_evals=100)
  else:
    trainer.train()
