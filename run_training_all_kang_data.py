import Trainer
import tensorflow as tf

# TODO(Chase) Bad hack, run it for now.
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
kang_small = kang_center_files[:2]

if __name__ == '__main__':
  sess = tf.Session()
  path = "./"
  kang_small = [path + x for x in kang_small]
  trainer = Trainer.Trainer(sess, kang_small, batch_size=1, save_dest="./models")
  trainer.train(10)