from load_data import read_data_sets
import tensorflow as tf
import os.path
import argparse
from model import *
# from PIL import Image

parser = argparse.ArgumentParser(description='Train a U-Net to learn the metastasis regions of lymph nodes.')

parser.add_argument('--output-dir', default='data/models/', help='Data directory (default: data/models/)', dest='output_dir')
parser.add_argument('--data-dir', default='data/', help='Data folder (default: data/)', dest='data_dir')
parser.add_argument('--width',  default=128, type=int, help='Width of Input Patches (default: 128)',  dest='width')
parser.add_argument('--height', default=128, type=int, help='Height of Input Patches (default: 128)', dest='height')
parser.add_argument('--start-file', help='Starting model file (optional)', dest='start_file')
parser.add_argument('--start-step', default=0, type=int, help='Starting step (Default: 0)', dest='start_step')
parser.add_argument('--num-steps', default=1000000, type=int, help='Number of steps of execution (default: 1000000)', dest='num_steps')
parser.add_argument('--learning-rate', default=1e-4, type=float, help='Learning Rate (default: 1e-4)', dest='learning_rate')
parser.add_argument('--batch-size', default=16, type=int, help='Batch size (default: 16)', dest='batch_size')
parser.add_argument('--filter-count', default=64, type=int, help='Number of convolutions filters in the first level  (default: 64)', dest='filter_count')
parser.add_argument('--layer-count', default=5, type=int, help='Number of convolutions layers  (default: 5)', dest='layer_count')
parser.add_argument('--test-interval', default=10000, type=int, help='Test Accuracy Interval (default: 10000)', dest='test_interval')

args = parser.parse_args()

num_input_layers = 9
num_output_layers = 1

data_set = read_data_sets(args.width, args.height, args.data_dir, args.start_step*args.batch_size)

print("Training Data Size: {}".format(data_set.train.num_samples))

model = Model(args.width, args.height, num_input_layers, num_output_layers, args.filter_count, args.layer_count, args.learning_rate)

saver = tf.train.Saver()
sess = tf.Session()
# sess.run(tf.global_variables_initializer())
sess.run(tf.initialize_all_variables())

if args.start_file:
  if os.path.isfile(args.start_file):
    saver.restore(sess, args.start_file)
    print("Model restored.")

train_error_sum = 0.0
train_acc_sum = 0.0

for i in range(args.num_steps):
  batch = data_set.train.next_batch(args.batch_size)
  if i % args.test_interval == 0:
    print "epoch: %g"%data_set.train.epoch()
    save_path = saver.save(sess, args.output_dir + "model_" + str(i+args.start_step) + ".ckpt")
    print("Model saved in file: ", save_path)

  if i%10 == 0:
    [err, acc] = sess.run([model.error, model.accuracy],feed_dict={model.x_image:batch[0], model.y_: batch[1]})
    train_error_sum += err
    train_acc_sum += acc

  if i%500 == 0:
    print "step %d, training error %g, training accuracy %g"%(i+args.start_step, train_error_sum/50, train_acc_sum/50)
    train_error_sum = 0
    train_acc_sum = 0

  sess.run(model.train_step, feed_dict={model.x_image: batch[0], model.y_: batch[1]})

save_path = saver.save(sess, args.output_dir + "model.ckpt")
print("Model saved in file: ", save_path)
