from load_data import read_data_sets
import tensorflow as tf
import os.path
import argparse
from model import *
from PIL import Image
import scipy.misc

parser = argparse.ArgumentParser(description='Train a DCNN to learn Metastasis regions of human cells.')

parser.add_argument('--output-dir', default='data/output/', help='Data directory (default: data/output/)', dest='output_dir')
parser.add_argument('--data-dir', default='data/', help='Data file (default: data/)', dest='data_dir')
parser.add_argument('--width',  default=512, type=int, help='Width of Input Patches',  dest='width')
parser.add_argument('--height', default=512, type=int, help='Height of Input Patches', dest='height')
parser.add_argument('--start-file', help='Starting data file', dest='start_file')
parser.add_argument('--start-step', default=0, type=int, help='Starting step (Default: 0)', dest='start_step')
parser.add_argument('--num-steps', default=300000, type=int, help='Number of steps of execution (default: 300000)', dest='num_steps')
parser.add_argument('--learning-rate', default=1e-4, type=float, help='Learning Rate (default: 1e-4)', dest='learning_rate')
parser.add_argument('--batch-size', default=16, type=int, help='Batch size (default: 16)', dest='batch_size')
parser.add_argument('--filter-count', default=64, type=int, help='Number of convolutions filters in the first level  (default: 64)', dest='filter_count')
parser.add_argument('--layer-count', default=5, type=int, help='Number of convolutions layers  (default: 5)', dest='layer_count')
parser.add_argument('--dropout', type=float, help='Dropout (if none is given, no dropout)', dest='dropout')
parser.add_argument('--test-interval', default=10000, type=int, help='Test Accuracy Interval (default: 10000)', dest='test_interval')

args = parser.parse_args()

num_input_layers = 3
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

train_accuracy_sum = 0.0

batch = data_set.train.next_batch(args.batch_size)

[segmented, train_error] = sess.run([model.y, model.error], feed_dict={model.x_image:batch[0], model.y_: batch[1], model.keep_prob: 1.0})

print train_error

for i in range(args.batch_size):
  inimage = batch[0][i]
  outimage = segmented[i].reshape(args.width, args.height) * 255
  truthimage = batch[1][i].reshape(args.width, args.height) * 255

  scipy.misc.imsave(args.output_dir + str(i) + '_in.jpg', inimage)
  scipy.misc.imsave(args.output_dir + str(i) + '_out.jpg', outimage)
  scipy.misc.imsave(args.output_dir + str(i) + '_truth.jpg', truthimage)


# print "test accuracy %g"%sess.run(model.accuracy, feed_dict={
#   model.x_image: data_set.test.all_inimages(), model.y_: data_set.test.all_outimages(), model.keep_prob: 1.0})
