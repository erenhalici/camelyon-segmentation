from load_data import read_data_sets
import tensorflow as tf
import os.path
import argparse
from model import *

parser = argparse.ArgumentParser(description='Train a DCNN to learn Metastasis regions of human cells.')

parser.add_argument('--output-dir', default='data/models/5_layers/', help='Data directory (default: data/models/5_layers/)', dest='output_dir')
parser.add_argument('--data-dir', default='data/', help='Data file (default: data/training/L9.hdf5)', dest='data_dir')
parser.add_argument('--width',  default=512, help='Width of Input Patches',  dest='width')
parser.add_argument('--heigth', default=512, help='Height of Input Patches', dest='height')
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
width = 512
height = 512

data_set = read_data_sets(args.width, args.height, args.data_dir, args.start_step*args.batch_size)

print("Training Data Size: {}".format(data_set.train.num_samples))
# print("Training InImage Shape: {}".format(data_set.train.inimages.shape))
# print("Training OutImage Shape: {}".format(data_set.train.outimages.shape))
# print("Test InImage Shape: {}".format(data_set.test.inimages.shape))
# print("Test OutImage Shape: {}".format(data_set.test.outimages.shape))

# num_input_layers  = data_set.train.inimages.shape[3]
# num_output_layers = data_set.train.outimages.shape[3]
# width = data_set.train.inimages.shape[1]
# height = data_set.train.inimages.shape[2]



model = Model(width, height, num_input_layers, num_output_layers, args.filter_count, args.layer_count, args.learning_rate)

saver = tf.train.Saver()
sess = tf.Session()
# sess.run(tf.global_variables_initializer())
sess.run(tf.initialize_all_variables())

if args.start_file:
  if os.path.isfile(args.start_file):
    saver.restore(sess, args.start_file)
    print("Model restored.")

train_accuracy_sum = 0.0

for i in range(args.num_steps):
  batch = data_set.train.next_batch(args.batch_size)
  if i%args.test_interval == 0:
    print "epoch: %g"%data_set.train.epoch()
    # print "test accuracy %g"%sess.run(model.accuracy, feed_dict={
    #   model.x_image: data_set.test.all_inimages(), model.y_: data_set.test.all_outimages(), model.keep_prob: 1.0})
    save_path = saver.save(sess, args.output_dir + "model_" + str(i+args.start_step) + ".ckpt")
    print("Model saved in file: ", save_path)

  if i%10 == 0:
    train_accuracy_sum += sess.run(model.accuracy,feed_dict={
      model.x_image:batch[0], model.y_: batch[1], model.keep_prob: 1.0})

  if i%500 == 0:
    print "step %d, training accuracy %g"%(i+args.start_step, train_accuracy_sum/50)
    train_accuracy_sum = 0

  sess.run(model.train_step, feed_dict={model.x_image: batch[0], model.y_: batch[1], model.keep_prob: args.dropout})

# print "test accuracy %g"%sess.run(model.accuracy, feed_dict={
#   model.x_image: data_set.test.all_inimages(), model.y_: data_set.test.all_outimages(), model.keep_prob: 1.0})
save_path = saver.save(sess, args.output_dir + "model.ckpt")
print("Model saved in file: ", save_path)
