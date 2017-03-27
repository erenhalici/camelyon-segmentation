from load_data import read_data_sets
import tensorflow as tf
import os.path
import argparse
from model2 import *
from PIL import Image
import scipy.misc
import numpy as np
import fnmatch

parser = argparse.ArgumentParser(description='Use a U-Net to find the metastasis regions of lymph nodes.')

parser.add_argument('--data-dir', default='data/input/', help='Data folder (default: data/input/)', dest='data_dir')
parser.add_argument('--width',  default=252, type=int, help='Width of Input Patches',  dest='width')
parser.add_argument('--height', default=252, type=int, help='Height of Input Patches', dest='height')
parser.add_argument('--model-file', default='data/models/model.ckpt', help='Model file (default: data/models/model.ckpt)', dest='model_file')
parser.add_argument('--filter-count', default=32, type=int, help='Number of convolutions filters in the first level  (default: 32)', dest='filter_count')
parser.add_argument('--layer-count', default=4, type=int, help='Number of convolutions layers  (default: 4)', dest='layer_count')

args = parser.parse_args()

num_input_layers = 3
num_output_layers = 1

model = Model(args.width, args.height, num_input_layers, num_output_layers, args.filter_count, args.layer_count, 1e-5)

saver = tf.train.Saver()
sess = tf.Session()
# sess.run(tf.global_variables_initializer())
sess.run(tf.initialize_all_variables())

if args.model_file:
  if os.path.isfile(args.model_file):
    saver.restore(sess, args.model_file)
    print("Model restored.")

indir = args.data_dir

total_count = 0
gtp = 0
gtn = 0
gfp = 0
gfn = 0

for root, dirnames, filenames in os.walk(indir):
  files = fnmatch.filter(filenames, '*_Mask.jpg')

  count = 0
  ftp = 0
  ftn = 0
  ffp = 0
  ffn = 0

  for maskfile in files:
    imagefile = maskfile[:-9]

    # print str(count) + '/' + str(len(files))
    count += 1

    image = np.array(Image.open(root + '/' + imagefile + '.jpg'))
    mask = np.array(Image.open(root + '/' + maskfile)).reshape((args.width, args.height, 1))
    ptp, ptn, pfp, pfn = sess.run([model.tp, model.tn, model.fp, model.fn] , feed_dict={model.x_image: [image], model.y_: [mask]})
    # print ptp, ptn, pfp, pfn
    ftp += ptp
    ftn += ptn
    ffp += pfp
    ffn += pfn

  if count == 0: continue
  print "For File: " + root.split('/')[-1]
  print "TP: " + str(ftp/count) + " TN: " + str(ftn/count) + " FP: " + str(ffp/count) + " FN: " + str(ffn/count)

  total_count += count
  gtp += ftp
  gtn += ftn
  gfp += ffp
  gfn += ffn

print "GLOBAL VALUES FOR: " + args.model_file
print "TP: " + str(gtp/total_count) + " TN: " + str(gtn/total_count) + " FP: " + str(gfp/total_count) + " FN: " + str(gfn/total_count)
print "Accuracy: " + str((gtp+gtn)/(gtp+gtn+gfp+gfn))
print "Sensitivity: " + str(gtp/(gtp+gfn))
print "Specificity: " + str(gtn/(gtn+gfp))
