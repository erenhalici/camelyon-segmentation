from load_data import read_data_sets
import tensorflow as tf
import os.path
import argparse
from model import *
from PIL import Image
import scipy.misc
import numpy as np
import openslide

parser = argparse.ArgumentParser(description='Train a DCNN to learn Metastasis regions of human cells.')

parser.add_argument('--output-dir', default='data/output/', help='Data directory (default: data/output/)', dest='output_dir')
parser.add_argument('--data-dir', default='data/', help='Data file (default: data/)', dest='data_dir')
parser.add_argument('--width',  default=128, type=int, help='Width of Input Patches',  dest='width')
parser.add_argument('--height', default=128, type=int, help='Height of Input Patches', dest='height')
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

num_input_layers = 9
num_output_layers = 1

LEVEL = 1
OUTLEVEL = 4

# data_set = read_data_sets(args.width, args.height, args.data_dir, args.start_step*args.batch_size)

# print("Training Data Size: {}".format(data_set.train.num_samples))

with tf.device('/gpu:1'):
  model = Model(args.width, args.height, num_input_layers, num_output_layers, args.filter_count, args.layer_count, args.learning_rate)

saver = tf.train.Saver()
sess = tf.Session()
# sess.run(tf.global_variables_initializer())
sess.run(tf.initialize_all_variables())

if args.start_file:
  if os.path.isfile(args.start_file):
    saver.restore(sess, args.start_file)
    print("Model restored.")

# train_accuracy_sum = 0.0

# batch = data_set.train.next_batch(args.batch_size)

# [segmented, train_error, train_accuracy] = sess.run([model.y, model.error, model.accuracy], feed_dict={model.x_image:batch[0], model.y_: batch[1], model.keep_prob: 1.0})

# print train_error, train_accuracy
# # Image.fromarray(segmented[0].reshape(args.width, args.height)*255).show()

# for i in range(args.batch_size):
#   inimage = (batch[0][i]).astype(int)
#   outimage = (segmented[i].reshape(args.width, args.height) * 255).astype(int)
#   outthr = (np.where(segmented[i].reshape(args.width, args.height) > 0.5, 255, 0)).astype(int)
#   truthimage = (batch[1][i].reshape(args.width, args.height) * 255).astype(int)

#   scipy.misc.imsave(args.output_dir + str(i) + '_in.jpg', inimage)
#   scipy.misc.imsave(args.output_dir + str(i) + '_out.jpg', outimage)
#   scipy.misc.imsave(args.output_dir + str(i) + '_out_thr.jpg', outthr)
#   scipy.misc.imsave(args.output_dir + str(i) + '_truth.jpg', truthimage)

# # print "test accuracy %g"%sess.run(model.accuracy, feed_dict={
# #   model.x_image: data_set.test.all_inimages(), model.y_: data_set.test.all_outimages(), model.keep_prob: 1.0})

def filter_inimage(width, height, image, i, j):
  im = np.array(image.read_region((i, j), 2, (width, height)))[:,:,0:3]
  avg = np.sum(im)/width/height/3
  return avg < 220

# def filter_outimage(width, height, image, i, j):
#   im = np.array(image.read_region((i, j), 2, (width, height)))[:,:,0]
#   avg = np.sum(im)/width/height
#   return avg > 0.0

width  = args.width
height = args.height

# infile = 'data/training/Tumor_091'
# outfile = 'data/output/Tumor_091'
# infile = 'data/test/Test_065'
# outfile = 'data/output/Test_065'
# infile = 'data/test/Test_122'
# outfile = 'data/output/Test_122'
# infile = 'data/test/Test_046'
# outfile = 'data/output/Test_046'
# infile = 'data/test/Test_001'
# outfile = 'data/output/Test_001'

indir = 'data/test'
outdir = 'data/output/multilayer'

for root, dirnames, filenames in os.walk(indir):
  for maskfile in fnmatch.filter(filenames, 'Test_*_Mask.tif'):
    imagefile = maskfile[:-9] + '.tif'
    print imagefile

    infile = indir + imagefile
    outfile = outdir + imagefile

    image = openslide.OpenSlide(infile + '.tif')
    mask  = openslide.OpenSlide(infile + '_Mask.tif')
    (w, h) = image.level_dimensions[LEVEL]

    scipy.misc.imsave(outfile + '.jpg', image.read_region((0, 0), OUTLEVEL, (w/(2**(OUTLEVEL-LEVEL)), h/(2**(OUTLEVEL-LEVEL)))))
    scipy.misc.imsave(outfile + '_Mask.jpg', mask.read_region((0, 0), OUTLEVEL, (w/(2**(OUTLEVEL-LEVEL)), h/(2**(OUTLEVEL-LEVEL)))))

    outimage = np.zeros((h/(2**(OUTLEVEL-LEVEL)), w/(2**(OUTLEVEL-LEVEL))))

    print outimage.shape
    print np.array(mask.read_region((0, 0), OUTLEVEL, (w/(2**(OUTLEVEL-LEVEL)), h/(2**(OUTLEVEL-LEVEL))))).shape

    scale = 1
    for i in range(w / width / scale):
      for j in range(h / height / scale):
        x = i*width * scale
        y = j*height * scale

        # in_im = np.array(image.read_region((x*4, y*4), 2, (width, height)))[:,:,0:3]
        im1 = np.array(image.read_region((x*(2**LEVEL), y*(2**LEVEL)), LEVEL, (width, height)))[:,:,0:3]

        if (np.sum(im1)/width/height/3) < 220:
          im2 = np.array(image.read_region((x*(2**LEVEL) - (width*3/2)*(2**LEVEL),  y*(2**LEVEL) - (height*3/2)*(2**LEVEL)),  LEVEL+2, (width, height)))[:,:,0:3]
          im3 = np.array(image.read_region((x*(2**LEVEL) - (width*15/2)*(2**LEVEL), y*(2**LEVEL) - (height*15/2)*(2**LEVEL)), LEVEL+4, (width, height)))[:,:,0:3]
          im = np.dstack((im1, im2, im3))

          # Image.fromarray(im[:,:,0:3]).show()
          # Image.fromarray(im[:,:,3:6]).show()
          # Image.fromarray(im[:,:,6:9]).show()

          o = (sess.run(model.y, feed_dict={model.x_image: [im]}).reshape(width, height) * 255.0).astype(np.uint8)
          out_im = np.array(Image.fromarray(o).resize((width/(2**(OUTLEVEL-LEVEL)), height/(2**(OUTLEVEL-LEVEL))), Image.ANTIALIAS))
          outimage[y/(2**(OUTLEVEL-LEVEL)):(y+height)/(2**(OUTLEVEL-LEVEL)), x/(2**(OUTLEVEL-LEVEL)):(x+width)/(2**(OUTLEVEL-LEVEL))] += out_im


      print i*1.0/(w/width)*scale
    scipy.misc.imsave(outfile + '_Output.tif', outimage)
    scipy.misc.imsave(outfile + '_Output_Thr.jpg', np.where(outimage > 128, 255, 0))

