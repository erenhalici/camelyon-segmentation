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

parser.add_argument('--output-dir', default='data/output/', help='Data directory (default: data/output/)', dest='output_dir')
parser.add_argument('--data-dir', default='data/input/', help='Data folder (default: data/input/)', dest='data_dir')
parser.add_argument('--width',  default=252, type=int, help='Width of Input Patches',  dest='width')
parser.add_argument('--height', default=252, type=int, help='Height of Input Patches', dest='height')
parser.add_argument('--stride', default=0, type=int, help='Stride (default: width)', dest='stride')
parser.add_argument('--start-layer', default=2, type=int, help='Lowest layer (default: 2)', dest='start_layer')
parser.add_argument('--model-file', default='data/models/model.ckpt', help='Model file (default: data/models/model.ckpt)', dest='model_file')
parser.add_argument('--filter-count', default=32, type=int, help='Number of convolutions filters in the first level  (default: 32)', dest='filter_count')
parser.add_argument('--layer-count', default=4, type=int, help='Number of convolutions layers  (default: 4)', dest='layer_count')
parser.add_argument('--out-level', default=4, type=int, help='Output level (default: 4)', dest='out_level')

args = parser.parse_args()

num_input_layers = 3
num_output_layers = 1

LEVEL = args.start_layer
OUTLEVEL = args.out_level

model = Model(args.width, args.height, num_input_layers, num_output_layers, args.filter_count, args.layer_count, 1e-5)

saver = tf.train.Saver()
sess = tf.Session()
# sess.run(tf.global_variables_initializer())
sess.run(tf.initialize_all_variables())

if args.model_file:
  if os.path.isfile(args.model_file):
    saver.restore(sess, args.model_file)
    print("Model restored.")

width  = args.width
height = args.height

stride = args.stride

if stride == 0:
  stride = width

indir = args.data_dir
outdir = args.output_dir

for root, dirnames, filenames in os.walk(indir):
  max_i = 0
  max_j = 0

  files = fnmatch.filter(filenames, '*_Mask.jpg')
  for maskfile in files:
    i, j = [int(e) for e in maskfile[:-9].split('_')]

    if max_i < i: max_i = i
    if max_j < j: max_j = j

  print root.split('/')[-1], (max_i+1)*stride, (max_j+1)*stride

  output = np.zeros(((max_j+1)*stride/4, (max_i+1)*stride/4), dtype=np.uint8)

  count = 0
  for maskfile in files:
    imagefile = maskfile[:-9]

    i, j = [int(e) for e in imagefile.split('_')]

    # print str(count) + '/' + str(len(files))
    count += 1

    image = np.array(Image.open(root + '/' + imagefile + '.jpg'))
    o = (sess.run(model.y, feed_dict={model.x_image: [image]})[0,:,:,0] * 255).astype(np.uint8)
    # o = image
    arr = np.array(Image.fromarray(o).resize((stride/4, stride/4), Image.ANTIALIAS))
    output[j*(stride/4):(j+1)*(stride/4), i*(stride/4):(i+1)*(stride/4)] = arr
  if max_i > 0:
    # print output.shape
    scipy.misc.imsave(outdir + '/' + root.split('/')[-1] + '_Output.tif', output)
    scipy.misc.imsave(outdir + '/' + root.split('/')[-1] + '_Output_Thr.jpg', np.where(output > 128, 255, 0))


def im(image, i, j):
  x = i*width * scale
  y = j*height * scale
  im1 = np.array(image.read_region((x*(2**LEVEL), y*(2**LEVEL)), LEVEL, (width, height)))[:,:,0:3]
  im2 = np.array(image.read_region((x*(2**LEVEL) - (width*3/2)*(2**LEVEL),  y*(2**LEVEL) - (height*3/2)*(2**LEVEL)),  LEVEL+2, (width, height)))[:,:,0:3]
  im3 = np.array(image.read_region((x*(2**LEVEL) - (width*15/2)*(2**LEVEL), y*(2**LEVEL) - (height*15/2)*(2**LEVEL)), LEVEL+4, (width, height)))[:,:,0:3]
  return np.dstack((im1, im2, im3))

def show(image, i=0):
  Image.fromarray(image[0,:,:,i]/np.max(image[0,:,:,i])*255).show()

def showlayer(image, layer):
  Image.fromarray(image[:,:,layer*3:layer*3+3]).show()
