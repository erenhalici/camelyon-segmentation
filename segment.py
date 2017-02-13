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
parser.add_argument('--data-dir', default='data/input/', help='Data folder (default: data/input/)', dest='data_dir')
parser.add_argument('--width',  default=128, type=int, help='Width of Input Patches',  dest='width')
parser.add_argument('--height', default=128, type=int, help='Height of Input Patches', dest='height')
parser.add_argument('--model-file', help='Model file', dest='model_file')
parser.add_argument('--filter-count', default=64, type=int, help='Number of convolutions filters in the first level  (default: 64)', dest='filter_count')
parser.add_argument('--layer-count', default=5, type=int, help='Number of convolutions layers  (default: 5)', dest='layer_count')
parser.add_argument('--out-level', default=4, type=int, help='Output level (default: 4)', dest='out_level')

args = parser.parse_args()

num_input_layers = 9
num_output_layers = 1

LEVEL = 1
OUTLEVEL = args.out_level

model = Model(args.width, args.height, num_input_layers, num_output_layers, args.filter_count, args.layer_count, 0)

saver = tf.train.Saver()
sess = tf.Session()
# sess.run(tf.global_variables_initializer())
sess.run(tf.initialize_all_variables())

if args.model_file:
  if os.path.isfile(args.model_file):
    saver.restore(sess, args.model_file)
    print("Model restored.")

def filter_inimage(width, height, image, i, j):
  im = np.array(image.read_region((i, j), 2, (width, height)))[:,:,0:3]
  avg = np.sum(im)/width/height/3
  return avg < 220

width  = args.width
height = args.height

indir = args.data_dir
outdir = args.output_dir

for root, dirnames, filenames in os.walk(indir):
  for maskfile in fnmatch.filter(filenames, '*_Mask.tif'):
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

        im1 = np.array(image.read_region((x*(2**LEVEL), y*(2**LEVEL)), LEVEL, (width, height)))[:,:,0:3]

        if (np.sum(im1)/width/height/3) < 220:
          im2 = np.array(image.read_region((x*(2**LEVEL) - (width*3/2)*(2**LEVEL),  y*(2**LEVEL) - (height*3/2)*(2**LEVEL)),  LEVEL+2, (width, height)))[:,:,0:3]
          im3 = np.array(image.read_region((x*(2**LEVEL) - (width*15/2)*(2**LEVEL), y*(2**LEVEL) - (height*15/2)*(2**LEVEL)), LEVEL+4, (width, height)))[:,:,0:3]
          im = np.dstack((im1, im2, im3))

          o = (sess.run(model.y, feed_dict={model.x_image: [im]}).reshape(width, height) * 255.0).astype(np.uint8)
          out_im = np.array(Image.fromarray(o).resize((width/(2**(OUTLEVEL-LEVEL)), height/(2**(OUTLEVEL-LEVEL))), Image.ANTIALIAS))
          outimage[y/(2**(OUTLEVEL-LEVEL)):(y+height)/(2**(OUTLEVEL-LEVEL)), x/(2**(OUTLEVEL-LEVEL)):(x+width)/(2**(OUTLEVEL-LEVEL))] += out_im

    scipy.misc.imsave(outfile + '_Output.tif', outimage)
    scipy.misc.imsave(outfile + '_Output_Thr.jpg', np.where(outimage > 128, 255, 0))

