from load_data import read_data_sets
import tensorflow as tf
import os.path
import argparse
from model2 import *
from PIL import Image
import scipy.misc
import numpy as np
import openslide
import fnmatch

parser = argparse.ArgumentParser(description='Train a U-Net to learn the metastasis regions of lymph nodes.')

parser.add_argument('--output-dir', default='data/output/', help='Data directory (default: data/output/)', dest='output_dir')
parser.add_argument('--data-dir', default='data/input/', help='Data folder (default: data/input/)', dest='data_dir')
parser.add_argument('--width',  default=128, type=int, help='Width of Input Patches',  dest='width')
parser.add_argument('--height', default=128, type=int, help='Height of Input Patches', dest='height')
parser.add_argument('--start-layer', default=2, type=int, help='Lowest layer (default: 2)', dest='start_layer')
parser.add_argument('--model-file', default='data/models/model.ckpt', help='Model file (default: data/models/model.ckpt)', dest='model_file')
parser.add_argument('--filter-count', default=64, type=int, help='Number of convolutions filters in the first level  (default: 64)', dest='filter_count')
parser.add_argument('--layer-count', default=5, type=int, help='Number of convolutions layers  (default: 5)', dest='layer_count')
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

def filter_image(image, width, height):
  s = np.sum(image,2)
  if np.sum(s==0) > 0.05*width*height:
    return False
  avg = np.sum(s)/width/height/3
  return avg < 220

width  = args.width
height = args.height

indir = args.data_dir
outdir = args.output_dir

scale = 1

for root, dirnames, filenames in os.walk(indir):
  for maskfile in fnmatch.filter(filenames, '*.tif'):
    imagefile = maskfile[:-4]
    print imagefile

    infile = indir + imagefile
    outfile = outdir + imagefile

    image = openslide.OpenSlide(infile + '.tif')
#    mask  = openslide.OpenSlide(infile + '_Mask.tif')
    (w, h) = image.level_dimensions[LEVEL]

    scipy.misc.imsave(outfile + '.jpg', image.read_region((0, 0), OUTLEVEL, (w/(2**(OUTLEVEL-LEVEL)), h/(2**(OUTLEVEL-LEVEL)))))
#    scipy.misc.imsave(outfile + '_Mask.jpg', mask.read_region((0, 0), OUTLEVEL, (w/(2**(OUTLEVEL-LEVEL)), h/(2**(OUTLEVEL-LEVEL)))))

    outimage = np.zeros((h/(2**(OUTLEVEL-LEVEL)), w/(2**(OUTLEVEL-LEVEL))))

    print outimage.shape
#    print np.array(mask.read_region((0, 0), OUTLEVEL, (w/(2**(OUTLEVEL-LEVEL)), h/(2**(OUTLEVEL-LEVEL))))).shape

    for i in range(w / width / scale):
      for j in range(h / height / scale):
        x = i*width * scale
        y = j*height * scale

        im1 = np.array(image.read_region((x*(2**LEVEL), y*(2**LEVEL)), LEVEL, (width, height)))[:,:,0:3]

        if filter_image(im1, width, height):
          # im2 = np.array(image.read_region((x*(2**LEVEL) - (width*3/2)*(2**LEVEL),  y*(2**LEVEL) - (height*3/2)*(2**LEVEL)),  LEVEL+2, (width, height)))[:,:,0:3]
          # im3 = np.array(image.read_region((x*(2**LEVEL) - (width*15/2)*(2**LEVEL), y*(2**LEVEL) - (height*15/2)*(2**LEVEL)), LEVEL+4, (width, height)))[:,:,0:3]
          # im = np.dstack((im1, im2, im3))
          im = im1

          o = sess.run(model.y, feed_dict={model.x_image: [im]})
          new_w, new_h = o.shape[1], o.shape[2]
          o = (o.reshape(new_w, new_h) * 255.0).astype(np.uint8)
          out_im = np.array(Image.fromarray(o).resize((new_w/(2**(OUTLEVEL-LEVEL)), new_h/(2**(OUTLEVEL-LEVEL))), Image.ANTIALIAS))
          offset_x = (width-new_w)/2/(2**(OUTLEVEL-LEVEL))
          offset_y = (height-new_h)/2/(2**(OUTLEVEL-LEVEL))
          outimage[y/(2**(OUTLEVEL-LEVEL))+offset_y:(y+new_h)/(2**(OUTLEVEL-LEVEL))+offset_y, x/(2**(OUTLEVEL-LEVEL))+offset_x:(x+new_w)/(2**(OUTLEVEL-LEVEL))+offset_x] += out_im

      print i*1.0/(w / width / scale)

    scipy.misc.imsave(outfile + '_Output.tif', outimage)
    scipy.misc.imsave(outfile + '_Output_Thr.jpg', np.where(outimage > 128, 255, 0))


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
