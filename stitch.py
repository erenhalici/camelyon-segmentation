from load_data import read_data_sets
import tensorflow as tf
import os.path
import argparse
from PIL import Image
import scipy.misc
import numpy as np
import fnmatch
from math import ceil

parser = argparse.ArgumentParser(description='Stitch patches to get the images and masks')

parser.add_argument('--output-dir', default='data/output/', help='Data directory (default: data/output/)', dest='output_dir')
parser.add_argument('--data-dir', default='data/input/', help='Data folder (default: data/input/)', dest='data_dir')
parser.add_argument('--width',  default=252, type=int, help='Width of Input Patches',  dest='width')
parser.add_argument('--height', default=252, type=int, help='Height of Input Patches', dest='height')
parser.add_argument('--stride', default=0, type=int, help='Stride (default: width)', dest='stride')
parser.add_argument('--out-level', default=4, type=int, help='Output level (default: 4)', dest='out_level')

args = parser.parse_args()

num_input_layers = 3
num_output_layers = 1

OUTLEVEL = args.out_level

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

  outimage = np.zeros(((max_j+1)*stride/4, (max_i+1)*stride/4, 3), dtype=np.uint8)
  outmask = np.zeros(((max_j+1)*stride/4, (max_i+1)*stride/4), dtype=np.uint8)

  count = 0
  for maskfile in files:
    imagefile = maskfile[:-9]

    i, j = [int(e) for e in imagefile.split('_')]

    # print str(count) + '/' + str(len(files))
    count += 1

    image = np.array(Image.open(root + '/' + imagefile + '.jpg').resize((width/4, height/4), Image.ANTIALIAS))
    mask = np.array(Image.open(root + '/' + imagefile + '_Mask.jpg').resize((width/4, height/4), Image.ANTIALIAS))
    outimage[j*(stride/4):j*(stride/4)+width/4, i*(stride/4):i*(stride/4)+height/4, :] = image
    outmask[j*(stride/4):j*(stride/4)+width/4, i*(stride/4):i*(stride/4)+height/4] = mask
  if max_i > 0:
    # print output.shape
    scipy.misc.imsave(outdir + '/' + root.split('/')[-1] + '.jpg', outimage)
    scipy.misc.imsave(outdir + '/' + root.split('/')[-1] + '_Truth.jpg', outmask)
