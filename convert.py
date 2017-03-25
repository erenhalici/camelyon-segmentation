import os.path
import argparse
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
parser.add_argument('--stride', default=0, type=int, help='Stride (default: width)', dest='stride')

args = parser.parse_args()

num_input_layers = 3
num_output_layers = 1

LEVEL = args.start_layer

def filter_image(image, width, height):
  s = np.sum(image,2)
  if np.sum(s==0) > 0.05*width*height:
    return False
  avg = np.sum(s)/width/height/3
  return avg < 220

width  = args.width
height = args.height

stride = args.stride

if stride == 0:
  stride = width

indir = args.data_dir
imagedir = indir+'images/'
maskdir = indir+'masks/'
outdir = args.output_dir


for root, dirnames, filenames in os.walk(imagedir):
  for filename in fnmatch.filter(filenames, '*.tif'):
    imagefile = imagedir + filename[:-4] + '.tif'
    # maskfile = maskdir + filename[:-4] + '_Mask.tif'
    print imagefile

    infile = indir + imagefile

    image = openslide.OpenSlide(imagefile)
    # mask  = openslide.OpenSlide(maskfile)
    (w, h) = image.level_dimensions[LEVEL]

    if not os.path.exists(outdir + filename[:-4]):
      os.makedirs(outdir + filename[:-4])

    for i in range(w / stride):
      for j in range(h / stride):
        x = i*stride
        y = j*stride

        im = np.array(image.read_region((x*(2**LEVEL), y*(2**LEVEL)), LEVEL, (width, height)))[:,:,0:3]

        if filter_image(im, width, height):
          im = np.array(image.read_region((x*(2**LEVEL), y*(2**LEVEL)), 0, (width*(2**LEVEL), height*(2**LEVEL))).resize((width,height),Image.ANTIALIAS))[:,:,0:3]
          # m = np.zeros((width, height))
          # m = np.array(mask.read_region((x*(2**LEVEL), y*(2**LEVEL)), LEVEL, (width, height)))[:,:,0]

          #if np.sum(m) > 0.0:
          #  m = np.array(mask.read_region((x*(2**LEVEL), y*(2**LEVEL)), 0, (width*(2**LEVEL), height*(2**LEVEL))).resize((width,height),Image.ANTIALIAS))[:,:,0]
          
          outfile = outdir + filename[:-4] + '/' + str(i) + '_' + str(j)
          scipy.misc.imsave(outfile + '.jpg', im)
          # scipy.misc.imsave(outfile + '_Mask.jpg', m)

      print i*1.0/(w / width)
