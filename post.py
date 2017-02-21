import numpy as np
from PIL import Image
import os.path
import fnmatch
import scipy.misc

THRESHOLD = 192

indir = 'data/output/multilayer/'

for root, dirnames, filenames in os.walk(indir):
  for maskfile in fnmatch.filter(filenames, '*_Mask.jpg'):
    imagefile = maskfile[:-9]

    mask = np.where(np.array(Image.open(indir + maskfile))[:,:,0] > 128, 255, 0)
    im = np.where(np.array(Image.open(indir + imagefile + '_Output.tif')) > THRESHOLD, 255, 0)

    green = mask&im
    red = mask&(255-im)
    blue = (255-mask)&im

    imout = np.dstack((red | blue, green | blue, blue)).astype(np.uint8)
    scipy.misc.imsave(indir + imagefile + '_Diff.jpg', imout)

    tp = np.sum(green)/255
    fn = np.sum(red)/255
    fp = np.sum(blue)/255

    recall = tp / float(tp+fn)
    precision = tp / float(tp+fp)

    print imagefile
    print "Positive: " + str(tp+fn)
    print "Recall: " + str(recall)
    print "Precision: " + str(precision)
    print "F1: " + str(2*precision*recall/(precision+recall))
