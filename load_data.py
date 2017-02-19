
import numpy as np
import openslide
import fnmatch
import os

LEVEL = 1

class DataSet(object):
  def __init__(self, width, height, inimages, outimages, num_samples):
    self._width  = width
    self._height = height

    self._num_samples = num_samples

    self._inimages = inimages
    self._outimages = outimages

    self._epochs_completed = 0
    self._index_in_epoch = 0

    self._indices = np.arange(self._num_samples)
    np.random.shuffle(self._indices)

  @property
  def inimages(self):
    return self._inimages
  @property
  def outimages(self):
    return self._outimages
  @property
  def num_samples(self):
    return self._num_samples
  @property
  def epochs_completed(self):
    return self._epochs_completed
  def all_inimages(self):
    return [self.get_inimage_at_index(index)  for index in range(self._num_samples)]
  def all_outimages(self):
    return [self.get_outimage_at_index(index) for index in range(self._num_samples)]
  def epoch(self):
    return self._epochs_completed + self._index_in_epoch*1.0/self._num_samples
  def augment_image(self, image, i):
    if i == 0:
      return np.rot90(image)
    elif i == 1:
      return np.rot90(image,2)
    elif i == 2:
      return np.rot90(image,3)
    elif i == 3:
      return image
    elif i == 4:
      return np.fliplr(image)
    elif i == 5:
      return np.flipud(image)
    elif i == 6:
      return image.transpose(1,0,2)
    elif i == 7:
      return np.fliplr(np.rot90(image))
  def load_inimage(self, image_data):
    (image, i, j, k) = image_data
    w = self._width
    h = self._height

    im1 = np.array(image.read_region((i, j), LEVEL, (w, h)))[:,:,0:3]
    im2 = np.array(image.read_region((i - (w*3/2)*(2**LEVEL), j - (h*3/2)*(2**LEVEL)), LEVEL+2, (w, h)))[:,:,0:3]
    im3 = np.array(image.read_region((i - (w*15/2)*(2**LEVEL), j - (h*15/2)*(2**LEVEL)), LEVEL+4, (w, h)))[:,:,0:3]

    im = np.dstack((im1, im2, im3))

    return self.augment_image(im, k)
  def load_outimage(self, image_data):
    if image_data == None:
      return np.zeros([self._width,self._height,1])
    else:
      (image, i, j, k) = image_data
      im = self.augment_image(np.array(image.read_region((i, j), LEVEL, (self._width, self._height)))[:,:,0].reshape([self._width,self._height,1]), k)
      return im/255.0
  def get_inimage_at_index(self, index):
    return self.load_inimage(self._inimages[index])
  def get_outimage_at_index(self, index):
    return self.load_outimage(self._outimages[index])
  def set_start_step(self, start_step):
    self._index_in_epoch = start_step
    if self._index_in_epoch > self._num_samples:
      self._epochs_completed = self._index_in_epoch / self._num_samples
      print("epoch {} completed".format(self._epochs_completed))
      # Shuffle the data
      np.random.shuffle(self._indices)
      # Start next epoch
      start = 0
      self._index_in_epoch = 0
  def next_batch(self, batch_size):
    start = self._index_in_epoch
    self._index_in_epoch += batch_size
    if self._index_in_epoch > self._num_samples:
      # Finished epoch
      self._epochs_completed += 1
      print("epoch {} completed".format(self._epochs_completed))
      # Shuffle the data
      np.random.shuffle(self._indices)
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size
      assert batch_size <= self._num_samples
    end = self._index_in_epoch

    indices = self._indices[start:end].tolist()
    indices.sort()

    inimages  = [self.get_inimage_at_index(index)  for index in indices]
    outimages = [self.get_outimage_at_index(index) for index in indices]

    return inimages, outimages

def filter_inimage(width, height, image, i, j):
  im = np.array(image.read_region((i, j), LEVEL, (width, height)))[:,:,0:3]
  avg = np.sum(im)/width/height/3
  return avg < 220

def filter_outimage(width, height, image, i, j):
  im = np.array(image.read_region((i, j), LEVEL, (width, height)))[:,:,0]
  avg = np.sum(im)/width/height
  return avg > 0.0


def read_data_sets(width, height, start_layer, data_dir, load_train=True, load_test=True, start_step=0):
  LEVEL = start_layer

  class DataSets(object):
    pass
  data_sets = DataSets()

  inimages  = []
  outimages = []

  total_count = 0
  added_count = 0
  metastasis_count = 0
  missed_metastasis_count = 0

  for root, dirnames, filenames in os.walk(data_dir + 'training/'):
    for maskfile in fnmatch.filter(filenames, 'Tumor_*_Mask.tif'):
      imagefile = maskfile[:-9] + '.tif'
      print imagefile

      mask  = openslide.OpenSlide(data_dir + 'training/' + maskfile)
      image = openslide.OpenSlide(data_dir + 'training/' + imagefile)

      assert mask.level_dimensions[LEVEL] == image.level_dimensions[LEVEL], "Image and Mask dimensions are not equal for " + imagefile

      (w, h) = mask.level_dimensions[LEVEL]

      for i in range(w / width):
        for j in range(h / height):
          if filter_inimage(width, height, image, i*width*(2**LEVEL), j*height*(2**LEVEL)):
            if filter_outimage(width, height, mask, i*width*(2**LEVEL), j*height*(2**LEVEL)):
              for k in range(8):
                inimages.append((image, i*width*(2**LEVEL), j*height*(2**LEVEL), k))
                outimages.append((mask, i*width*(2**LEVEL), j*height*(2**LEVEL), k))
              metastasis_count += 8
              added_count += 8
              total_count += 8
            else:
              inimages.append((image, i*width*(2**LEVEL), j*height*(2**LEVEL), 0))
              outimages.append((mask, i*width*(2**LEVEL), j*height*(2**LEVEL), 0))
              added_count += 1
              total_count += 1
          else:
            if filter_outimage(width, height, mask, i*width*(2**LEVEL), j*height*(2**LEVEL)):
              missed_metastasis_count += 8
              total_count += 8
            else:
              total_count += 1


      print "Total Count: {}, Added Count: {}, Metastasis Count: {}, Missed Metastasis Count: {}".format(total_count, added_count, metastasis_count, missed_metastasis_count)

    for imagefile in fnmatch.filter(filenames, 'Normal_*.tif'):
      print imagefile

      image = openslide.OpenSlide(data_dir + 'training/' + imagefile)
      (w, h) = image.level_dimensions[LEVEL]

      for i in range(w / width):
        for j in range(h / height):
          if filter_inimage(width, height, image, i*width*(2**LEVEL), j*height*(2**LEVEL)):
            inimages.append((image, i*width*(2**LEVEL), j*height*(2**LEVEL), 0))
            outimages.append(None)
            added_count += 1
          total_count += 1

      print "Total Count: {}, Added Count: {}, Metastasis Count: {}, Missed Metastasis Count: {}".format(total_count, added_count, metastasis_count, missed_metastasis_count)


  num_samples = len(inimages)
  train_inimages = inimages
  train_outimages = outimages

  data_sets.train = DataSet(width, height, train_inimages, train_outimages, num_samples)
  data_sets.train.set_start_step(start_step)

  return data_sets
