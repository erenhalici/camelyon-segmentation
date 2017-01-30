
import numpy as np
import openslide
import fnmatch
import os

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
    return self.augment_image(np.array(image.read_region((i, j), 2, (self._width, self._height)))[:,:,0:3], k)
  def load_outimage(self, image_data):
    if image_data == None:
      return np.zeros([self._width,self._height,1])
    else:
      (image, i, j, k) = image_data
      return self.augment_image(np.array(image.read_region((i, j), 2, (self._width, self._height)))[:,:,0].reshape([self._width,self._height,1]), k)
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
    # inimages  = self._pool.map(self.get_inimage_at_index,   indices)
    # outimages = self._pool.map(self.get_outimage_at_index,  indices)

    return inimages, outimages

def filter_inimage(width, height, image, i, j):
  im = np.array(image.read_region((i, j), 2, (width, height)))[:,:,0:3]
  avg = np.sum(im)/width/height/3
  return avg < 220

def filter_outimage(width, height, image, i, j):
  im = np.array(image.read_region((i, j), 2, (width, height)))[:,:,0]
  avg = np.sum(im)/width/height
  return avg > 0.0


def read_data_sets(width, height, data_dir, load_train=True, load_test=True, start_step=0):
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

      assert mask.level_dimensions[2] == image.level_dimensions[2], "Image and Mask dimensions are not equal for " + imagefile

      (w, h) = mask.level_dimensions[2]

      for i in range(w / width):
        for j in range(h / height):
          if filter_inimage(width, height, image, i*width*4, j*height*4):
            if filter_outimage(width, height, mask, i*width*4, j*height*4):
              for k in range(8):
                inimages.append((image, i*width*4, j*height*4, k))
                outimages.append((mask, i*width*4, j*height*4, k))
              metastasis_count += 8
              added_count += 8
              total_count += 8
            else:
              inimages.append((image, i*width*4, j*height*4, 0))
              outimages.append((mask, i*width*4, j*height*4, 0))
              added_count += 1
              total_count += 1
          else:
            if filter_outimage(width, height, mask, i*width*4, j*height*4):
              missed_metastasis_count += 8
              total_count += 8
            else:
              total_count += 1


      print "Total Count: {}, Added Count: {}, Metastasis Count: {}, Missed Metastasis Count: {}".format(total_count, added_count, metastasis_count, missed_metastasis_count)

    for imagefile in fnmatch.filter(filenames, 'Normal_*.tif'):
      print imagefile

      image = openslide.OpenSlide(data_dir + 'training/' + imagefile)
      (w, h) = image.level_dimensions[2]

      for i in range(w / width):
        for j in range(h / height):
          if filter_inimage(width, height, image, i*width*4, j*height*4):
            inimages.append((image, i*width*4, j*height*4, 0))
            outimages.append(None)
            added_count += 1
          total_count += 1

      print "Total Count: {}, Added Count: {}, Metastasis Count: {}, Missed Metastasis Count: {}".format(total_count, added_count, metastasis_count, missed_metastasis_count)


  num_samples = len(inimages)
  TEST_SIZE = 32

  test_inimages = inimages[-TEST_SIZE:]
  test_outimages = outimages[-TEST_SIZE:]

  train_inimages = inimages[:-TEST_SIZE]
  train_outimages = outimages[:-TEST_SIZE]

  data_sets.train = DataSet(width, height, train_inimages, train_outimages, num_samples - TEST_SIZE)
  data_sets.train.set_start_step(start_step)

  data_sets.test  = DataSet(width, height, test_inimages,  test_outimages, TEST_SIZE)

  return data_sets
