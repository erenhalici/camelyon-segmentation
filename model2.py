
import tensorflow as tf

W_list = []
b_list = []
h_list = []

def lists():
  return (W_list, b_list, h_list)

class Model(object):
  def __init__(self, width, height, num_input_channels, num_output_channels, filter_count, layer_count, learning_rate=1e-4):
    self._x_image = tf.placeholder(tf.float32, [None, width, height, num_input_channels])
    self._y_ = tf.placeholder(tf.float32, [None, width, height, num_output_channels])

    batch_size = tf.shape(self._x_image)[0]

    last_filter_count = num_input_channels
    last_h = self._x_image

    layers = []

    for i in range(layer_count - 1):
      h_conv_1 = self.conv_layer(last_h, last_filter_count, filter_count)
      h_list.append(h_conv_1)
      h_conv_2 = self.conv_layer(h_conv_1, filter_count, filter_count)
      h_list.append(h_conv_2)
      layers.append(h_conv_2)
      h_pool   = self.max_pool_2x2(h_conv_2)

      width  = width/2 - 2
      height = height/2 - 2

      print width, height, h_pool.get_shape()

      last_filter_count = filter_count
      filter_count = filter_count * 2
      last_h = h_pool

    h_conv_1 = self.conv_layer(last_h, last_filter_count, filter_count)
    h_conv_2 = self.conv_layer(h_conv_1, filter_count, filter_count)
    h_list.append(h_conv_1)
    h_list.append(h_conv_2)

    last_filter_count = filter_count
    filter_count = last_filter_count/2
    last_h = h_conv_2

    width -= 4
    height -= 4

    print width, height, last_h.get_shape()

    for i in range(layer_count - 1):
      width  = width*2
      height = height*2

      upsampled = tf.image.resize_bilinear(last_h, [width, height])

      width -= 2
      height -= 2

      old_layer = layers.pop()
      old_shape = old_layer.get_shape().as_list()
      old_w, old_h = (old_shape[1], old_shape[2])

      cropped = tf.slice(old_layer, [0, (old_w - width) / 2, (old_h - height) / 2,0], [-1, width, height, -1])
      h_conv_1 = tf.concat(3, [cropped, self.conv_layer(upsampled, last_filter_count, filter_count)])
      # h_conv_1 = tf.concat(3, [cropped, upsampled])
      h_conv_2 = self.conv_layer(h_conv_1, last_filter_count, filter_count)
      # h_conv_2 = self.conv_layer(h_conv_1, last_filter_count + filter_count, filter_count)
      h_conv_3 = self.conv_layer(h_conv_2, filter_count, filter_count)

      h_list.append(h_conv_1)
      h_list.append(h_conv_2)
      h_list.append(h_conv_3)

      last_filter_count = filter_count
      filter_count = filter_count / 2
      last_h = h_conv_3

      width -= 4
      height -= 4

      print width, height, last_h.get_shape()

    self._y = self.s_conv_layer(last_h, last_filter_count, num_output_channels)

    width -= 2
    height -= 2

    old_shape = self._y_.get_shape().as_list()
    old_w, old_h = (old_shape[1], old_shape[2])
    mask = tf.slice(self._y_, [0, (old_w - width) / 2, (old_h - height) / 2,0], [-1, width, height, -1])

    cross_entropy = -tf.reduce_sum(mask*tf.log(tf.clip_by_value(self._y,1e-10,1.0)))

    difference = mask - self._y
    error_sq = tf.reduce_mean(tf.square(difference))


    mask_positive = tf.greater_equal(mask, 0.5)
    mask_negative = tf.logical_not(mask_positive)
    y_positive = tf.greater_equal(self._y, 0.5)
    y_negative = tf.logical_not(y_positive)

    tp = tf.logical_and(mask_positive, y_positive)
    tn = tf.logical_and(mask_negative, y_negative)
    fp = tf.logical_and(mask_negative, y_positive)
    fn = tf.logical_and(mask_positive, y_negative)

    self._tp = tf.reduce_mean(tf.cast(tp, "float"))
    self._tn = tf.reduce_mean(tf.cast(tn, "float"))
    self._fp = tf.reduce_mean(tf.cast(fp, "float"))
    self._fn = tf.reduce_mean(tf.cast(fn, "float"))

    correct_prediction = tf.less(tf.abs(difference), 0.5)
    self._accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))


    self._train_step = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-4).minimize(error_sq)
    self._error = tf.sqrt(error_sq)

  @property
  def x_image(self):
    return self._x_image
  @property
  def y(self):
    return self._y
  @property
  def y_(self):
    return self._y_
  @property
  def train_step(self):
    return self._train_step
  @property
  def accuracy(self):
      return self._accuracy
  @property
  def tp(self):
    return self._tp
  @property
  def tn(self):
    return self._tn
  @property
  def fp(self):
    return self._fp
  @property
  def fn(self):
    return self._fn
  @property
  def error(self):
    return self._error

  def weight_variable(self, shape):
    initial = tf.truncated_normal(shape, stddev=0.04)
    return tf.Variable(initial)

  def bias_variable(self, shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

  def conv2d(self, x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="VALID")

  def conv2d_transpose(self, x, W, output_shape):
    return tf.nn.conv2d_transpose(x, W, output_shape=output_shape, strides=[1, 2, 2, 1], padding="VALID")

  def max_pool_2x2(self, x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

  def conv_layer(self, input_layer, input_channes, output_channels):
    W_conv = self.weight_variable([3, 3, input_channes, output_channels])
    b_conv = self.bias_variable([output_channels])
    W_list.append(W_conv)
    b_list.append(b_conv)
    return tf.nn.relu(self.conv2d(input_layer, W_conv) + b_conv)
  def s_conv_layer(self, input_layer, input_channes, output_channels):
    W_conv = self.weight_variable([3, 3, input_channes, output_channels])
    b_conv = self.bias_variable([output_channels])
    return tf.sigmoid(self.conv2d(input_layer, W_conv) + b_conv)
