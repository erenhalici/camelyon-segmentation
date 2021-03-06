
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

      width  = width/2
      height = height/2

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

    for i in range(layer_count - 1):
      width  = width*2
      height = height*2

      W_conv_1 = self.weight_variable([2, 2, filter_count, last_filter_count])
      b_conv_1 = self.bias_variable([filter_count])
      output_shape = tf.pack([batch_size, width, height, filter_count])
      h_conv_1 = tf.concat(3, [layers.pop(), tf.nn.relu(self.conv2d_transpose(last_h, W_conv_1, output_shape) + b_conv_1)])

      h_conv_2 = self.conv_layer(h_conv_1, last_filter_count, filter_count)
      h_conv_3 = self.conv_layer(h_conv_2, filter_count, filter_count)

      h_list.append(h_conv_1)
      h_list.append(h_conv_2)
      h_list.append(h_conv_3)

      last_filter_count = filter_count
      filter_count = filter_count / 2
      last_h = h_conv_3

    self._y = self.s_conv_layer(last_h, last_filter_count, num_output_channels)

    cross_entropy = -tf.reduce_sum(self._y_*tf.log(tf.clip_by_value(self._y,1e-10,1.0)))

    difference = self._y_ - self._y
    correct_prediction = tf.less(tf.abs(difference), 0.5)
    self._accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    error_sq = tf.reduce_mean(tf.square(difference))

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
  def error(self):
      return self._error

  def weight_variable(self, shape):
    initial = tf.truncated_normal(shape, stddev=0.04)
    return tf.Variable(initial)

  def bias_variable(self, shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

  def conv2d(self, x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="SAME")

  def conv2d_transpose(self, x, W, output_shape):
    return tf.nn.conv2d_transpose(x, W, output_shape=output_shape, strides=[1, 2, 2, 1], padding="SAME")

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
