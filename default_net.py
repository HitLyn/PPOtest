import tensorflow as tf
import numpy as np

mapping = {}

def get_net(name):
	return mapping[name]

def register(name):
	def _thunk(fn):
		mapping[name] = fn
		return fn
	return _thunk

def nature_cnn(image, **kwargs):
	# normalization
	image = tf.cast(image, tf.float32)/255
	with tf.variable_scope('conv1'):
		layer1 = tf.layers.conv2d(image, 32, [8, 8], 4, activation = tf.nn.relu)
	with tf.variable_scope('conv2'):
		layer2 = tf.layers.conv2d(layer1, 64, [4, 4], 2, activation = tf.nn.relu)
	with tf.variable_scope('conv3'):
		layer3 = tf.layers.conv2d(layer2, 64, [3, 3], 1, activation = tf.nn.relu)

	fc1 = conv_to_fc(layer3)
	fc1 = tf.layers.dense(fc1, 512, activation = tf.nn.relu)

	return fc1

def dense_net(obs, num_layers = 2, num_hidden = 64, layer_norm = False, **kwargs):
	layer = tf.layers.flatten(obs)
	for i in np.arange(num_layers):
		layer = tf.layers.dense(layer, num_hidden)
		if layer_norm:
			layer = tf.contrib.layers.layer_norm(layer, center = True, scale = True)
		layer = tf.tanh(layer)
	return layer

def conv_to_fc(image):
	x = np.prod([d.value for d in image.get_shape()[1:]])
	return(tf.reshape(image, [-1, x]))

### net for some picture games, image as input ##########
@register('cnn')
def cnn(**kwargs):
	def cnn_f(image):
		return naturn_cnn(image, **kwargs)
	return cnn_f

### net for robotics, joint pos, vel and goal states as input
## it is actually the tf.layers.densenet for some picture games,
@register('mlp')
def mlp(**kwargs):
	def mlp_fn(obs):
		return dense_net(obs, **kwargs)
	return mlp_fn


if __name__ == '__main__':
	print(mapping)
