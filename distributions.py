import numpy as np
from gym import spaces
import tensorflow as tf

def make_pdtype(action_space):
	if isinstance(action_space, spaces.Box):
		return DiagGaussianPdType(action_space.shape[0])
	if isinstance(action_space, spaces.Discrete):
		return CateforicalPdType(action_space)
	else:
		raise NotImplementedError

def get_mean_from_latent(latent, size):
	"""
	latent: shape(None, 64) vector
	size: the action space size which we need to turn latent into shape(None, size)

	"""

	if latent.shape[-1] == size:
		return latent
	else:
		return tf.layers.dense(latent, size)


class DiagGaussianPdType():
	def __init__(self, size):
		#### size : size of the action space
		self.size = size
	def pd_from_latent(self, latent):
		mean = get_mean_from_latent(latent, self.size)
		logstd = tf.get_variable(name = 'logstd', shape = [1, self.size], dtype = tf.float32, initializer = tf.zeros_initializer())
		pdparam = tf.concat([mean, mean*0.0 + logstd], axis = 1)
		return self.pdclass()(pdparam)
	def pdclass(self):
		return DiagGaussianPd
	def sample_placeholder(self, batch_size):
		### return action placeholders for model
		return tf.placeholder(dtype = tf.float32, shape = batch_size + [self.size])

class DiagGaussianPd():
	def __init__(self, pdparam):
		self.pdparam = pdparam
		mean, logstd = tf.split(axis = len(pdparam.shape) - 1, num_or_size_splits = 2, value = pdparam)
		self.mean = mean
		self.logstd = logstd
		self.std = tf.exp(logstd)
	def neglogp(self, x):
		return 0.5 * tf.reduce_sum(tf.square((x - self.mean) / self.std), axis=-1) \
               + 0.5 * np.log(2.0 * np.pi) * tf.to_float(tf.shape(x)[-1]) \
               + tf.reduce_sum(self.logstd, axis=-1)

	def sample(self):
		return self.mean + self.std * tf.random_normal(tf.shape(self.mean))

	def entropy(self):
		return tf.reduce_sum(self.logstd + 0.5 * np.log(2.0 * np.pi * np.e), axis = -1)
