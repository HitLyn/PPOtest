import numpy as np
import tensorflow as tf
import gym

from distributions import make_pdtype

# def dict_unwrapper(obs_dict):
# 	return np.concatenate((obs_dict['observation'], obs_dict['desired_goal']), axis = -1)

def get_env_dims(env_name):
	env = gym.make(env_name)
	env.reset()
	obs, r, d, info = env.step(env.action_space.sample())
	dims = {'o': obs['observation'].shape[0],
			'a': env.action_space.shape[0],
			'g': obs['desired_goal'].shape[0]}

	return dims

def build_placeholder(env_name, batch_size):
	"""
	especially for Robotic Env
	the observation_space is different form other environments
	for more detail refer to: https://blog.openai.com/ingredients-for-robotics-research/
	"""

	dims = get_env_dims(env_name)

	# placeholder_o = tf.placeholder(dtype = tf.float32, shape = (batch_size, dims['o']))
	# placeholder_a = tf.placeholder(dtype = tf.float32, shape = (batch_size, dims['a']))
	# placeholder_g = tf.placeholder(dtype = tf.float32, shape = (batch_size, dims['g']))

	placeholder = tf.placeholder(dtype = tf.float32, shape = (batch_size, dims['o'] + dims['g']), name = 'obervation_placeholder')
	return placeholder


def get_policy(net, ob):
	return net(ob)


class Policy():
	def __init__(self, env, latent, vf_latent, ob, sess):

		self.sess = sess
		self.latent = tf.layers.flatten(latent)
		self.vf_latent = tf.layers.flatten(vf_latent)
		self.ob = ob

		self.pdtype = make_pdtype(env.action_space)
		self.pd = self.pdtype.pd_from_latent(latent)

		### get action
		self.action = self.pd.sample()
		### get neglogp of the chosen action
		self.neglogp = self.pd.neglogp(self.action)

		self.vf = tf.layers.dense(self.latent, 1)
		self.vf = self.vf[:,0]


	def step(self, observations):
		return self._evaluate([self.action, self.vf, self.neglogp], observations)

	def _evaluate(self, variables, observations):
		# np_observations = dict_unwrapper(observations_dict)
		feed_dict = {self.ob:observations} ############ self.ob(placeholder) and observations(inputdate) must be the same size
		return self.sess.run(variables, feed_dict)

	def value(self, observations):
		return self._evaluate(self.vf, observations)



def build_policy(env, net):
	if env is not None:   ##
		def get_policy_object(batch_size, sess):
			ob = build_placeholder(env.specs[0].id, batch_size)
			X = ob
			with tf.variable_scope('pi', reuse = tf.AUTO_REUSE):
				policy_latent = net()(X)

			vf_latent = policy_latent ####### value network

			policy = Policy(env = env, latent = policy_latent, vf_latent = vf_latent, ob = X, sess = sess)


			return policy

		return get_policy_object
