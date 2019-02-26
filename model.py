import tensorflow as tf
import numpy as np
import gym
import functools

from baselines.common.tf_util import initialize, get_session, save_variables, load_variables

# from policy_fn import dict_unwrapper


# ALREADY_INITIALIZED = set()
#
# def initializer():
# 	new_variables = set(tf.global_variables()) - ALREADY_INITIALIZED
# 	tf.get_default_session().run(tf.variables_initializer(new_variables))
# 	ALREADY_INITIALIZED.update(new_variables)

class Model():
	"""
	The PPO training model is built here:
	there will be two different inner model
	(trainning model for trainning and update and running model for runner)

	__init__:
	- create act-model
	- create train-model

	train():
	- backpropagation
	- update parameters

	save/load():
	waiting to be completed!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	- save and load model

	"""
	def __init__(self, env, policy, batch_size, grad_coef = 100, vf_coef = 0.01, ent_coef = 0.0, max_grad_norm = 0.5):
		# batch_size: train_batchsize, batch_size for the train model
		# act_batchsize is the num_envs

		self.sess = get_session()
		self.env = env
		self.observation_space = env.observation_space
		self.action_space = env.action_space

		with tf.variable_scope('ppo-model', reuse = tf.AUTO_REUSE):
			self.act_model = policy(env.num_envs, self.sess)
			self.train_model = policy(batch_size, self.sess)

		### create necessary placeholders
		self.A = self.train_model.pdtype.sample_placeholder([None])
		self.ADV = tf.placeholder(dtype = tf.float32, shape = [None])
		self.R = tf.placeholder(dtype = tf.float32, shape = [None])

		### old actor
		self.OldNegLogPa = tf.placeholder(dtype = tf.float32, shape = [None]) # one action to one logp (through algorithm to sum all the action dims once)
		### old critic
		self.OldVf = tf.placeholder(dtype = tf.float32, shape = [None])
		self.LR = tf.placeholder(dtype = tf.float32, shape = [])

		### cliprange
		self.ClipRange = tf.placeholder(dtype = tf.float32, shape = [])

		neglogpa = self.train_model.pd.neglogp(self.A)

		### entropy
		entropy = tf.reduce_mean(self.train_model.pd.entropy())

		### loss
		### Total loss = gradient loss + valuecoefficient * valueloss - entropycoefficient * entropy

		### value loss
		vpred = self.train_model.vf
		vpredclipped = self.OldVf + tf.clip_by_value(self.train_model.vf - self.OldVf, -self.ClipRange, self.ClipRange)

		### unclipped Value
		vf_loss1 = tf.square(vpred - self.R)
		vf_loss2 = tf.square(vpredclipped - self.R)
		vf_loss = 0.5 * tf.reduce_mean(tf.maximum(vf_loss1, vf_loss2))

		### calculate gradient loss (pi current policy/old policy)
		ratio = tf.exp(self.OldNegLogPa - neglogpa)

		pg_loss1 = -self.ADV * ratio
		pg_loss2 = -self.ADV * tf.clip_by_value(ratio, 1.0 - self.ClipRange, 1 + self.ClipRange)
		pg_loss = 0.5 * tf.reduce_mean(tf.maximum(pg_loss1, pg_loss2))

		### Total loss
		loss = pg_loss * grad_coef - entropy * ent_coef + vf_loss * vf_coef

		### update parameters
		params = tf.trainable_variables('ppo-model')

		### build trainer
		self.trainer = tf.train.AdamOptimizer(learning_rate = self.LR, epsilon = 1e-5)

		### calculate gradients
		grads_and_var = self.trainer.compute_gradients(loss, params)
		grads, var = zip(*grads_and_var)

		grads, _grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
		grads_and_var = list(zip(grads, var))

		self.grads = grads
		self.var = var
		self._train_op = self.trainer.apply_gradients(grads_and_var)
		self.loss_names = ['gradient_loss', 'value_loss', 'policy_entropy_loss']
		self.stats_list = [pg_loss, vf_loss, entropy]

		self.step = self.act_model.step
		self.value = self.act_model.value

		self.save = functools.partial(save_variables, sess=self.sess)
		self.load = functools.partial(load_variables, sess=self.sess)

		initialize()

	def train(self, lr, cliprange, obs, returns, actions, values, neglogpas):
		### calculate advs
		advs = returns - values

		### normalize the advs
		advs = (advs - advs.mean())/(advs.std() + 1e-8)

		# obs = dict_unwrapper(obs_dict)

		### feed_dict
		feed_dict = {
		self.train_model.ob:obs,
		self.A:actions,
		self.ADV: advs,
		self.R:returns,
		self.LR:lr,
		self.ClipRange:cliprange,
		self.OldNegLogPa:neglogpas,
		self.OldVf:values
		}

		return self.sess.run(self.stats_list + [self._train_op], feed_dict)[:-1]
