from model import Model
from default_net import get_net
from runner import Runner
import env_builder
import policy_fn
from baselines import logger
from baselines.common import explained_variance


import gym
import numpy as np
import time

def get_const(x):
	def fn(_):
		return x
	return fn

def learn(env, total_steps = 1e6, load_path = None, net = 'mlp', n_steps = 512, n_agents = 1, train_loops = 4, train_batchsize = 128, cliprange = 0.2, lr = 1e-3, log_interval = 10):

	"""
	train_loops: The loop nums of every n_batch samples

	"""

	envrionment = env_builder.build(env, n_agents)
	network = get_net(net)
	### policy
	policy = policy_fn.build_policy(envrionment, network)
	### model
	model = Model(envrionment, policy, train_batchsize)
	if load_path is not None:
		model.load(load_path)
		return model, envrionment
	### runner
	dims = policy_fn.get_env_dims(env)
	runner = Runner(envrionment, model, n_steps, dims)

	n_batch = n_steps*n_agents
	n_updates = total_steps//n_batch

	l_r = get_const(lr)
	cliprange = get_const(cliprange)

	for update in np.arange(n_updates):

		tstart = time.time()

		frac = 1.0 - (update-1.0)/n_updates
		lr = l_r(frac)
		cliprangenow = cliprange(frac)

		ob, returns, done, action, value, neglogpa, _ = runner.run()
		loss = []
		for _ in np.arange(train_loops):
			index = np.arange(n_batch)
			np.random.shuffle(index)
			for begin in np.arange(0, n_batch, train_batchsize):
				end = begin + train_batchsize
				mbinds = index[begin:end]
				samples = [array[mbinds] for array in (ob, returns, action, value, neglogpa)]
				# print('Begin to trian with samples:', begin)
				loss.append(model.train(lr, cliprangenow, *samples))

		loss = np.mean(loss, axis = 0)
		# print('loss:', loss)

		tnow = time.time()
		fps = int(n_batch/(tnow - tstart))

		if update % log_interval == 0 or update == 1:
			ev = explained_variance(value, returns)
			logger.logkv('serial_timesteps', update * n_steps)
			logger.logkv('nupdates', update)
			logger.logkv('total_time_steps', update * n_batch)
			logger.logkv('fps', fps)
			logger.logkv('explained_variance', float(ev))
			for (lossval, lossname) in zip(loss, model.loss_names):
				logger.logkv(lossname, lossval)
			print(model.loss_names, loss)

	return model, envrionment
