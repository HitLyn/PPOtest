import numpy as np

def get_concat_obs(obs_dict):
	return np.concatenate((obs_dict['observation'], obs_dict['desired_goal']), axis = -1)


class Runner():
	def __init__(self, env, model, nsteps, dims, gamma = 0.99, lam = 0.95):
		self.env = env
		self.model = model
		self.nsteps = nsteps
		self.dones = [False for _ in range(self.env.num_envs)]
		self.lam = lam
		self.gamma = gamma
		self.act_batchsize = env.num_envs

		### observation is different from other envs, need to be reset
		self.dims = dims
		self.env_reset()

	def env_reset(self):
		self.obs_dict = self.env.reset()
		self.initial_o = self.obs_dict['observation']
		self.initial_ag = self.obs_dict['achieved_goal']
		self.goal = self.obs_dict['desired_goal']
		self.obs = get_concat_obs(self.obs_dict)




	def run(self):
		### initialize some list
		obs, rewards, actions, values, dones, neglogpas = [], [], [], [], [], []
		successes = []

		for _ in range(self.nsteps):
			### given observations, get actions, values and neglogpas
			_actions, _values, _neglogpas = self.model.step(self.obs)
			obs.append(self.obs.copy())
			actions.append(_actions)
			values.append(_values)
			neglogpas.append(_neglogpas)
			dones.append(self.dones)
			### take actions int env and record the results
			self.obs_dict, _rewards, self.dones, _infos = self.env.step(_actions)
			# print(_actions)
			# self.env.render()
			self.obs = get_concat_obs(self.obs_dict)
			rewards.append(_rewards)
			# for info in _infos:
			# 	maybeepinfo = info.get('episode')
			# 	if maybeepinfo: epinfos.append(maybeepinfo)
			success = np.array([info.get('is_success', 0) for info in _infos])
			successes.append(success)


		obs = np.asarray(obs, dtype = self.obs.dtype)
		rewards = np.asarray(rewards, dtype = np.float32)
		actions = np.asarray(actions, dtype = np.float32)
		values = np.asarray(values, dtype = np.float32)
		neglogpas = np.asarray(neglogpas, dtype = np.float32)
		dones = np.asarray(dones, dtype = np.bool)
		last_values = self.model.value(self.obs)
		successes = np.asarray(successes, dtype = np.float32)

		### discount
		returns = np.zeros_like(rewards)
		advs = np.zeros_like(rewards)
		lastgaelam = 0
		for t in reversed(range(self.nsteps)):
			if t == self.nsteps - 1:
				nextnonterminals = 1 - self.dones
				nextvalues = last_values
			else:
				nextnonterminals = 1 - dones[t + 1]
				nextvalues = values[t + 1]
			delta = rewards[t] + self.gamma * nextvalues * nextnonterminals - values[t]
			advs[t] = lastgaelam = delta + self.gamma * self.lam * nextnonterminals * lastgaelam
		returns = advs + values
		return (*map(sf01, (obs, returns, dones, actions, values, neglogpas)), successes)

def sf01(arr):
	s = arr.shape
	return arr.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:])
