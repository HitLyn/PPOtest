import multiprocessing
import gym
import numpy as np
import sys
import tensorflow as tf

from baselines.common.cmd_util import make_vec_env, make_env
from baselines.common.vec_env.vec_normalize import VecNormalize

try:
	from mpi4py import MPI
except ImportError:
	MPI = None

def build(env, num_envs):
	ncpu = multiprocessing.cpu_count()
	if sys.platform == 'darwin': ncpu //= 2
	nenv = ncpu or num_envs
	env_id = env
	env_type = 'robotics'
	env = make_vec_env(env_id, env_type, num_envs, None, reward_scale = 1.0, flatten_dict_observations = False)
	return env

if __name__ == '__main__':
	env = build('FetchReach-v1', 1)
	print(env)
