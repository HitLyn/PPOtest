import sys
import os
import argparse

import ppo
from baselines import logger
import env_builder
from runner import get_concat_obs




def main():
	parser = argparse.ArgumentParser()
	# parser.add_argument('algo', type = str, default = 'PPO')
	parser.add_argument('--env', type = str, default = 'FetchReach-v1')
	parser.add_argument('--save_path', type = str, default = '~/Trained_model/FetchReach_PPO_1000000')
	parser.add_argument('--play', type = bool, default = None)
	parser.add_argument('--load_path', type = str, default = None)

	args = parser.parse_args()

	env = args.env
	# algo = args.algo

	# print('Ready to train...')

	model, env = ppo.learn(env = env, load_path = args.load_path)
	env.close()

	if args.save_path:
		path = os.path.expanduser(args.save_path)
		model.save(path)

	if args.play:
		logger.log('Running trainned model...')
		env_replay = env_builder.build(args.env, 1)
		obs_dict = env_replay.reset()
		i = 0
		while True:
			actions, _, _, = model.step(get_concat_obs(obs_dict))
			obs_dict, _, done, _ = env_replay.step(actions)

			env_replay.render()
			if done:
				obs_dict = env.reset()

		env.close()







if __name__ == '__main__':
	main()
