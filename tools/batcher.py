import numpy as np
import argparse

import scipy

from baselines.common.vec_env.vec_frame_stack import VecFrameStack
from baselines.common.cmd_util import make_atari_env

tf.reset_default_graph()

class Batcher():
	'''
	Instantiate a Batcher instance with env, actor, batch_size, and dataset_path
	batcher = Batcher(env, actor)

	Call batcher.create_dataset(num_iter) to create the dataset. The dataset will
	be saved into 4 files in the tmp directory. Be sure to "mkdir tmp"

	To generate data create a data generator and call next on it
	generator = batcher.data_generator()
	states, actions, next_states, rewards = next(generator)

	'''
	def __init__(self, env, actor, batch_size=32, dataset_path="tmp/dataset"):
		self.env = env
		self.actor = actor
		self.batch_size = batch_size
		self.dataset_path = dataset_path
		self.num_iter = 0

	def create_dataset(self, num_iter):
		self.num_iter = num_iter
		state = self.env.reset()
		self.actor.reset()
		for i in range(num_iter):
			action = [self.actor.act(scipy.misc.imresize(np.squeeze(state, axis=0), (42,42)))]
			next_state, reward, done, info = self.env.step(action)
			try:
				states
			except:
				states = state
			try:
				rewards
			except:
				rewards = reward
			try:
				actions
			except:
				actions = action
			try:
				next_states
			except:
				next_states = next_state

			states = np.concatenate((states, state), axis=0)
			actions = np.concatenate((actions, action), axis=0)
			next_states = np.concatenate((next_states, next_state), axis=0)
			rewards = np.concatenate((rewards, reward), axis=0)

			state = next_state
			if done[-1]:
				state = self.env.reset()

		np.save(self.dataset_path+"current_states", states)
		np.save(self.dataset_path+"actions", actions)
		np.save(self.dataset_path+"next_states", next_states)
		np.save(self.dataset_path+"rewards", rewards)


	def data_generator(self):
		# yields batch size amounts of data

		states = np.load(self.dataset_path+"current_states.npy")
		actions = np.load(self.dataset_path+"actions.npy")
		next_states = np.load(self.dataset_path+"next_states.npy")
		rewards = np.load(self.dataset_path+"rewards.npy")

		for i in range(0, self.num_iter, self.batch_size):
			if i+self.batch_size > self.num_iter:
				raise StopIteration
			yield (
				states[i:i+self.batch_size],
				actions[i:i+self.batch_size],
				next_states[i:i+self.batch_size],
				rewards[i:i+self.batch_size]
				)

class Actor():
	def __init__(self, config):
		self.config = config
		with tf.variable_scope("global"):
			self.network = LSTMPolicy([42, 42, 1], 6)
			saver = tf.train.Saver()
			self.sess = tf.Session()
			saver.restore(self.sess, "/tmp/pong/train/model.ckpt-3022926")
		self.last_features = self.network.get_initial_features()

	def act(self, state):
		assert state.shape == (42, 42, 1)
		with self.sess:
			stuff = self.network.act(state, *self.last_features)
			action, value_, features = stuff[0], stuff[1], stuff[2]
			self.last_features = features
		return action, value_

	def reset(self):
		self.last_features = self.network.get_initial_features()


parser = argparse.ArgumentParser(description='A3C')
parser.add_argument('--env', default = 'PongNoFrameskip-v4',
                    help='environment to test on')
parser.add_argument('--num_env', type=int, default = '1',
                    help='number of envs')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='learning rate')
parser.add_argument('--max_grad_norm', type=float, default=0.5,
                    help='maximum gradient norm for clipping')
parser.add_argument('--reuse', type=bool, default=False,
                    help='policy architecture')
parser.add_argument('--batch_size', type=int, default=128,
                    help='batch size')
parser.add_argument('--normalize_adv', type=bool, default=True,
                    help='normalize advantage')
parser.add_argument('--seed', type=int, default=123,
                	help='random seed')
parser.add_argument('--loss_weight', type=float, default=0.9,
                	help='environment loss weighting')
parser.add_argument('--total_updates', type=int, default=30000,
                	help='total number of updates')
parser.add_argument('--n_steps', type=int, default=1,
					help='use n-step loss for training')
parser.add_argument('--load_ckpt', type=str, default=None,
					help='path to checkpoint to load')
parser.add_argument('--save_ckpt', type=str, default="../ckpts/model.ckpt",
					help='path to save checkpoint')

if __name__ == '__main__':
	config = parser.parse_args()

	env = VecFrameStack(make_atari_env(config.env, config.num_env, config.seed), 4)
	actor = Actor(config)

	config.frame_dims = [84, 84]
	config.n_stacked = 4
	config.n_actions = int(env.action_space.n)
	config.scope = 'worker'


	batcher = Batcher(env, actor)
	batcher.create_dataset(310)

	generator = batcher.data_generator()
	while True:
		states, actions, next_states, rewards = next(generator)
		print("STATES SHAPE: {}".format(states.shape))
		print("ACTIONS SHAPE: {}".format(actions.shape))
		print("NEXT_STATES SHAPE: {}".format(next_states.shape))
		print("REWARDS SHAPE: {}".format(rewards.shape))






