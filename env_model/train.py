import tensorflow as tf
import numpy as np
import argparse
import gym
import tensorflow.contrib.layers as layers

from baselines.common.vec_env.vec_frame_stack import VecFrameStack
from baselines.common.cmd_util import make_atari_env

class EnvironmentModel(object):
	def __init__(self, config):
		self.x = tf.placeholder(tf.float32, [None] + config.frame_dims + [config.n_stacked+config.n_actions])
		self.next_states = tf.placeholder(tf.float32, [None] + config.frame_dims + [config.n_stacked])
		self.rewards = tf.placeholder(tf.float32, [None,])
		# extract features
		with tf.variable_scope(config.scope):
			x_padded = tf.pad(self.x, tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]]))
			conv1 = layers.conv2d(x_padded, 64, 8, stride=2, padding='valid')
			conv1_padded = tf.pad(conv1, tf.constant([[0, 0], [2, 2], [2, 2], [0, 0]]))
			conv2 = layers.conv2d(conv1_padded, 128, 6, stride=2, padding='valid')
			conv2_padded = tf.pad(conv2, tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]]))
			conv3 = layers.conv2d(conv2_padded, 128, 4, stride=2, padding='valid')
			feats = layers.flatten(conv3)
			fc1 = layers.fully_connected(feats, 2048)
			fc2 = layers.fully_connected(fc1, 2048)
			# get the predicted reward
			self.pred_reward = tf.squeeze(layers.fully_connected(fc2, 1, activation_fn=None))
			# get the predicted next state
			fc3 = layers.fully_connected(fc2, 2048)
			fc4 = layers.fully_connected(fc3, 12800)
			fc4 = tf.reshape(fc4, [-1, 10, 10, 128])
			upconv3 = layers.conv2d_transpose(fc4, 128, 4, stride=2, padding='same')
			upconv2 = layers.conv2d_transpose(upconv3, 128, 6, stride=2, padding='same')
			upconv2_padded = tf.pad(upconv2, tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]]))
			self.pred_state = layers.conv2d_transpose(upconv2_padded, config.n_stacked, 8, stride=2, activation_fn=None, padding='same')

		self.loss =  config.loss_weight * tf.losses.mean_squared_error(self.next_states, self.pred_state) + \
					(1 - config.loss_weight) * tf.losses.mean_squared_error(self.rewards, self.pred_reward)

		variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, config.scope)
		gradients = tf.gradients(self.loss, variables)
		optimizer = tf.train.AdamOptimizer(learning_rate=config.lr)
		gradients, _ = tf.clip_by_global_norm(gradients, config.max_grad_norm)
		grads_and_vars = zip(gradients, variables)
		self.train_op = optimizer.apply_gradients(grads_and_vars)

	def predict(self, state, action):
		action_tile = np.zeros((*config.frame_dims, config.n_actions))
		action_tile[:, :, action] = 1
		stacked = np.concatenate([state, action_tile], axis=-1)
		print(stacked.shape)
		pred_state, pred_reward = sess.run(
			[self.pred_state, self.pred_reward],
			feed_dict={
				self.x: stacked
			})
		return pred_state, pred_reward

	def train(self, s, a, r, s_prime):
		s = np.stack(s, axis=0)
		a = np.array(a)
		r = np.array(r)
		s_prime = np.stack(s_prime, axis=0)

		action_tile = np.zeros((config.batch_size, *config.frame_dims, config.n_actions))
		for idx, act in enumerate(a):
			action_tile[idx, :, :, act] = 1
		stacked = np.concatenate([s, action_tile], axis=-1)


		loss, _ = sess.run(
			[self.loss, self.train_op],
			feed_dict={
				self.x: stacked,
				self.rewards: r,
				self.next_states: s_prime
			})
		return loss

class Actor():
	def __init__(self, config, sess):
		self.config = config
		# self.sess = sess
		# saver = tf.train.import_meta_graph(config.meta_graph)
		# saver.restore(sess,tf.train.latest_checkpoint('./'))
		# graph = tf.get_default_graph()
		# self.action = graph.get_tensor_by_name('action')

	def act(self, state):
		action = np.random.choice(list(range(config.n_actions)))
		# self.inputs = graph.get_tensor_by_name('inputs:0')
		# action = self.sess.run(
		# 	[self.action],
		# 	feed_dict={
		# 		self.inputs: state
		# 	})
		return action

class Worker():
	def __init__(self, config, env):
		self.actor = Actor(config, sess)
		self.env = env

	def get_batch(self, batch_size):
		states = []
		rewards = []
		actions = []
		next_states = []

		state = self.env.reset()
		for i in range(batch_size):
			action = [self.actor.act(state)]
			next_state, reward, done, info = self.env.step(action)

			next_states.append(np.squeeze(next_state))
			actions.append(np.squeeze(action))
			rewards.append(np.squeeze(reward))
			states.append(np.squeeze(state))

			state = next_state

			if done:
				state = self.env.reset()

		return states, actions, rewards, next_states

def run(sess, config, env):
	worker = Worker(config, env)
	model = EnvironmentModel(config)
	sess.run(tf.global_variables_initializer())
	sess.run(tf.local_variables_initializer())
	for i in range(config.total_updates // config.batch_size):
		states, actions, rewards, next_states = worker.get_batch(config.batch_size)
		loss = model.train(states, actions, rewards, next_states)

		print('Batch {}-{}: Loss {}'.format(i * config.batch_size, (i + 1) * config.batch_size, loss))

parser = argparse.ArgumentParser(description='A3C')
parser.add_argument('--env', default = 'PongNoFrameskip-v4',
                    help='environment to test on')
parser.add_argument('--num_env', type=int, default = '1',
                    help='number of envs')
parser.add_argument('--vf_coeff', type=float, default=0.5,
                    help='value function loss coefficient')
parser.add_argument('--entropy_coeff', type=float, default=0.01,
                    help='entropy loss coefficient')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='learning rate')
parser.add_argument('--max_grad_norm', type=float, default=5,
                    help='maximum gradient norm for clipping')
parser.add_argument('--gamma', type=float, default=0.99,
                    help='discount factor')
parser.add_argument('--history', type=int, default=4,
                    help='number of frames in the past to keep in the state')
parser.add_argument('--policy_type', default='cnn',
                    help='policy architecture')
parser.add_argument('--reuse', type=bool, default=False,
                    help='policy architecture')
parser.add_argument('--batch_size', type=int, default=64,
                    help='batch size')
parser.add_argument('--normalize_adv', type=bool, default=True,
                    help='normalize advantage')
parser.add_argument('--seed', type=int, default=1234,
                	help='random seed')
parser.add_argument('--loss_weight', type=float, default=0.5,
                	help='environment loss weighting')
parser.add_argument('--total_updates', type=int, default=10000,
                	help='total number of updates')

if __name__ == '__main__':
	config = parser.parse_args()

	env = VecFrameStack(make_atari_env(config.env, config.num_env, config.seed), 4)
	sess = tf.Session()

	config.frame_dims = [84, 84]
	config.n_stacked = 4
	config.n_actions = env.action_space.n
	config.scope = 'worker'
	config.reuse = tf.AUTO_REUSE

	run(sess, config, env)
