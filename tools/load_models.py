import tensorflow as tf
import sys, time
import numpy as np
from policies import CnnPolicy
from baselines.common.vec_env.vec_frame_stack import VecFrameStack
from baselines.common.cmd_util import make_atari_env
import tensorflow.contrib.layers as layers

import matplotlib.pyplot as plt

tf.reset_default_graph()
sess = tf.Session()

class EnvironmentModel(object):
	def __init__(self, config):
		sess = tf.get_default_session()
		self.config = config
		self.x = tf.placeholder(tf.float32, [None] + config.frame_dims + [config.channels + config.n_actions])
		self.build_graph()
		variables_to_restore = [v for v in tf.global_variables() if v.name.startswith("worker")]
		saver = tf.train.Saver(variables_to_restore)
		saver.restore(sess, config.ckpt_file)

	def build_graph(self):
		with tf.variable_scope('worker', reuse=self.config.reuse):
			# extract features
			conv1 = layers.conv2d(self.x, 32, 8, stride=2)
			conv2 = layers.conv2d(conv1, 32, 3)
			conv3 = layers.conv2d(conv2, 32, 3, activation_fn=None)
			conv3 = tf.nn.relu(conv3 + conv1)
			conv4 = layers.conv2d(conv3, 32, 3)
			conv5 = layers.conv2d(conv2, 32, 3, activation_fn=None)
			encoded = tf.nn.relu(conv3 + conv5)

			# predict state
			self.pred_state = layers.conv2d_transpose(encoded, 1, 8, stride=2, activation_fn=None)

			# predict reward
			rconv1 = layers.conv2d(encoded, 32, 3)
			rpool1 = layers.max_pool2d(rconv1, 2)
			rconv2 = layers.conv2d(rpool1, 32, 3)
			rpool2 = layers.max_pool2d(rconv1, 2, padding="same")
			rpool2 = layers.flatten(rpool2)
			self.pred_reward = layers.fully_connected(rpool2, 1, activation_fn = None)

# env = VecFrameStack(make_atari_env('PongNoFrameskip-v4', 1, 123), 4)
# ob_space = env.observation_space
# ac_space = env.action_space

# class EnvironmentModel(object):
# 	def __init__(self, config):
# 		self.states_placeholder = tf.placeholder(tf.float32, [None] + config.frame_dims + [config.n_stacked])
# 		self.actions_placeholder = tf.placeholder(tf.int32, [None, config.n_steps])
# 		self.nf_placeholder = tf.placeholder(tf.float32, [None] + config.frame_dims + [config.n_steps])
# 		self.rewards_placeholder = tf.placeholder(tf.float32, [None, config.n_steps])

# 		dataset = tf.data.Dataset.from_tensor_slices({
# 			"states": self.states_placeholder,
# 			"actions": self.actions_placeholder,
# 			"rewards": self.rewards_placeholder,
# 			"next_frames": self.nf_placeholder
# 		})
# 		dataset = dataset.shuffle(config.seed).repeat(config.num_epochs).batch(config.batch_size)
# 		self.iterator = dataset.make_initializable_iterator()
# 		it_dict = self.iterator.get_next()

# 		self.states = it_dict["states"]
# 		self.actions = it_dict["actions"]
# 		self.rewards = it_dict["rewards"]
# 		self.next_frames = it_dict["next_frames"]

# 		current_state = self.states
# 		pred_frames_list = []
# 		pred_states_list = []
# 		pred_rewards_list = []
# 		reuse = False
# 		for i in range(config.n_steps):
# 			# pred_reward, pred_frame = self.create_one_step_pred(current_state, self.actions[:,i], config, reuse)
# 			action_1hot = tf.expand_dims(tf.expand_dims(tf.one_hot(self.actions[:,i], config.n_actions), axis=1), axis=1)
# 			action_tile = tf.tile(action_1hot, [1, config.frame_dims[0], config.frame_dims[1],1])
# 			current_inputs = tf.concat([current_state, action_tile], axis=-1)
# 			pred_frame, pred_reward = self.create_one_step_pred(current_inputs, config, reuse)
# 			pred_state = tf.concat([current_state[:, :, :, 1:config.n_stacked], pred_frame], axis=3)
# 			pred_frames_list.append(pred_frame)
# 			pred_states_list.append(pred_state)
# 			pred_rewards_list.append(pred_reward)
# 			current_state = pred_state
# 			reuse = True

# 		self.pred_state = pred_states_list[0]
# 		self.pred_reward = pred_rewards_list[0]

# 		pred_frames = tf.concat(pred_frames_list, axis=3)
# 		pred_rewards = tf.concat(pred_rewards_list, axis=1)

# 		# self.loss = 	tf.losses.mean_squared_error(self.next_states, pred_states) + \
# 		# 				tf.losses.mean_squared_error(self.rewards, pred_rewards)

# 		self.loss = config.loss_weight * tf.losses.mean_squared_error(self.next_frames, pred_frames) + (1 - config.loss_weight) * tf.losses.mean_squared_error(self.rewards, pred_rewards)
# 		# self.loss = tf.Print(self.loss, [self.next_states, self.pred_state], summarize=15)

# 		variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, config.scope)
# 		gradients = tf.gradients(self.loss, variables)
# 		optimizer = tf.train.AdamOptimizer(learning_rate=config.lr)
# 		gradients, _ = tf.clip_by_global_norm(gradients, config.max_grad_norm)
# 		grads_and_vars = zip(gradients, variables)
# 		self.train_op = optimizer.apply_gradients(grads_and_vars)

# 	def create_one_step_pred(self, x, config, reuse):
# 		with tf.variable_scope(config.scope, reuse=reuse):
# 			# extract features
# 			conv1 = layers.conv2d(x, 32, 8, stride=2)
# 			conv2 = layers.conv2d(conv1, 32, 3)
# 			conv3 = layers.conv2d(conv2, 32, 3, activation_fn=None)
# 			conv3 = tf.nn.relu(conv3 + conv1)
# 			conv4 = layers.conv2d(conv3, 32, 3)
# 			conv5 = layers.conv2d(conv2, 32, 3, activation_fn=None)
# 			encoded = tf.nn.relu(conv3 + conv5)

# 			# predict state
# 			pred_frame = layers.conv2d_transpose(encoded, 1, 8, stride=2, activation_fn=None)

# 			# predict reward
# 			rconv1 = layers.conv2d(encoded, 32, 3)
# 			rpool1 = layers.max_pool2d(rconv1, 2)
# 			rconv2 = layers.conv2d(rpool1, 32, 3)
# 			rpool2 = layers.max_pool2d(rconv1, 2, padding="same")
# 			rpool2 = layers.flatten(rpool2)
# 			pred_reward = layers.fully_connected(rpool2, 1, activation_fn = None)

# 			return pred_frame, pred_reward

# 	def predict(self):
# 		pred_state, next_frames = sess.run([self.pred_state, self.next_frames])
# 		return pred_state, next_frames

# 	def train(self):
# 		loss, pred_state, pred_reward, _ = sess.run([self.loss, self.pred_state, self.pred_reward, self.train_op])
# 		return loss, pred_state, pred_reward

class Config():
    n_actions = 6
    state_dims = [84, 84, 4]
    channels = 4
    frame_dims = [84, 84]
    rollout_length = 3
    hidden_dim = 512
    lstm_layers = 1
    reuse = False
    # filename must end in .ckpt. do not attempt to specify an actual file here
    ckpt_file = "/Users/williamhang/Downloads/env_model/model_pretrained_150epochs.ckpt"#"/home/cs234/env_model/model_pretrained_150epochs.ckpt" # this is an example
    frame_mean_path = "/Users/williamhang/Downloads/s_mean.npy"

config = Config()

states = np.load('/Users/williamhang/Downloads/tmp/datasetcurrent_states.npy')
actions = np.load('/Users/williamhang/Downloads/tmp/datasetactions.npy')
next_states = np.load('/Users/williamhang/Downloads/tmp/datasetnext_states.npy')
rewards = np.load('/Users/williamhang/Downloads/tmp/datasetrewards.npy')

stacked = np.load('results/stacked.npy')


frame = np.expand_dims(states[-30], axis=0)
action = actions[-30].astype(np.uint8)

frame_mean = np.load(config.frame_mean_path)

# frame, _, _, _ = env.step([1])

print(frame_mean.shape, frame.shape)

frame = (frame - frame_mean) / 255.

acts = np.zeros((1, 84, 84, 6))

acts[:, :, :, 4] = 1

inputs = np.concatenate([frame, acts], axis=-1)

with sess:
	model = EnvironmentModel(config)
	# saver = tf.train.Saver()
	# saver.restore(sess, config.load_ckpt)
	# print("Model restored.")

	output = sess.run(model.pred_state, feed_dict={model.x: stacked})[-1]
	output = np.expand_dims
	output = output * 255. + frame_mean
	plt.imshow(output.squeeze().astype(np.uint8), cmap='Greys_r')
	plt.show()




