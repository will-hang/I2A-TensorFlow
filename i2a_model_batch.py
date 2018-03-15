# pi.x you need to figure this one out
# DELETE policy.get_initial_features from universe-starter-agent

import tensorflow.contrib.layers as layers
from tensorflow import nn
import tensorflow as tf
from tensorflow.contrib.layers import conv2d, fully_connected
import numpy as np
import matplotlib.pyplot as plt
from models import Policy, natureCNN, linear, dynamic_rnn, justCNN

# TEST = True

# sess = tf.Session()
		
########## BRANDON'S SECTION ##########
class ImaginationCore(object):
	def __init__(self, config, policy):
		self.policy = policy
		self.config = config
		self.frame_mean = np.load(config.frame_mean_path)
		# you need to make the EnvironmentModel class
		self.states = tf.placeholder(tf.float32, [None, *config.state_dims])
		with tf.variable_scope('rollout_policy', reuse=True):
			_, _, self.actions, _ = policy.forward(self.states)

	def set_env(self, env_model):
		self.env_model = env_model

	def predict(self, states, init=False):
		# TODO: i give you a batch of states, and you give me the predictions for the next state
		#		if init == True, then each action for each rollout should be fixed, one distinct action per rollout
		#		if init == False, then sample random actions for each rollout
		#		use sess.run to retrieve actions from self.policy
		#		make your own environment model and use it however you want!

		sess = tf.get_default_session()
		config = self.config
		bsz = states.shape[0]
		flat_bsz = bsz * config.n_actions
		# states is [-1, n_actions, 84, 84, 4]
		states = states.reshape((flat_bsz, 84, 84, 4))
		imagined_actions = None
		if init:
			imagined_actions = np.array(bsz * [range(config.n_actions)]).reshape(-1)
		else:
			imagined_actions = 	sess.run(
									self.actions,
									feed_dict={
										self.states: states
									})
		# states is now [-1, 84, 84, 4] and actions is [-1]
		action_tile = np.zeros((flat_bsz, *config.frame_dims, config.n_actions))
		for idx, act in enumerate(imagined_actions.tolist()):
			action_tile[idx, :, :, act] = 1

		states_ = (states - self.frame_mean) / 255.
		stacked = np.concatenate([states_, action_tile], axis=-1)

		imagined_next_states, imagined_rewards = sess.run(
			[self.env_model.pred_state, self.env_model.pred_reward],
			feed_dict={
				self.env_model.x: stacked
			})

		imagined_next_states = imagined_next_states * 255. + self.frame_mean
		
		imagined_next_states = np.concatenate([states[:, :, :, 1:], imagined_next_states], axis=-1)
		imagined_next_states = imagined_next_states.reshape(bsz, config.n_actions, 84, 84, 4)
		imagined_rewards = imagined_rewards.reshape(bsz, config.n_actions)

		return imagined_next_states, imagined_rewards

		# states dims: [n_actions, *state_dims]
		# expected output dims: 
		# 	next_state: [n_actions, *state_dims]
		#	reward: [n_actions]

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

class Encoder(object):
	def __init__(self, stem, config):
		self.stem = stem
		self.config = config

	# states dims: [-1, n_actions, rollout_length, *state_dims]
	# reward dims: [-1, n_actions, rollout_length]
	# expected output dims:
	# 	encodings: [n_actions, hidden_dim]
	def forward(self, x, r):
		# compress this so that we have [bsz, *state_dims]
		x = tf.reshape(x, [-1] + self.config.state_dims)
		feats = self.stem(x)
		# now feats is something like [bsz, 11, 11, 64]
		# we need to generate a reward tensor that is [bsz, 11, 11, 1] and concat this with feats
		# sorry, what you are about to see really sucks
		# r = tf.reshape(r, [-1])
		# r = tf.tile(r, [121])
		# r = tf.reshape(r, [121, -1])
		# r = tf.transpose(r)
		# r = tf.reshape(r, [-1, 11, 11, 1])
		r = tf.expand_dims(tf.expand_dims(r, axis=1), axis=2)
		r = tf.tile(r, [1, 2, 2])
		# stick feats and r together!
		feats = tf.concat([feats, r], -1)
		# feats should be [-1, 11, 11, 65] right now
		feats = layers.flatten(feats)
		# reshape so that we get [bsz * n_actions, rollout_length, 7865]
		# 7865 because we are flattening the tensor
		feats = tf.reshape(feats, [-1, self.config.rollout_length, 7865])
		# return feats
		outputs, state = dynamic_rnn(self.config.lstm_layers, feats, self.config.hidden_dim)
		state = state[-1][-1]
		state = tf.reshape(state, [-1, self.config.n_actions, self.config.hidden_dim])
		return state

class I2A(object):
	def __init__(self, config):
		self.config = config
		# for universe-starter-agent
		self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)
		# build some graph nodes
		self.inputs_s = tf.placeholder(tf.float32, [None] + [config.n_actions, config.rollout_length] + config.state_dims)
		self.inputs_r = tf.placeholder(tf.float32, [None] + [config.n_actions, config.rollout_length])
		self.X = tf.placeholder(tf.float32, [None] + config.state_dims)

		# instantiate the model free policy
		self.mf_policy = justCNN
		with tf.variable_scope('mf_policy', reuse=config.reuse):
			mf_feats = self.mf_policy(self.X)
			mf_feats = layers.flatten(mf_feats)
		# instantiate the rollout policy
		self.rollout_policy = Policy(natureCNN, config)
		with tf.variable_scope('rollout_policy', reuse=config.reuse):
			self.rp_logits, rp_pi, rp_actions, rp_vf = self.rollout_policy.forward(self.X)
		# instantiate the imagination core
		# we can only instantiate this once we have defined our rollout policy
		self.imagination_core = ImaginationCore(config, self.rollout_policy)
		# instantiate the encoder
		self.encoder = Encoder(justCNN, config)
		with tf.variable_scope('encoder', reuse=config.reuse):
			encodings = self.encoder.forward(self.inputs_s, self.inputs_r)
		aggregate = tf.reshape(encodings, shape=[-1, config.n_actions * config.hidden_dim])
		# we can experiment with this next line of code
		# you can either concat, add, or multiply
		i2a_inputs = tf.concat([aggregate, mf_feats], -1)
		# instantiate the I2A policy
		self.i2a_policy = Policy(linear, config)
		with tf.variable_scope('i2a_policy', reuse=config.reuse):
			self.logits, self.pi, self.actions, self.vf = self.i2a_policy.forward(i2a_inputs)

		# TODO:
		#		during training, run actions through the rollout policy and set the loss to negative log likelihood with
		#		the i2a policy in order to keep the KL between rollout nad i2a policy small
		#
		# 		the reason we define a separate rollout policy and pass it into the env model rather than contain
		#		it within the env model is that we also want to directly pass states through the rollout model during
		#		training

	def act(self, state):
		# i2a gets a 1-batch of states
		# [-1, 84, 84, 4]
		sess = tf.get_default_session()
		rollouts_s, rollouts_r = self.rollout(state)
		# we should get something that is
		# rollouts_s: [-1, n_actions, rollout_length, 84, 84, 4]
		# rollouts_r: [-1, n_actions, rollout_length]
		logits, pi, actions, vf = sess.run(
			[self.logits, self.pi, self.actions, self.vf],
			feed_dict={
				self.inputs_s: rollouts_s,
				self.inputs_r: rollouts_r,
				self.X: state
			})
		# (rollouts_s, rollouts_r) is in a tuple because this is literally our state for I2A
		return actions, vf, rollouts_s, rollouts_r

	def value(self, state):
		# i2a gets a 1-batch of states
		# [-1, 84, 84, 4]
		sess = tf.get_default_session()
		rollouts_s, rollouts_r = self.rollout(state)
		# we should get something that is
		# rollouts_s: [-1, n_actions, rollout_length, 84, 84, 4]
		# rollouts_r: [-1, n_actions, rollout_length]
		logits, pi, actions, vf = sess.run(
			[self.logits, self.pi, self.actions, self.vf],
			feed_dict={
				self.inputs_s: rollouts_s,
				self.inputs_r: rollouts_r,
				self.X: state
			})
		# (rollouts_s, rollouts_r) is in a tuple because this is literally our state for I2A
		return vf

	def rollout(self, states):
		# initialize state for batch rollouts
		# this is of shape [-1, n_actions, 84, 84, 4]
		config = self.config
		states = np.expand_dims(states, axis=1)
		states = np.concatenate([states] * config.n_actions, axis=1)
		# roll everything out and put it in a placeholder
		rollouts_s = []
		rollouts_r = []
		for i in range(config.rollout_length):
			# on the first timestep, each rollout will take its own action
			# afterwards, each rollout will take randomly sampled actions
			next_states, rewards = self.imagination_core.predict(states, init=i == 0)
			# add to our rollout
			rollouts_s.append(next_states)
			rollouts_r.append(rewards)
			# advance
			states = next_states
		# rollouts_s should be a list of objects of [-1, n_actions, 84, 84, 4]
		# we want to stack them so the new rollouts_s is [-1, rollout_length, n_actions, 84, 84, 4]
		rollouts_s = np.stack(rollouts_s, axis=1)
		rollouts_r = np.stack(rollouts_r, axis=1).reshape(-1, config.rollout_length, config.n_actions)
		rollouts_s = np.transpose(rollouts_s, (0, 2, 1, 3, 4, 5))
		rollouts_r = np.transpose(rollouts_r, (0, 2, 1))
		return rollouts_s, rollouts_r

# # ###### TEST CODE
# class config():
# 	n_actions = 6
# 	state_dims = [84, 84, 4]
# 	channels = 4
# 	frame_dims = [84, 84]
# 	rollout_length = 3
# 	hidden_dim = 512
# 	lstm_layers = 1

# config = config()

# # ##### VALIDITY TEST #######
# i2a = I2A(config)

# real_rewards = tf.placeholder(tf.float32, [None])

# aux_policy_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=i2a.actions, logits=i2a.rp_logits)
# loss = tf.losses.mean_squared_error(real_rewards, i2a.vf) #+ aux_policy_loss
# opt = tf.train.AdamOptimizer(learning_rate=1e-3)
# train_op = opt.minimize(loss)

# sess.run(tf.global_variables_initializer())
# sess.run(tf.local_variables_initializer())

# state = np.random.random((2, 84, 84, 4))
# with sess:
# 	print(i2a.act(state))

# for i in range(64):
# 	rand1 = 4#np.random.random()
# 	rand2 = 4#np.random.random()
# 	state = np.ones((32, 84, 84, 4)) * rand1
# 	rollouts_s = np.ones((32, 6, 3, 84, 84, 4)) * 2 * rand1
# 	rollouts_r = np.ones((32, 6, 3)) * 3 * rand2
# 	rewards = np.ones((32)) * rand1 * rand2 * 10

# 	_, l = sess.run(
# 		[train_op, loss], 
# 		feed_dict={
# 			i2a.inputs_s: rollouts_s,
# 			i2a.inputs_r: rollouts_r,
# 			i2a.x: state,
# 			real_rewards: rewards
# 		})
# 	print(l)

# rand1 = 4#np.random.random()
# rand2 = 4#np.random.random()
# state = np.ones((32, 84, 84, 4)) * rand1
# rollouts_s = np.ones((32, 6, 3, 84, 84, 4)) * 2 * rand1
# rollouts_r = np.ones((32, 6, 3)) * 3 * rand2
# rewards = np.ones((32)) * rand1 * rand2 * 10

# vf, l = sess.run(
# 	[i2a.vf, loss],
# 	feed_dict={
# 		i2a.inputs_s: rollouts_s,
# 		i2a.inputs_r: rollouts_r,
# 		i2a.x: state,
# 		real_rewards: rewards
# 	})

# print(vf, l)

######## ENCODER TEMPORAL AND ACTION SPACE DEPENDENCY TEST######
# inputs = tf.placeholder(tf.float32, [None] + [config.n_actions, config.rollout_length] + config.state_dims)
# labels = tf.placeholder(tf.float32, [None, config.n_actions, 64])

# encoder = Encoder(natureCNN, config, 'haha')
# with tf.variable_scope('haha'):
# 	pred = encoder.forward(inputs)

# loss = tf.losses.mean_squared_error(labels, pred)
# opt = tf.train.AdamOptimizer(learning_rate=2e-3)
# train_op = opt.minimize(loss)

# sess.run(tf.global_variables_initializer())
# sess.run(tf.local_variables_initializer())

# for k in range(32):
# 	x = np.ones((32, 6, 15, 84, 84, 4))
# 	# truth = np.ones((192, 15, 512)) * 3
# 	truth = np.ones((32, config.n_actions, config.hidden_dim)) * 5
# 	# for i in range(15):
# 	# # for i in range(6):
# 	# 	truth[:, i, :] = i
# 	l, _, p = sess.run(
# 		[loss, train_op, pred],
# 		feed_dict={
# 			inputs: x,
# 			labels: truth
# 		})
# 	print(np.mean(p, axis=(0, 2)).tolist())
# 	print(l)

# x = np.ones((1, 6, 15, 84, 84, 4))
# # for i in range(15):
# for i in range(6):
# 	# x[:, :, :, :, :, :] = i * 3
# 	x[:, i, :, :, :, :] = i * 0.1
# p = sess.run(
# 	[pred],
# 	feed_dict={
# 		inputs: x
# 	})[0]

# print(np.mean(p, axis=(0, 2)).tolist())

# x = np.ones((1, 6, 15, 84, 84, 4))
# # for i in range(15):
# for i in range(15):
# 	x[:, :, i, :, :, :] = i * 0.1
# 	# x[:, i, :, :, :, :] = i
# p = sess.run(
# 	[pred],
# 	feed_dict={
# 		inputs: x
# 	})[0]

# print(np.mean(p, axis=(0, 2)).tolist())