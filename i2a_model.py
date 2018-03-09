# pi.x you need to figure this one out
# DELETE policy.get_initial_features from universe-starter-agent

import tensorflow.contrib.layers as layers
from tensorflow import nn
import tensorflow as tf
from tensorflow.contrib.layers import conv2d, fully_connected
import numpy as np

from models import Policy, natureCNN, linear, dynamic_rnn

TEST = True

sess = tf.Session()
		
########## BRANDON'S SECTION ##########
class ImaginationCore(object):
	def __init__(self, config, policy):
		self.policy = policy
		# you need to make the EnvironmentModel class
		self.env_model = EnvironmentModel(config)

		self.states = tf.placeholder(tf.float32, [config.n_actions, *config.state_dims])
		with tf.variable_scope('rollout_policy', reuse=True):
			_, _, self.actions, _ = policy.forward(self.states)

	def predict(self, states, init=False):
		# TODO: i give you a batch of states, and you give me the predictions for the next state
		#		if init == True, then each action for each rollout should be fixed, one distinct action per rollout
		#		if init == False, then sample random actions for each rollout
		#		use sess.run to retrieve actions from self.policy
		#		make your own environment model and use it however you want!
		if TEST:
			return np.zeros(states.shape), np.zeros(states.shape[0])

		# imagined_actions = None
		# if init:
		# 	imagined_actions = np.array(range(config.n_actions))
		# else:
		# 	imagined_actions = sess.run(
		# 		[self.actions]
		# 		feed_dict={
		# 			self.states: states
		# 		})

		# action_tile = np.zeros((config.n_actions, *config.state_dims, config.n_actions))
		# for idx, act in enumerate(imagined_actions.tolist()):
		# 	action_tile[idx, :, :, act] = 1
		# stacked = np.concatenate([states, action_tile], axis=-1)

		# imagined_next_states, imagined_rewards = sess.run(
		# 	[self.env_model.pred_state, self.env_model.pred_reward],
		# 	feed_dict={
		# 		self.env_model.x: stacked
		# 	})

		# return imagined_next_states, imagined_rewards

		# states dims: [n_actions, *state_dims]
		# expected output dims: 
		# 	next_state: [n_actions, *state_dims]
		#	reward: [n_actions]

class EnvironmentModel(object):
	def __init__(self, config):
		pass

########## END BRANDON'S SECTION ##########

########## KEVIN'S SECTION ##########

class Encoder(object):
	def __init__(self, stem, config, scope):
		self.stem = stem
		self.config = config
		self.scope = scope

	# TODO: i give you a list of states and rewards, and you do an encoding in the reverse order
	#		since this is part of the computation graph, you need to define the inputs to this class as tf.placeholder instead of
	#			function parameters. you also can't return anything, you need to store the result in the object

	# states dims: [-1, n_actions, rollout_length, *state_dims]
	# reward dims: [-1, n_actions, rollout_length]
	# expected output dims:
	# 	encodings: [n_actions, hidden_dim]
	def forward(self, x):
		# compress this so that we have [bsz, *state_dims]
		x = tf.reshape(x, [-1] + config.state_dims)
		feats = self.stem(x)
		# reshape so that we get [bsz * n_actions, rollout_length, 512]
		print(feats)
		feats = tf.reshape(feats, [-1, self.config.rollout_length, 512])
		# return feats
		outputs, state = dynamic_rnn(self.config.lstm_layers, feats, self.config.hidden_dim)
		state = state[0][-1]
		state = tf.reshape(state, [-1, self.config.n_actions, self.config.hidden_dim])
		return state
		##feats = tf.reshape(x, [-1, config.n_actions, config.rollout_length])

########## END KEVIN'S SECTION ##########
class I2A(object):
	def __init__(self, config):
		self.config = config
		self.encoder = Encoder(natureCNN, config, 'encoder')
		self.mf_policy = Policy(natureCNN, config, 'mf_policy')
		# TODO: write the linear policy function class
		self.i2a_policy = Policy(linear, config, 'i2a_policy')
		self.rollout_policy = Policy(natureCNN, config, 'rollout_policy')
		# for universe-starter-agent
		self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)
		# build some graph nodes
		self.inputs_s = tf.placeholder(tf.float32, [None] + [config.n_actions, config.rollout_length] + config.state_dims)
		self.inputs_r = tf.placeholder(tf.float32, [None] + [config.n_actions, config.rollout_length])
		self.x = tf.placeholder(tf.float32, [None] + config.state_dims)

		with tf.variable_scope('rollout_policy', reuse=False):
			self.rp_logits, rp_pi, rp_actions, rp_vf = self.rollout_policy.forward(self.x)

		self.imagination_core = ImaginationCore(config, self.rollout_policy)

		with tf.variable_scope('encoder', reuse=False):
			encodings = self.encoder.forward(self.inputs_s)
		with tf.variable_scope('mf_policy', reuse=False):
			mf_logits, _, _, _ = self.mf_policy.forward(self.x)

		aggregate = tf.reshape(encodings, shape=[-1, config.n_actions * config.hidden_dim])
		# we can experiment with this next line of code
		# you can either concat, add, or multiply
		i2a_inputs = tf.concat([aggregate, mf_logits], -1)
		
		with tf.variable_scope('i2a_policy', reuse=False):
			self.logits, self.pi, self.actions, self.vf = self.i2a_policy.forward(i2a_inputs)

		self.var_list = []

		# you need to do these things:
		# 	make sure that the rollout policy is defined in i2a and passed into imagination core
		#		during training, run actions through the rollout policy and set the loss to negative log likelihood with
		#		the i2a policy

	def act(self, state):
		# i2a gets a 1-batch of states
		# [1, 84, 84, 4]
		rollouts_s, rollouts_r = self.rollout(state)

		logits, pi, actions, vf = sess.run(
			[self.logits, self.pi, self.actions, self.vf],
			feed_dict={
				self.inputs_s: rollouts_s,
				self.inputs_r: rollouts_r,
				self.x: state
			})
		# (rollouts_s, rollouts_r) is in a tuple because this is literally our state for I2A
		return actions, vf, (state, rollouts_s, rollouts_r)

	def rollout(self, state):
		# initialize state for batch rollouts
		# this is of shape [n_actions, 84, 84, 4]
		states = np.stack([state] * self.config.n_actions, axis=0)
		# roll everything out and put it in a placeholder
		rollouts_s = []
		rollouts_r = []
		for i in range(self.config.rollout_length):
			# on the first timestep, each rollout will take its own action
			# afterwards, each rollout will take randomly sampled actions
			next_states, rewards = self.imagination_core.predict(states, init=i == 0)
			# add to our rollout
			rollouts_s.append(next_states)
			rollouts_r.append(rewards)
			# advance
			states = next_states
		# you now have something of [rollout_length, n_actions, 84, 84, 4]
		rollouts_s = np.transpose(rollouts_s, (2, 1, 0, 3, 4, 5))
		rollouts_r = np.transpose(np.expand_dims(rollouts_r, 0), (0, 2, 1))
		print(rollouts_s.shape)
		return rollouts_s, rollouts_r

# ###### TEST CODE
class config():
	n_actions = 6
	state_dims = [84, 84, 4]
	rollout_length = 15
	hidden_dim = 64
	lstm_layers = 3

config = config()

##### VALIDITY TEST #######
# i2a = I2A(config)

# real_rewards = tf.placeholder(tf.float32, [None])

# aux_policy_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=i2a.actions, logits=i2a.rp_logits)
# loss = tf.losses.mean_squared_error(real_rewards, i2a.vf) #+ aux_policy_loss
# opt = tf.train.AdamOptimizer(learning_rate=1e-3)
# train_op = opt.minimize(loss)

# sess.run(tf.global_variables_initializer())
# sess.run(tf.local_variables_initializer())

# for i in range(64):
# 	rand1 = 4#np.random.random()
# 	rand2 = 4#np.random.random()
# 	state = np.ones((32, 84, 84, 4)) * rand1
# 	rollouts_s = np.ones((32, 6, 15, 84, 84, 4)) * 2 * rand1
# 	rollouts_r = np.ones((32, 6, 15)) * 3 * rand2
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
# rollouts_s = np.ones((32, 6, 15, 84, 84, 4)) * 2 * rand1
# rollouts_r = np.ones((32, 6, 15)) * 3 * rand2
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
inputs = tf.placeholder(tf.float32, [None] + [config.n_actions, config.rollout_length] + config.state_dims)
labels = tf.placeholder(tf.float32, [None, config.n_actions, 64])

encoder = Encoder(natureCNN, config, 'haha')
with tf.variable_scope('haha'):
	pred = encoder.forward(inputs)

loss = tf.losses.mean_squared_error(labels, pred)
opt = tf.train.AdamOptimizer(learning_rate=2e-3)
train_op = opt.minimize(loss)

sess.run(tf.global_variables_initializer())
sess.run(tf.local_variables_initializer())

for k in range(32):
	x = np.ones((32, 6, 15, 84, 84, 4))
	# truth = np.ones((192, 15, 512)) * 3
	truth = np.ones((32, config.n_actions, config.hidden_dim)) * 5
	# for i in range(15):
	# # for i in range(6):
	# 	truth[:, i, :] = i
	l, _, p = sess.run(
		[loss, train_op, pred],
		feed_dict={
			inputs: x,
			labels: truth
		})
	print(np.mean(p, axis=(0, 2)).tolist())
	print(l)

x = np.ones((1, 6, 15, 84, 84, 4))
# for i in range(15):
for i in range(6):
	# x[:, :, :, :, :, :] = i * 3
	x[:, i, :, :, :, :] = i * 0.1
p = sess.run(
	[pred],
	feed_dict={
		inputs: x
	})[0]

print(np.mean(p, axis=(0, 2)).tolist())

x = np.ones((1, 6, 15, 84, 84, 4))
# for i in range(15):
for i in range(15):
	x[:, :, i, :, :, :] = i * 0.1
	# x[:, i, :, :, :, :] = i
p = sess.run(
	[pred],
	feed_dict={
		inputs: x
	})[0]

print(np.mean(p, axis=(0, 2)).tolist())



