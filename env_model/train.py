class EnvironmentModel(object):
	def __init__(self, config):
		self.x = tf.placeholder(tf.float32, [None] + config.frame_dims + [stacked, config.n_actions])
		# extract features
		with tf.variable_scope(config.scope):
			conv1 = layers.conv2d(self.x, 64, 8, stride=2)
			conv2 = layers.conv2d(conv1, 128, 6, stride=2)
			conv3 = layers.conv2d(conv2, 128, 6, stride=2)
			conv4 = layers.conv2d(conv3, 128, 4, stride=2)
			feats = layers.flatten(conv4)
			fc1 = layers.fully_connected(feats, 2048)
			fc2 = layers.fully_connected(fc1, 2048)
			# get the predicted reward
			self.pred_reward = layers.fully_connected(fc2, 1, activation_fn=None)
			# get the predicted next state
			fc3 = layers.fully_connected(fc2, 2048)
			fc4 = layers.fully_connected(fc3, 11264)
			fc4 = tf.reshape(fc4, [128, 11, 8])
			upconv_4 = layers.conv2d_transpose(fc4, 128, 4, stride=2)
			upconv_3 = layers.conv2d_transpose(upconv_4, 128, 6, stride=2)
			upconv_2 = layers.conv2d_transpose(upconv_3, 128, 6, stride=2)
			self.pred_state = layers.conv2d_transpose(upconv_2, 4, 8, stride=2, activation_fn=None)

	def predict(self, state, action):
		action_tile = np.zeros((*config.frame_dims, config.n_actions))
		action_tile[:, :, action] = 1
		stacked = np.stack([state, action_tile], axis=-1)
		pred_state, pred_reward = sess.run(
			[self.pred_state, self.pred_reward],
			feed_dict={
				self.x: stacked
			})
		return pred_state, pred_reward

class Actor():
	def __init__(self, config, sess):
		self.config = config
		self.sess = sess
		saver = tf.train.import_meta_graph(config.meta_graph)
		saver.restore(sess,tf.train.latest_checkpoint('./'))
		graph = tf.get_default_graph()
		self.action = graph.get_tensor_by_name('action')

	def act(self, state):
		self.inputs = graph.get_tensor_by_name('inputs:0')
		action = self.sess.run(
			[self.action],
			feed_dict={
				self.inputs: state
			})
		return action

class Worker():
	def __init__(self, config):
		self.actor = Actor(config, sess)

	def get_batch(self, batch_size):
		states = []
		rewards = []
		actions = []
		next_states = []

		state = self.env.reset()
		for i in range(batch_size):
			action = self.actor.act(state)
			next_state, reward, done, info = self.env.step(action)

			self.next_states.append(np.squeeze(next_state))
			self.actions.append(np.squeeze(action))
			self.rewards.append(np.squeeze(rewards))
			self.states.append(np.squeeze(state))

			state = next_state

			if done:
				state = self.env.reset()

		return states, actions, rewards, next_states

def train(s, a, r, s_prime, model):
	s = np.stack(s, axis=0)
	a = np.array(a)
	r = np.array(r)
	s_prime = np.stack(s_prime, axis=0)

	self.loss = 

	sess.run([], feed_dict={})

def run(config):
	worker = Worker(config)
	model = EnvironmentModel(config)
	for i in range(config.total_updates // config.batch_size):
		states, actions, rewards, next_states = worker.get_batch(config.batch_size)
		train(states, actions, rewards, next_states, model)







		
