import numpy as np
import argparse
import tensorflow as tf
import tensorflow.contrib.layers as layers

'''
def normalizedColumnsInitializer(std=1.0):
	def _initializer(shape, dtype=None, partition_info=None):
		out = np.random.randn(*shape).astype(np.float32)
		out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
		return tf.constant(out)
	return _initializer

def show_images(images, cols = 1, titles = None):
	"""Display a list of images in a single figure with matplotlib.

	Parameters
	---------
	images: List of np.arrays compatible with plt.imshow.

	cols (Default = 1): Number of columns in figure (number of rows is
						set to np.ceil(n_images/float(cols))).

	titles: List of titles corresponding to each image. Must have
			the same length as titles.
	"""
	assert((titles is None) or (len(images) == len(titles)))
	n_images = len(images)
	if titles is None: titles = ['Image (%d)' % i for i in range(1,n_images + 1)]
	fig = plt.figure()
	for n, (image, title) in enumerate(zip(images, titles)):
		a = fig.add_subplot(cols, np.ceil(n_images/float(cols)), n + 1)
		if image.ndim == 2:
			plt.gray()
		plt.imshow(image)
		a.set_title(title)
	fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
	plt.show()
'''

class EnvironmentModel(object):
	def __init__(self, config):
		self.states_placeholder = tf.placeholder(tf.float32, [None] + config.frame_dims + [config.n_stacked])
		self.actions_placeholder = tf.placeholder(tf.int32, [None, config.n_steps])
		self.nf_placeholder = tf.placeholder(tf.float32, [None] + config.frame_dims + [config.n_steps])
		self.rewards_placeholder = tf.placeholder(tf.float32, [None, config.n_steps])

		dataset = tf.data.Dataset.from_tensor_slices({
			"states": self.states_placeholder,
			"actions": self.actions_placeholder,
			"rewards": self.rewards_placeholder,
			"next_frames": self.nf_placeholder
		})
		dataset = dataset.shuffle(config.seed).repeat(config.num_epochs).batch(config.batch_size)
		self.iterator = dataset.make_initializable_iterator()
		it_dict = self.iterator.get_next()

		self.states = it_dict["states"]
		self.actions = it_dict["actions"]
		self.rewards = it_dict["rewards"]
		self.next_frames = it_dict["next_frames"]

		current_state = self.states
		pred_frames_list = []
		pred_states_list = []
		pred_rewards_list = []
		reuse = False
		for i in range(config.n_steps):
			# pred_reward, pred_frame = self.create_one_step_pred(current_state, self.actions[:,i], config, reuse)
			action_1hot = tf.expand_dims(tf.expand_dims(tf.one_hot(self.actions[:,i], config.n_actions), axis=1), axis=1)
			action_tile = tf.tile(action_1hot, [1, config.frame_dims[0], config.frame_dims[1],1])
			current_inputs = tf.concat([current_state, action_tile], axis=-1)
			pred_frame, pred_reward = self.create_one_step_pred(current_inputs, config, reuse)
			pred_state = tf.concat([current_state[:, :, :, 1:config.n_stacked], pred_frame], axis=3)
			pred_frames_list.append(pred_frame)
			pred_states_list.append(pred_state)
			pred_rewards_list.append(pred_reward)
			current_state = pred_state
			reuse = True

		self.pred_state = pred_states_list[0]
		self.pred_reward = pred_rewards_list[0]

		pred_frames = tf.concat(pred_frames_list, axis=3)
		pred_rewards = tf.concat(pred_rewards_list, axis=1)

		# self.loss = 	tf.losses.mean_squared_error(self.next_states, pred_states) + \
		# 				tf.losses.mean_squared_error(self.rewards, pred_rewards)

		self.loss = config.loss_weight * tf.losses.mean_squared_error(self.next_frames, pred_frames) + (1 - config.loss_weight) * tf.losses.mean_squared_error(self.rewards, pred_rewards)
		# self.loss = tf.Print(self.loss, [self.next_states, self.pred_state], summarize=15)

		variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, config.scope)
		gradients = tf.gradients(self.loss, variables)
		optimizer = tf.train.AdamOptimizer(learning_rate=config.lr)
		gradients, _ = tf.clip_by_global_norm(gradients, config.max_grad_norm)
		grads_and_vars = zip(gradients, variables)
		self.train_op = optimizer.apply_gradients(grads_and_vars)

	def create_one_step_pred(self, x, config, reuse):
		with tf.variable_scope(config.scope, reuse=reuse):
			# extract features
			conv1 = layers.conv2d(x, 32, 8, stride=2)
			conv2 = layers.conv2d(conv1, 32, 3)
			conv3 = layers.conv2d(conv2, 32, 3, activation_fn=None)
			conv3 = tf.nn.relu(conv3 + conv1)
			conv4 = layers.conv2d(conv3, 32, 3)
			conv5 = layers.conv2d(conv2, 32, 3, activation_fn=None)
			encoded = tf.nn.relu(conv3 + conv5)

			# predict state
			pred_frame = layers.conv2d_transpose(encoded, 1, 8, stride=2, activation_fn=None)

			# predict reward
			rconv1 = layers.conv2d(encoded, 32, 3)
			rpool1 = layers.max_pool2d(rconv1, 2)
			rconv2 = layers.conv2d(rpool1, 32, 3)
			rpool2 = layers.max_pool2d(rconv1, 2, padding="same")
			rpool2 = layers.flatten(rpool2)
			pred_reward = layers.fully_connected(rpool2, 1, activation_fn = None)

			return pred_frame, pred_reward

	def predict(self):
		pred_state, next_frames = sess.run([self.pred_state, self.next_frames])
		return pred_state, next_frames

	def train(self):
		loss, pred_state, pred_reward, _ = sess.run([self.loss, self.pred_state, self.pred_reward, self.train_op])
		return loss, pred_state, pred_reward

'''
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

	def get_batch(self, batch_size, n_steps):
		states = []
		rewards = []
		actions = []
		next_states = []

		state = self.env.reset()
		for i in range(batch_size + n_steps - 1):
			action = [self.actor.act(state)]
			next_state, reward, done, info = self.env.step(action)

			actions.append(np.squeeze(action))
			rewards.append(np.squeeze(reward))
			states.append(np.squeeze(state))
			next_states.append(np.squeeze(next_state))

			state = next_state
			if done[-1]:
				state = self.env.reset()

		nstep_states = np.stack(states[:batch_size], axis=0).astype(np.float32)
		nstep_rewards = np.stack([rewards[i: i+n_steps] for i in range(batch_size)], axis=0).astype(np.float32)
		nstep_actions = np.stack([actions[i: i+n_steps] for i in range(batch_size)], axis=0).astype(np.int32)
		nstep_next_states = np.stack([np.concatenate(next_states[i: i+n_steps], axis=2) for i in range(batch_size)], axis=0).astype(np.float32)

		return nstep_states, nstep_actions, nstep_rewards, nstep_next_states
'''

def run(sess, config):
	model = EnvironmentModel(config)
	saver = tf.train.Saver()
	if config.load_ckpt:
		saver.restore(sess, config.load_ckpt)
		print("Model restored.")
	else:
		sess.run(tf.global_variables_initializer())
		sess.run(tf.local_variables_initializer())

	# load dataset
	states = np.load("/home/brandon1/mspacman_env_data/states.npy")
	actions = np.load("/home/brandon1/mspacman_env_data/actions.npy")
	rewards = np.load("/home/brandon1/mspacman_env_data/rewards.npy")
	next_states = np.load("/home/brandon1/mspacman_env_data/next_states.npy")

	train_states = states[:-config.batch_size, :, :, :]
	train_actions = actions[:-config.batch_size, :]
	train_rewards = rewards[:-config.batch_size, :]
	train_next_states = next_states[:-config.batch_size, :, :, :]

	s_mean = np.mean(train_states, axis=(0,3), keepdims=True)
	train_states = (train_states - s_mean)/255.0
	train_next_states = (train_next_states - s_mean)/255.0
	train_next_frames = np.expand_dims(train_next_states[:, :, :, config.n_stacked-1], axis=3)

	test_states = states[-config.batch_size:, :, :, :]
	test_actions = actions[-config.batch_size:, :]
	test_rewards = rewards[-config.batch_size:, :]
	test_next_states = next_states[-config.batch_size:, :, :, :]

	test_states = (test_states - s_mean)/255.0
	test_next_states = (test_next_states - s_mean)/255.0
	test_next_frames = np.expand_dims(test_next_states[:, :, :, config.n_stacked-1], axis=3)

	# train loop
	sess.run(model.iterator.initializer, feed_dict={
		model.states_placeholder: train_states,
		model.actions_placeholder: train_actions,
		model.rewards_placeholder: train_rewards,
		model.nf_placeholder: train_next_frames
	})

	losses = []
	train_steps = 0
	while True:
		train_steps += 1
		try:
			loss, pred_state, pred_reward = model.train()
			losses.append(loss)
			print('Train Step {}: Loss {}'.format(train_steps, loss))
		except tf.errors.OutOfRangeError:
			break

	# test loop
	sess.run(model.iterator.initializer, feed_dict={
		model.states_placeholder: test_states,
		model.actions_placeholder: test_actions,
		model.rewards_placeholder: test_rewards,
		model.nf_placeholder: test_next_frames
	})

	pred_state, next_frames = model.predict()

	save_path = saver.save(sess, config.save_ckpt)
	print("Model saved in path: %s" % save_path)
	np.save('../outputs/next_frames', next_frames * 255.0 + s_mean)
	np.save('../outputs/preds', pred_state * 255.0 + s_mean)
	np.save('../outputs/losses', losses)

parser = argparse.ArgumentParser(description='A3C')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='learning rate')
parser.add_argument('--max_grad_norm', type=float, default=0.5,
                    help='maximum gradient norm for clipping')
parser.add_argument('--batch_size', type=int, default=128,
                    help='batch size')
parser.add_argument('--num_epochs', type=int, default=10,
                    help='number of training epochs')
parser.add_argument('--seed', type=int, default=123,
                	help='random seed')
parser.add_argument('--loss_weight', type=float, default=0.9,
                	help='environment loss weighting')
parser.add_argument('--n_steps', type=int, default=1,
					help='use n-step loss for training')
parser.add_argument('--load_ckpt', type=str, default=None,
					help='path to checkpoint to load')
parser.add_argument('--save_ckpt', type=str, default="../ckpts/model.ckpt",
					help='path to save checkpoint')

if __name__ == '__main__':
	config = parser.parse_args()

	tf.set_random_seed(config.seed)
	sess = tf.Session()

	config.frame_dims = [84, 84]
	config.n_stacked = 4
	config.n_actions = 9
	config.scope = 'worker'

	run(sess, config)
