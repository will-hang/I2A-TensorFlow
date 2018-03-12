import tensorflow as tf
import sys, time
import numpy as np
from model import LSTMPolicy

# role = sys.argv[1]
# cluster = tf.train.ClusterSpec({"ps": ["localhost:8000"], "worker": ["localhost:8001"]})
# server = tf.train.Server(cluster, job_name=role, task_index=0)
# sess = tf.Session(target=server.target)

# if role == 'ps':
# 	while True:
# 		time.sleep(1000)
# #First let's load meta graph and restore weights
# saver = tf.train.import_meta_graph('/tmp/pong/train/model.ckpt-1006649.meta')
# saver.restore(sess, tf.train.latest_checkpoint('/tmp/pong/train/'))

# graph = tf.get_default_graph()

# policy = graph.get_tensor_by_name("w1:0")

# variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')

# print(variables)


# with tf.variable_scope("global"):
# 	network = LSTMPolicy([42, 42, 1], 6)

# saver = tf.train.Saver()
# sess = tf.Session()

# saver.restore(sess, "/tmp/pong/train/model.ckpt-1894351")

tf.reset_default_graph()

class Actor():
	def __init__(self, config):
		self.config = config
		with tf.variable_scope("global"):
			self.network = LSTMPolicy([42, 42, 1], 6)
			saver = tf.train.Saver()
			self.sess = tf.Session()
			saver.restore(self.sess, "/tmp/pong/train/model.ckpt-2230477")
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

actor = Actor(4)
print(actor.act(np.ones((42, 42, 1))))