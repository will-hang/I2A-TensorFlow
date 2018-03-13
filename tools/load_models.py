import tensorflow as tf
import sys, time
import numpy as np
from policies import CnnPolicy
from baselines.common.vec_env.vec_frame_stack import VecFrameStack
from baselines.common.cmd_util import make_atari_env

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
sess = tf.Session()

class Actor():
	def __init__(self, ob_space, ac_space, n_batch, n_steps):
		with tf.variable_scope("global"):
			self.network = CnnPolicy(sess, ob_space, ac_space, n_batch, n_steps)
			saver = tf.train.Saver()
			saver.restore(sess, "./checkpoints/model.ckpt")

	def act(self, state):
		stuff = self.network.step(state)
		action, value_, _, _= stuff[0], stuff[1], stuff[2:]
		return action, value_

env = VecFrameStack(make_atari_env('PongNoFrameskip-v4', 1, 123), 4)
ob_space = env.observation_space
ac_space = env.action_space
actor = Actor(ob_space, ac_space, 32, 1)
with sess:
	print(actor.act(np.ones((32, 84, 84, 4))))