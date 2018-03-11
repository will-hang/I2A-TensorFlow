import numpy as np
import argparse
import gym
import tensorflow.contrib.layers as layers
import tensorflow as tf
import matplotlib.pyplot as plt
import warnings

from baselines.common.vec_env.vec_frame_stack import VecFrameStack
from baselines.common.cmd_util import make_atari_env

from skimage.transform import resize

class EnvironmentModel(object):
    def __init__(self, config):
        self.x = tf.placeholder(tf.float32, [None] + config.frame_dims + [config.n_stacked * 4])
        self.actions = tf.placeholder(tf.int32, [None])
        self.next_states = tf.placeholder(tf.float32, [None] + config.frame_dims + [config.n_stacked])
        # extract features
        with tf.variable_scope(config.scope):
            x_padded = self.x
            conv1 = layers.conv2d(x_padded, 64, 8, stride=2)
            conv2 = layers.conv2d(conv1, 128, 6, stride=2)
            conv3 = layers.conv2d(conv2, 128, 6, stride=2)
            conv4 = layers.conv2d(conv3, 128, 4, stride=2)
            feats = layers.flatten(conv4)
            fc1 = layers.fully_connected(feats, 512)
            fc2 = layers.fully_connected(fc1, 512, activation_fn=None, 
                weights_initializer=tf.random_uniform_initializer(minval=-1.0, maxval=1.0))
            # add the action
            actions_1hot = tf.one_hot(self.actions, config.n_actions)
            action_vector = layers.fully_connected(actions_1hot, 512, activation_fn=None,
                weights_initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1))
            combined = fc2 * action_vector
            # get the predicted next state
            fc3 = layers.fully_connected(combined, 512, activation_fn=None,
                weights_initializer=tf.random_uniform_initializer(minval=-1.0, maxval=1.0))
            fc4 = layers.fully_connected(fc3, 4608)
            fc4 = tf.reshape(fc4, [-1, 6, 6, 128])
            upconv4 = layers.conv2d_transpose(fc4, 128, 4, stride=2)
            upconv3 = layers.conv2d_transpose(upconv4, 128, 6, stride=2)
            upconv2 = layers.conv2d_transpose(upconv3, 128, 6, stride=2)
            upconv1 = layers.conv2d_transpose(upconv2, config.n_stacked, 8, stride=2, activation_fn=None)
            self.pred_state = upconv1

        self.loss = tf.losses.mean_squared_error(self.next_states, self.pred_state)

        variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, config.scope)
        gradients = tf.gradients(self.loss, variables)
        optimizer = tf.train.AdamOptimizer(learning_rate=config.lr)
        gradients, _ = tf.clip_by_global_norm(gradients, config.max_grad_norm)
        grads_and_vars = zip(gradients, variables)
        self.train_op = optimizer.apply_gradients(grads_and_vars)

    def train(self, s, a, s_prime, evaluate=False):
        s = np.stack(s, axis=0)
        a = np.array(a)
        s_prime = np.stack(s_prime, axis=0)
        if evaluate:
            fetch = [self.loss]
        else:
            fetch = [self.loss, self.train_op]
        stuff = sess.run(
            fetch,
            feed_dict={
                self.x: s,
                self.actions: a,
                self.next_states: s_prime
            })
        loss = None
        if evaluate:
            loss = stuff
        else:
            loss = stuff[0]
        return loss

    def predict(self, s, a):
        s = np.stack(s, axis=0)
        a = np.array(a)
        pred_state = sess.run(
            [self.pred_state],
            feed_dict={
                self.x: s,
                self.actions: a
            })[0]
        return pred_state

class Actor():
    def __init__(self, config, sess):
        self.config = config

    def act(self, state):
        action = np.random.choice(list(range(config.n_actions)))
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
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                next_state = np.squeeze(next_state[:, :, :, -1]).reshape((84, 84, 1))
                next_states.append(resize(next_state, (96, 96, 1), preserve_range=True) / 255.)
                actions.append(np.squeeze(action))
                rewards.append(np.squeeze(reward))
                states.append(resize(np.squeeze(state), (96, 96, 4), preserve_range=True) / 255.)

            state = next_state
            if done[-1]:
                state = self.env.reset()

        return states, actions, rewards, next_states

def run(sess, config, env):
    worker = Worker(config, env)
    model = EnvironmentModel(config)
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    losses = []
    for i in range(config.total_updates // config.batch_size):
        states, actions, rewards, next_states = worker.get_batch(config.batch_size)
        loss = model.train(states, actions, next_states)
        pred_state = model.predict(states, actions)

        if i % 50 == 0:
            np.save('../outputs/preds', pred_state)

        losses.append(loss)
        np.save('../outputs/losses', losses)
        print('Batch {}-{}: Loss {}'.format(i * config.batch_size, (i + 1) * config.batch_size, loss))

parser = argparse.ArgumentParser(description='A3C')
parser.add_argument('--env', default = 'PongNoFrameskip-v4',
                    help='environment to test on')
parser.add_argument('--num_env', type=int, default = '1',
                    help='number of envs')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='learning rate')
parser.add_argument('--max_grad_norm', type=float, default=0.1,
                    help='maximum gradient norm for clipping')
parser.add_argument('--reuse', type=bool, default=False,
                    help='policy architecture')
parser.add_argument('--batch_size', type=int, default=64,
                    help='batch size')
parser.add_argument('--normalize_adv', type=bool, default=True,
                    help='normalize advantage')
parser.add_argument('--seed', type=int, default=1234,
                    help='random seed')
parser.add_argument('--loss_weight', type=float, default=0.9,
                    help='environment loss weighting')
parser.add_argument('--total_updates', type=int, default=10000000,
                    help='total number of updates')

if __name__ == '__main__':
    config = parser.parse_args()

    tf.set_random_seed(config.seed)

    env = VecFrameStack(make_atari_env(config.env, config.num_env, config.seed), 4)
    sess = tf.Session()

    config.frame_dims = [96, 96]
    config.n_stacked = 1
    config.n_actions = int(env.action_space.n)
    config.scope = 'worker'
    # config.reuse = tf.AUTO_REUSE

    run(sess, config, env)