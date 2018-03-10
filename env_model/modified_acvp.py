#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

import numpy as np
import pickle
import torchvision
from skimage import io
from collections import deque
import gym
from utils import *
from tqdm import tqdm
from network import *

import tensorflow.contrib.layers as layers
import tensorflow as tf

from skimage.transform import resize

PREFIX = '.'
sess = tf.Session()
# PREFIX = '/local/data'

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

class EnvironmentModel(object):
    def __init__(self, config):
        self.x = tf.placeholder(tf.float32, [None] + config.frame_dims + [config.n_stacked * 4])
        self.actions = tf.placeholder(tf.int32, [None])
        self.next_states = tf.placeholder(tf.float32, [None] + config.frame_dims + [config.n_stacked])
        # extract features
        with tf.variable_scope(config.scope):
            x_padded = tf.pad(self.x, tf.constant([[0, 0], [6, 6], [6, 6], [0, 0]]))
            conv1 = layers.conv2d(x_padded, 64, 8, stride=2)
            conv2 = layers.conv2d(conv1, 128, 6, stride=2)
            conv3 = layers.conv2d(conv2, 128, 4, stride=2)
            conv4 = layers.conv2d(conv3, 128, 4, stride=2)
            feats = layers.flatten(conv4)
            fc1 = layers.fully_connected(feats, 2048)
            fc2 = layers.fully_connected(fc1, 2048, activation_fn=None, 
                weights_initializer=tf.random_uniform_initializer(minval=-1.0, maxval=1.0))
            # add the action
            actions_1hot = tf.one_hot(self.actions, config.n_actions)
            action_vector = layers.fully_connected(actions_1hot, 2048,
                weights_initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1))
            combined = fc2 * action_vector
            # get the predicted next state
            fc3 = layers.fully_connected(combined, 2048, activation_fn=None)
            fc4 = layers.fully_connected(fc3, 4608)
            fc4 = tf.reshape(fc4, [-1, 6, 6, 128])
            upconv4 = layers.conv2d_transpose(fc4, 128, 4, stride=2)
            upconv3 = layers.conv2d_transpose(upconv4, 128, 4, stride=2)
            upconv2 = layers.conv2d_transpose(upconv3, 128, 6, stride=2)
            upconv1 = layers.conv2d_transpose(upconv2, config.n_stacked, 8, stride=2, activation_fn=None)
            self.pred_state = upconv1[:, 6:90, 6:90, :]

        self.loss =     tf.losses.mean_squared_error(self.next_states, self.pred_state)

        variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, config.scope)
        gradients = tf.gradients(self.loss, variables)
        optimizer = tf.train.AdamOptimizer(learning_rate=config.lr)
        gradients, _ = tf.clip_by_global_norm(gradients, config.max_grad_norm)
        grads_and_vars = zip(gradients, variables)
        self.train_op = optimizer.apply_gradients(grads_and_vars)

    def train(self, s, a, s_prime, evaluate=False):
        s = np.stack(s, axis=0).transpose(0, 2, 3, 1)
        a = np.array(a)
        s_prime = np.stack(s_prime, axis=0).transpose(0, 2, 3, 1)
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
        s = np.stack(s, axis=0).transpose(0, 2, 3, 1)
        a = np.array(a)
        pred_state = sess.run(
            [self.pred_state],
            feed_dict={
                self.x: s,
                self.actions: a
            })[0]
        return pred_state.transpose(0, 3, 1, 2)

def load_episode(game, ep, num_actions):
    path = '%s/dataset/%s/%05d' % (PREFIX, game, ep)
    with open('%s/action.bin' % (path), 'rb') as f:
        actions = pickle.load(f)
    num_frames = len(actions) + 1
    frames = []

    for i in range(1, num_frames):
        frame = io.imread('%s/%05d.png' % (path, i))
        frame = np.transpose(frame, (2, 0, 1))
        frames.append(frame.astype(np.uint8))

    actions = actions[1:]
    encoded_actions = np.zeros((len(actions), num_actions))
    encoded_actions[np.arange(len(actions)), actions] = 1

    return frames, encoded_actions

def extend_frames(frames, actions):
    buffer = deque(maxlen=4)
    extended_frames = []
    targets = []

    for i in range(len(frames) - 1):
        buffer.append(frames[i])
        if len(buffer) >= 4:
            extended_frames.append(np.vstack(buffer))
            targets.append(frames[i + 1])
    actions = actions[3:, :]

    return np.stack(extended_frames), actions, np.stack(targets)

class Config():
    frame_dims = [84, 84]
    n_stacked = 1
    scope = 'model'
    n_actions = 6
    lr = 1e-4
    max_grad_norm = 0.1

def train(game):
    env = gym.make(game)
    num_actions = env.action_space.n

    config = Config()
    net = EnvironmentModel(config)

    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    with open('%s/dataset/%s/meta.bin' % (PREFIX, game), 'rb') as f:
        meta = pickle.load(f)
    episodes = meta['episodes']
    mean_obs = meta['mean_obs']
    print(mean_obs.shape)
    mean_obs = resize(mean_obs, (3, 84, 84))

    def pre_process(x):
        bsz, dep, _, _ = x.shape
        x = np.array([resize(x[i, :, :, :], (dep, 84, 84)) for i in range(bsz)])
        if x.shape[1] == 12:
            return (x - np.vstack([mean_obs] * 4)) / 255.0
        elif x.shape[1] == 3:
            return (x - mean_obs) / 255.0
        else:
            assert False

    def post_process(y):
        return (y * 255 + mean_obs).astype(np.uint8)

    train_episodes = int(episodes * 0.95)
    indices_train = np.arange(train_episodes)
    iteration = 0
    while True:
        np.random.shuffle(indices_train)
        for ep in indices_train:
            frames, actions = load_episode(game, ep, num_actions)
            frames, actions, targets = extend_frames(frames, actions)
            batcher = Batcher(32, [frames, actions, targets])
            batcher.shuffle()
            while not batcher.end():
                if iteration % 2000 == 0:
                    mkdir('data_tf/acvp-sample')
                    losses = []
                    test_indices = range(train_episodes, episodes)
                    ep_to_print = np.random.choice(test_indices)
                    for test_ep in tqdm(test_indices):
                        frames, actions = load_episode(game, test_ep, num_actions)
                        frames, actions, targets = extend_frames(frames, actions)
                        test_batcher = Batcher(32, [frames, actions, targets])
                        while not test_batcher.end():
                            x, a, y = test_batcher.next_batch()
                            losses.append(net.train(pre_process(x), np.argmax(a, axis=1), pre_process(y), evaluate=True))
                        if test_ep == ep_to_print:
                            test_batcher.reset()
                            x, a, y = test_batcher.next_batch()
                            y_ = post_process(net.predict(pre_process(x), np.argmax(a, axis=1)))
                            torchvision.utils.save_image(torch.from_numpy(y_), 'data_tf/acvp-sample/%s-%09d.png' % (game, iteration))
                            torchvision.utils.save_image(torch.from_numpy(y), 'data_tf/acvp-sample/%s-%09d-truth.png' % (game, iteration))

                    logger.info('Iteration %d, test loss %f' % (iteration, np.mean(losses)))

                x, a, y = batcher.next_batch()
                loss = net.train(pre_process(x), np.argmax(a, axis=1), pre_process(y))
                if iteration % 100 == 0:
                    logger.info('Iteration %d, loss %f' % (iteration, loss))

                iteration += 1