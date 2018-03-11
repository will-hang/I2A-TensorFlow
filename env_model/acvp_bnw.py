# import torch
# from torch.autograd import Variable
# import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np
# import pickle
# import torchvision
# from skimage import io
# from collections import deque
# import gym
# import torch.optim
# from utils import *
# from tqdm import tqdm
# from network import *

# from skimage.transform import resize

# PREFIX = '.'

# class Network(nn.Module, BasicNet):
#     def __init__(self, num_actions, gpu=0):
#         super(Network, self).__init__()

#         self.conv1 = nn.Conv2d(12, 64, 8, 2)
#         self.conv2 = nn.Conv2d(64, 128, 6, 2)
#         self.conv3 = nn.Conv2d(128, 128, 6, 2)
#         self.conv4 = nn.Conv2d(128, 128, 4, 2)

#         self.hidden_units = 128 * 3 * 3

#         self.fc5 = nn.Linear(self.hidden_units, 512)
#         self.fc_encode = nn.Linear(512, 512)
#         self.fc_action = nn.Linear(num_actions, 512)
#         self.fc_decode = nn.Linear(512, 512)
#         self.fc8 = nn.Linear(512, self.hidden_units)

#         self.deconv9 = nn.ConvTranspose2d(128, 128, 4, 2)
#         self.deconv10 = nn.ConvTranspose2d(128, 128, 6, 2)
#         self.deconv11 = nn.ConvTranspose2d(128, 128, 6, 2)
#         self.deconv12 = nn.ConvTranspose2d(128, 3, 8, 2)

#         self.init_weights()
#         self.criterion = nn.MSELoss()
#         self.opt = torch.optim.Adam(self.parameters(), 1e-4)

#         BasicNet.__init__(self, gpu)

#     def init_weights(self):
#         for layer in self.children():
#             if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.ConvTranspose2d):
#                 nn.init.xavier_uniform(layer.weight.data)
#                 nn.init.constant(layer.bias.data, 0)
#         nn.init.uniform(self.fc_encode.weight.data, -1, 1)
#         nn.init.uniform(self.fc_decode.weight.data, -1, 1)
#         nn.init.uniform(self.fc_action.weight.data, -0.1, 0.1)

#     def forward(self, obs, action):
#         x = F.relu(self.conv1(obs))
#         x = F.relu(self.conv2(x))
#         x = F.relu(self.conv3(x))
#         x = F.relu(self.conv4(x))
#         x = x.view((-1, self.hidden_units))
#         x = F.relu(self.fc5(x))
#         x = self.fc_encode(x)
#         action = self.fc_action(action)
#         x = torch.mul(x, action)
#         x = self.fc_decode(x)
#         x = F.relu(self.fc8(x))
#         x = x.view((-1, 128, 3, 3))
#         x = F.relu(self.deconv9(x))
#         x = F.relu(self.deconv10(x))
#         x = F.relu(self.deconv11(x))
#         x = F.pad(x, (0, 1, 0, 1))
#         x = self.deconv12(x)
#         return x

#     def fit(self, x, a, y):
#         x = self.variable(x)
#         a = self.variable(a)
#         y = self.variable(y)
#         y_ = self.forward(x, a)
#         loss = self.criterion(y_, y)
#         self.opt.zero_grad()
#         loss.backward()
#         for param in self.parameters():
#             param.grad.data.clamp_(-0.1, 0.1)
#         self.opt.step()
#         return np.asscalar(loss.cpu().data.numpy())

#     def evaluate(self, x, a, y):
#         x = self.variable(x)
#         a = self.variable(a)
#         y = self.variable(y)
#         y_ = self.forward(x, a)
#         loss = self.criterion(y_, y)
#         return np.asscalar(loss.cpu().data.numpy())

#     def predict(self, x, a):
#         x = self.variable(x)
#         a = self.variable(a)
#         return self.forward(x, a).cpu().data.numpy()

# def load_episode(game, ep, num_actions):
#     path = '%s/dataset/%s/%05d' % (PREFIX, game, ep)
#     with open('%s/action.bin' % (path), 'rb') as f:
#         actions = pickle.load(f)
#     num_frames = len(actions) + 1
#     frames = []

#     for i in range(1, num_frames):
#         frame = io.imread('%s/%05d.png' % (path, i))
#         frame = np.transpose(frame, (2, 0, 1))
#         frames.append(frame.astype(np.uint8))

#     actions = actions[1:]
#     encoded_actions = np.zeros((len(actions), num_actions))
#     encoded_actions[np.arange(len(actions)), actions] = 1

#     return frames, encoded_actions

# def extend_frames(frames, actions):
#     buffer = deque(maxlen=4)
#     extended_frames = []
#     targets = []

#     for i in range(len(frames) - 1):
#         buffer.append(frames[i])
#         if len(buffer) >= 4:
#             extended_frames.append(np.vstack(buffer))
#             targets.append(frames[i + 1])
#     actions = actions[3:, :]

#     return np.stack(extended_frames), actions, np.stack(targets)

# def train(game):
#     env = gym.make(game)
#     num_actions = env.action_space.n

#     config = Config()
#     net = Network(num_actions)

#     with open('%s/dataset/%s/meta.bin' % (PREFIX, game), 'rb') as f:
#         meta = pickle.load(f)
#     episodes = meta['episodes']
#     mean_obs = meta['mean_obs']
#     print(mean_obs.shape)
#     mean_obs = resize(mean_obs, (3, 96, 96))

#     def pre_process(x):
#         bsz, dep, _, _ = x.shape
#         x = np.array([resize(x[i, :, :, :], (dep, 96, 96), preserve_range=True) for i in range(bsz)]).astype(np.uint8)
#         # if x.shape[1] == 12:
#         #     x = (x - np.vstack([mean_obs] * 4)) / 255.0
#         # elif x.shape[1] == 3:
#         #     x = (x - mean_obs) / 255.0
#         # else:
#         #     assert False
#         # xx = []
#         # for i in range(0, x.shape[1], 3):
#         #     xx.append(x[:, i, :, :] * 0.299 + x[:, i + 1, :, :] * 0.587 + x[:, i + 2, :, :] * 0.114)
#         # xx = np.stack(xx, axis=1)
#         # x = xx
#         # return x
#         if x.shape[1] == 12:
#             return (x - np.vstack([mean_obs] * 4)) / 255.0
#         elif x.shape[1] == 3:
#             return (x - mean_obs) / 255.0
#         else:
#             assert False

#     def post_process(y):
#         #return (y * 255 + mean_obs[0, :, :] * 0.299 + mean_obs[1, :, :] * 0.587 + mean_obs[2, :, :] * 0.114).astype(np.uint8), (y * 255).astype(np.uint8)
#         return (y * 255 + mean_obs).astype(np.uint8), (y * 255).astype(np.uint8)

#     train_episodes = int(episodes * 0.95)
#     indices_train = np.arange(train_episodes)
#     iteration = 0
#     while True:
#         np.random.shuffle(indices_train)
#         for ep in indices_train:
#             frames, actions = load_episode(game, ep, num_actions)
#             frames, actions, targets = extend_frames(frames, actions)
#             batcher = Batcher(32, [frames, actions, targets])
#             batcher.shuffle()
#             while not batcher.end():
#                 if iteration % 100 == 0:
#                     mkdir('data_bnw/acvp-sample')
#                     losses = [0]
#                     test_indices = range(train_episodes, episodes)
#                     ep_to_print = np.random.choice(test_indices)
#                     for test_ep in test_indices:
#                         # frames, actions = load_episode(game, test_ep, num_actions)
#                         # frames, actions, targets = extend_frames(frames, actions)
#                         #  test_batcher = Batcher(32, [frames, actions, targets])
#                         #while not test_batcher.end():
#                         #    pass
#                             #x, a, y = test_batcher.next_batch()
#                             #losses.append(net.evaluate(pre_process(x), a, pre_process(y)))
#                         if test_ep == ep_to_print:
#                             test_batcher = Batcher(32, [frames, actions, targets])
#                             test_batcher.reset()
#                             x, a, y = test_batcher.next_batch()
#                             preds = net.predict(pre_process(x), a)
#                             y_, y__y = post_process(preds)
#                             torchvision.utils.save_image(torch.from_numpy(y_), 'data_bnw/acvp-sample/%s-%09d.png' % (game, iteration))
#                             torchvision.utils.save_image(torch.from_numpy(y__y), 'data_bnw/acvp-sample/%s-%09dpoopybutt.png' % (game, iteration))
#                             torchvision.utils.save_image(torch.from_numpy(y), 'data_bnw/acvp-sample/%s-%09d-truth.png' % (game, iteration))
#                             np.save('data_bnw/acvp-sample/%s-%09d' % (game, iteration), preds)

#                     logger.info('Iteration %d, test loss %f' % (iteration, np.mean(losses)))

#                 x, a, y = batcher.next_batch()
#                 loss = net.fit(pre_process(x), a, pre_process(y))
#                 if iteration % 100 == 0:
#                     logger.info('Iteration %d, loss %f' % (iteration, loss))

#                 iteration += 1

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle
import torchvision
from skimage import io
from collections import deque
import gym
import torch.optim
from utils import *
from tqdm import tqdm
from network import *

from skimage.transform import resize

PREFIX = '.'

class Network(nn.Module, BasicNet):
    def __init__(self, num_actions, gpu=0):
        super(Network, self).__init__()

        self.conv1 = nn.Conv2d(12, 64, 8, 2)
        self.conv2 = nn.Conv2d(64, 128, 6, 2)
        self.conv3 = nn.Conv2d(128, 128, 6, 2)
        self.conv4 = nn.Conv2d(128, 128, 4, 2)

        self.hidden_units = 128 * 3 * 3

        self.fc5 = nn.Linear(self.hidden_units, 512)
        self.fc_encode = nn.Linear(512, 512)
        self.fc_action = nn.Linear(num_actions, 512)
        self.fc_decode = nn.Linear(512, 512)
        self.fc8 = nn.Linear(512, self.hidden_units)

        self.deconv9 = nn.ConvTranspose2d(128, 128, 4, 2)
        self.deconv10 = nn.ConvTranspose2d(128, 128, 6, 2)
        self.deconv11 = nn.ConvTranspose2d(128, 128, 6, 2)
        self.deconv12 = nn.ConvTranspose2d(128, 3, 8, 2)

        self.init_weights()
        self.criterion = nn.MSELoss()
        self.opt = torch.optim.Adam(self.parameters(), 1e-4)

        BasicNet.__init__(self, gpu)

    def init_weights(self):
        for layer in self.children():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.ConvTranspose2d):
                nn.init.xavier_uniform(layer.weight.data)
                nn.init.constant(layer.bias.data, 0)
        nn.init.uniform(self.fc_encode.weight.data, -1, 1)
        nn.init.uniform(self.fc_decode.weight.data, -1, 1)
        nn.init.uniform(self.fc_action.weight.data, -0.1, 0.1)

    def forward(self, obs, action):
        x = F.relu(self.conv1(obs))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.view((-1, self.hidden_units))
        x = F.relu(self.fc5(x))
        x = self.fc_encode(x)
        action = self.fc_action(action)
        x = torch.mul(x, action)
        x = self.fc_decode(x)
        x = F.relu(self.fc8(x))
        x = x.view((-1, 128, 3, 3))
        x = F.relu(self.deconv9(x))
        x = F.relu(self.deconv10(x))
        x = F.relu(self.deconv11(x))
        x = F.pad(x, (0, 1, 0, 1))
        x = self.deconv12(x)
        return x

    def fit(self, x, a, y):
        x = self.variable(x)
        a = self.variable(a)
        y = self.variable(y)
        y_ = self.forward(x, a)
        loss = self.criterion(y_, y)
        self.opt.zero_grad()
        loss.backward()
        for param in self.parameters():
            param.grad.data.clamp_(-0.1, 0.1)
        self.opt.step()
        return np.asscalar(loss.cpu().data.numpy())

    def evaluate(self, x, a, y):
        x = self.variable(x)
        a = self.variable(a)
        y = self.variable(y)
        y_ = self.forward(x, a)
        loss = self.criterion(y_, y)
        return np.asscalar(loss.cpu().data.numpy())

    def predict(self, x, a):
        x = self.variable(x)
        a = self.variable(a)
        return self.forward(x, a).cpu().data.numpy()

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

def train(game):
    env = gym.make(game)
    num_actions = env.action_space.n

    config = Config()
    net = Network(num_actions)

    with open('%s/dataset/%s/meta.bin' % (PREFIX, game), 'rb') as f:
        meta = pickle.load(f)
    episodes = meta['episodes']
    mean_obs = meta['mean_obs']
    print(mean_obs.shape)
    mean_obs = resize(mean_obs, (3, 96, 96))

    def pre_process(x):
        bsz, dep, _, _ = x.shape
        x = np.array([resize(x[i, :, :, :], (dep, 96, 96), preserve_range=True) for i in range(bsz)]).astype(np.uint8)
        if x.shape[1] == 12:
            return (x - np.vstack([mean_obs] * 4)) / 255.0
        elif x.shape[1] == 3:
            return (x - mean_obs) / 255.0
        else:
            assert False

    def post_process(y):
        return (y * 255 + mean_obs).astype(np.uint8), (y * 255).astype(np.uint8)

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
                if iteration % 100 == 0:
                    mkdir('data_pyt/acvp-sample')
                    losses = []
                    test_indices = range(train_episodes, episodes)
                    ep_to_print = np.random.choice(test_indices)
                    for test_ep in tqdm(test_indices):
                        frames, actions = load_episode(game, test_ep, num_actions)
                        frames, actions, targets = extend_frames(frames, actions)
                        test_batcher = Batcher(32, [frames, actions, targets])
                        while not test_batcher.end():
                            x, a, y = test_batcher.next_batch()
                            losses.append(net.evaluate(pre_process(x), a, pre_process(y)))
                        if test_ep == ep_to_print:
                            test_batcher.reset()
                            x, a, y = test_batcher.next_batch()
                            preds = net.predict(pre_process(x), a)
                            y_, y__y = post_process(preds)
                            torchvision.utils.save_image(torch.from_numpy(y_), 'data_bnw/acvp-sample/%s-%09d.png' % (game, iteration))
                            torchvision.utils.save_image(torch.from_numpy(y__y), 'data_bnw/acvp-sample/%s-%09dpoopybutt.png' % (game, iteration))
                            torchvision.utils.save_image(torch.from_numpy(y), 'data_bnw/acvp-sample/%s-%09d-truth.png' % (game, iteration))
                            np.save('data_bnw/acvp-sample/%s-%09d' % (game, iteration), preds)

                    logger.info('Iteration %d, test loss %f' % (iteration, np.mean(losses)))

                x, a, y = batcher.next_batch()
                loss = net.fit(pre_process(x), a, pre_process(y))
                if iteration % 100 == 0:
                    logger.info('Iteration %d, loss %f' % (iteration, loss))

                iteration += 1