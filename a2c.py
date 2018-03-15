import os
import os.path as osp
import gym
import time
import joblib
import logging
import numpy as np
import tensorflow as tf
from baselines import logger

from baselines.common import set_global_seeds, explained_variance
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common.atari_wrappers import wrap_deepmind
from baselines.common import tf_util

from baselines.a2c.utils import discount_with_dones
from baselines.a2c.utils import Scheduler, make_path, find_trainable_variables
from baselines.a2c.utils import cat_entropy, mse

from i2a_model_batch import EnvironmentModel
import matplotlib.pyplot as plt

DISPLAY_TIME = True

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
    env_model = None

class Model(object):

    def __init__(self, policy, ob_space, ac_space, nenvs, nsteps,
            ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5, lr=7e-4,
            alpha=0.99, epsilon=1e-5, lambda_dist=0.01, total_timesteps=int(20e6), lrschedule='linear'):

        sess = tf.get_default_session()
        nact = ac_space.n
        nbatch = nenvs*nsteps

        A = tf.placeholder(tf.int32, [nbatch])
        ADV = tf.placeholder(tf.float32, [nbatch])
        R = tf.placeholder(tf.float32, [nbatch])
        LR = tf.placeholder(tf.float32, [])

        config = Config()

        act_model = policy(config)
        config.reuse = True
        train_model = policy(config)

        neglogpac = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=train_model.logits, labels=A)
        pg_loss = tf.reduce_mean(ADV * neglogpac)
        vf_loss = tf.reduce_mean(mse(tf.squeeze(train_model.vf), R))
        entropy = tf.reduce_mean(cat_entropy(train_model.logits))

        aux_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=train_model.rp_logits, labels=A)
        loss = pg_loss - entropy * ent_coef + vf_loss * vf_coef + aux_loss * lambda_dist

        params = find_trainable_variables("model")
        grads = tf.gradients(loss, params)
        if max_grad_norm is not None:
            grads, grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
        grads = list(zip(grads, params))
        trainer = tf.train.RMSPropOptimizer(learning_rate=LR, decay=alpha, epsilon=epsilon)
        _train = trainer.apply_gradients(grads)

        lr = Scheduler(v=lr, nvalues=total_timesteps, schedule=lrschedule)

        saver = tf.train.Saver()

        def train(obs, rs, rr, rewards, masks, actions, values):
            advs = rewards - values
            for step in range(len(obs)):
                cur_lr = lr.value()
            td_map =    {
                            train_model.X:obs, 
                            A:actions, 
                            ADV:advs, 
                            R:rewards, 
                            LR:cur_lr, 
                            train_model.inputs_s: rs, 
                            train_model.inputs_r: rr
                        }

            policy_loss, value_loss, policy_entropy, _ = sess.run(
                [pg_loss, vf_loss, entropy, _train],
                td_map
            )
            return policy_loss, value_loss, policy_entropy

        def save(save_path):
            saver.save(sess, save_path + 'model.ckpt')

        def load(load_path):
            saver.restore(sess, load_path + 'model.ckpt')

        self.train = train
        self.train_model = train_model
        self.act_model = act_model
        self.act = act_model.act
        self.value = act_model.value
        self.save = save
        self.load = load

class Runner(object):

    def __init__(self, env, model, nsteps=5, gamma=0.99):
        config = Config()
        self.env = env
        self.model = model
        nh, nw, nc = env.observation_space.shape
        self.nenv = env.num_envs
        self.batch_ob_shape = (self.nenv*nsteps, nh, nw, nc)
        self.batch_rs_shape = (self.nenv*nsteps, config.n_actions, config.rollout_length, nh, nw, nc)
        self.batch_rr_shape = (self.nenv*nsteps, config.n_actions, config.rollout_length)
        self.obs = np.zeros((self.nenv, nh, nw, nc), dtype=np.uint8)
        self.nc = nc
        obs = env.reset()
        self.gamma = gamma
        self.nsteps = nsteps
        self.dones = [False for _ in range(self.nenv)]

        self.ep_rewards = np.zeros(self.nenv)
        self.tot_rewards = []

    def run(self):
        self.tot_rewards = []
        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones = [],[],[],[],[]
        mb_rs, mb_rr = [], []
        for n in range(self.nsteps):
            actions, values, rs, rr = self.model.act(self.obs)

            actions = np.array(actions)
            values = np.array(values)
            mb_rs.append(rs)
            mb_rr.append(rr)

            mb_obs.append(np.copy(self.obs))
            mb_actions.append(actions)
            mb_values.append(values)
            mb_dones.append(self.dones)
            obs, rewards, dones, _ = self.env.step(actions)

            self.ep_rewards += rewards

            self.dones = dones
            for n, done in enumerate(dones):
                if done:
                    self.tot_rewards.append(self.ep_rewards[n])
                    self.ep_rewards[n] = 0
                    self.obs[n] = self.obs[n]*0
            self.obs = obs
            mb_rewards.append(rewards)
        mb_dones.append(self.dones)

        #batch of steps to batch of rollouts
        mb_obs = np.asarray(mb_obs, dtype=np.uint8).swapaxes(1, 0).reshape(self.batch_ob_shape)
        mb_rs = np.asarray(mb_rs, dtype=np.uint8).swapaxes(1, 0).reshape(self.batch_rs_shape)
        mb_rr = np.asarray(mb_rr, dtype=np.uint8).swapaxes(1, 0).reshape(self.batch_rr_shape)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32).swapaxes(1, 0)
        mb_actions = np.asarray(mb_actions, dtype=np.int32).swapaxes(1, 0)
        mb_values = np.asarray(mb_values, dtype=np.float32).swapaxes(1, 0)
        mb_dones = np.asarray(mb_dones, dtype=np.bool).swapaxes(1, 0)
        mb_masks = mb_dones[:, :-1]
        mb_dones = mb_dones[:, 1:]
        last_values = []
        for i in range(self.nenv):
            last_values.append(self.model.value(np.expand_dims(self.obs[i], axis=0)).tolist())
        #discount/bootstrap off value fn
        for n, (rewards, dones, value) in enumerate(zip(mb_rewards, mb_dones, last_values)):
            rewards = rewards.tolist()
            dones = dones.tolist()
            if dones[-1] == 0:
                rewards = discount_with_dones(rewards+[value], dones+[0], self.gamma)[:-1]
            else:
                rewards = discount_with_dones(rewards, dones, self.gamma)
            mb_rewards[n] = rewards
        mb_rewards = mb_rewards.flatten()
        mb_actions = mb_actions.flatten()
        mb_values = mb_values.flatten()
        mb_masks = mb_masks.flatten()

        ep_reward_means = np.mean(self.tot_rewards) if len(self.tot_rewards) > 0 else None
        return mb_obs, mb_rs, mb_rr, mb_rewards, mb_masks, mb_actions, mb_values, ep_reward_means

def learn(policy, env, seed, nsteps=5, total_timesteps=int(80e6), vf_coef=0.5, ent_coef=0.01, max_grad_norm=0.5, lr=7e-4, lrschedule='linear', epsilon=1e-5, alpha=0.99, gamma=0.99, log_interval=100, args=None):
    tf.reset_default_graph()
    set_global_seeds(seed)

    with tf.Session() as sess:
        nenvs = env.num_envs
        ob_space = env.observation_space
        ac_space = env.action_space
        model = Model(policy=policy, ob_space=ob_space, ac_space=ac_space, nenvs=nenvs, nsteps=nsteps, ent_coef=ent_coef, vf_coef=vf_coef,
            max_grad_norm=args.max_grad_norm, lr=args.lr, alpha=alpha, epsilon=epsilon, total_timesteps=total_timesteps, lrschedule=lrschedule, lambda_dist=args.lambda_dist)
        # this next single line of code is responsible for 2 hours of my life lost, never to be seen again
        tf.global_variables_initializer().run(session=sess)

        config = Config()
        env_model = EnvironmentModel(config)

        model.act_model.imagination_core.set_env(env_model)
        model.train_model.imagination_core.set_env(env_model)
        
        runner = Runner(env, model, nsteps=nsteps, gamma=gamma)
        file_writer = tf.summary.FileWriter('results/', sess.graph)
        nbatch = nenvs*nsteps
        tstart = time.time()

        entropy = []
        reward_list = []
        log_rewards = []
        for update in range(1, total_timesteps//nbatch+1):
            if DISPLAY_TIME:
                s = time.time()
            obs, rs, rr, rewards, masks, actions, values, ep_reward_means = runner.run()
            if DISPLAY_TIME:
                e = time.time()
                print('Running took {} seconds'.format(e - s))
                s = time.time()
            policy_loss, value_loss, policy_entropy = model.train(obs, rs, rr, rewards, masks, actions, values)
            if DISPLAY_TIME:
                e = time.time()
                print('Training took {} seconds'.format(e - s))
            
            nseconds = time.time()-tstart
            fps = int((update*nbatch)/nseconds)

            if ep_reward_means != None:
                summary = tf.Summary()
                summary.value.add(tag='Rewards', simple_value=ep_reward_means)
                file_writer.add_summary(summary, update*nbatch)
                reward_list.append([update*nbatch, ep_reward_means])
                np.save('checkpoints/rewards', reward_list)
                print('LOG: Mean rewards - {}'.format(ep_reward_means))
                log_rewards.append(ep_reward_means)

            if update % log_interval == 0 or update == 1:
                ev = explained_variance(values, rewards)
                logger.record_tabular("nupdates", update)
                logger.record_tabular("total_timesteps", update*nbatch)
                logger.record_tabular("fps", fps)
                logger.record_tabular("policy_entropy", float(policy_entropy))
                logger.record_tabular("value_loss", float(value_loss))
                logger.record_tabular("explained_variance", float(ev))
                if len(log_rewards) > 0:
                    logger.record_tabular("mean_episode_rewards", np.mean(log_rewards))
                    log_rewards = []
                logger.dump_tabular()

                summary = tf.Summary()
                summary.value.add(tag='Entropy', simple_value=policy_entropy)
                    
                file_writer.add_summary(summary, update*nbatch)
                file_writer.flush()

                print('LOG: Saved checkpoint!')
                model.save("checkpoints/")

                entropy.append([update*nbatch, policy_entropy])
                np.save('checkpoints/entropy', entropy)

        env.close()
