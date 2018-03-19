#!/usr/bin/env python3

from baselines import logger
from baselines.common.cmd_util import make_atari_env, atari_arg_parser
from baselines.common.vec_env.vec_frame_stack import VecFrameStack
from a2c import learn
from baselines.ppo2.policies import CnnPolicy, LstmPolicy, LnLstmPolicy
from i2a_model_batch import I2A

def train(env_id, num_timesteps, seed, policy, lrschedule, num_env, args):
    if policy == 'i2a':
        policy_fn = I2A
    elif policy == 'cnn':
        policy_fn = CnnPolicy
    elif policy == 'lstm':
        policy_fn = LstmPolicy
    elif policy == 'lnlstm':
        policy_fn = LnLstmPolicy
    env = VecFrameStack(make_atari_env('MsPacmanNoFrameskip-v0', num_env, seed), 4)
    learn(policy_fn, env, seed, total_timesteps=int(num_timesteps * 1.1), lrschedule=lrschedule, args=args)
    env.close()

def main():
    parser = atari_arg_parser()
    parser.add_argument('--policy', help='Policy architecture', choices=['i2a', 'cnn', 'lstm', 'lnlstm'], default='i2a')
    parser.add_argument('--lrschedule', help='Learning rate schedule', choices=['constant', 'linear'], default='constant')
    parser.add_argument('--lr', help='Learning rate', type=float, default=7e-4)
    parser.add_argument('--lambda_dist', help='Distillation loss weight', type=float, default=0.01)
    parser.add_argument('--max_grad_norm', help='Max grad norm', type=float, default=0.5)
    args = parser.parse_args()
    logger.configure()
    train(args.env, num_timesteps=args.num_timesteps, seed=args.seed,
        policy=args.policy, lrschedule=args.lrschedule, num_env=16, args=args)

if __name__ == '__main__':
    main()
