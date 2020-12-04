import argparse

import gym
import numpy as np
import visdom; vis = visdom.Visdom()
from pyvirtualdisplay import Display 

from stable_baselines.common.cmd_util import make_mujoco_env, mujoco_arg_parser
from stable_baselines.trpo_kpi import TRPO
from stable_baselines import logger

display_ = Display(visible=0, size=(550, 500)) 
display_.start()

def main(args):
    """
    Run a trained model for the cartpole problem

    :param args: (ArgumentParser) the input arguments
    """

    logger.configure()
    env = make_mujoco_env(args.env, 0)
    model = TRPO.load(args.dir, env)

    while True:
        obs, done = env.reset(), False
        episode_rew = 0
        while not done:
            if not args.no_render:
                img = env.render(mode="rgb_array")
                img = np.array(img) / 255.
                img = np.transpose(img, (2, 0, 1))
                vis.image(img, win='frame')
            action, _, _, _ = model.policy_pi.step(obs.reshape(-1, *obs.shape), deterministic=True)
            obs, rew, done, _ = env.step(action)
            episode_rew += rew
        print("Episode reward", episode_rew)
        # No render is only used for automatic testing
        if args.no_render:
            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Enjoy trained DQN on cartpole")
    parser.add_argument('dir', type=str)
    parser.add_argument('--env', help='environment ID', type=str)
    parser.add_argument('--no-render', default=True, action="store_true", help="Disable rendering")
    args = parser.parse_args()
    main(args)
