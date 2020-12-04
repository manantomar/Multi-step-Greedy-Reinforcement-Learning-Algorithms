import os
import argparse
from functools import partial

import gym
from stable_baselines import bench, logger
from stable_baselines.common import set_global_seeds
from stable_baselines.common.atari_wrappers import make_atari
from stable_baselines.deepq_kvi import DQN, wrap_atari_dqn, CnnPolicy


def main():
    """
    Run the atari test
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', help='environment ID', default='BreakoutNoFrameskip-v4')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--prioritized', type=int, default=0)
    parser.add_argument('--dueling', type=int, default=0)
    parser.add_argument('--prioritized-replay-alpha', type=float, default=0.6)
    parser.add_argument('--num-timesteps', type=int, default=int(10e6))
    parser.add_argument('--checkpoint-freq', type=int, default=10000)
    parser.add_argument('--checkpoint-path', type=str, default=None)
    parser.add_argument('--kappa', default=1.0, type=float, help="Kappa value")
    parser.add_argument('--log', default=False, type=bool, help="True if you wanna log progress, else False")
    parser.add_argument('--run', default=0, type=int, help="which run?")
    parser.add_argument('--phi_grad_update_freq', default=1, type=int, help="gradient scaling")

    args = parser.parse_args()
    if not args.log:
        logger.configure(folder='./experiments/Atari/'+str(args.env)+'/final_nkappa_new/'+str(args.kappa)+'_'+str(args.run)+'_'+str(args.phi_grad_update_freq), format_strs=["csv"])
        checkpoint_path = "./experiments/Atari/"+str(args.env)+"/models/"+str(args.kappa)+"_"+str(args.run)+'_xnew'+str(args.phi_grad_update_freq)+'.pkl'
    else:
        logger.configure()
        checkpoint_path = None
    
    set_global_seeds(args.run)
    env = make_atari(args.env)
    env = bench.Monitor(env, logger.get_dir())
    env = wrap_atari_dqn(env)
    policy = partial(CnnPolicy, dueling=args.dueling == 1)

    #test_env = make_atari(args.env)
    #lives = test_env.env.ale.lives()
    #if lives == 0:
    #    lives = 1
    #test_env = bench.Monitor(test_env, None)
    #test_env = wrap_atari_dqn(test_env)
    test_env = None
    lives=1

    model = DQN(
        env=env,
        test_env=test_env,
        policy=policy,
        learning_rate=1e-4, #0.00025, #1e-4,
        buffer_size=1e5, #1e6, #1e5,
        exploration_fraction=0.1,
        exploration_final_eps=0.1, #0.01
        train_freq=4,
        learning_starts=10000, #50000,
        target_network_update_freq=1000, #10000,
        gamma=0.99,
        kappa=args.kappa,
        verbose=1,
        checkpoint_freq=args.checkpoint_freq,
        checkpoint_path=checkpoint_path,
        phi_grad_update_freq=args.phi_grad_update_freq,
        seed=args.run,
        eval_episodes=lives
    )
    #model = DQN.load(args.env+"/models/"+str(args.kappa)+"_0_xnew1.pkl", env, test_env=test_env, checkpoint_path=checkpoint_path, eval_episodes=lives, kappa=args.kappa)

    model.learn(total_timesteps=args.num_timesteps)

    env.close()


if __name__ == '__main__':
    main()
