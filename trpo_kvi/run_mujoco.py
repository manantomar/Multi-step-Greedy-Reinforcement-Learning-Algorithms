#!/usr/bin/env python3
# noinspection PyUnresolvedReferences
from mpi4py import MPI

from stable_baselines.common.cmd_util import make_mujoco_env, mujoco_arg_parser
from stable_baselines.common.policies import MlpPolicy
from stable_baselines import logger
from stable_baselines.trpo_kvi import TRPO
import stable_baselines.common.tf_util as tf_util


def train(env_id, num_timesteps, run, kappa, vf_phi_update_interval, log):
    """
    Train TRPO model for the mujoco environment, for testing purposes

    :param env_id: (str) Environment ID
    :param num_timesteps: (int) The total number of samples
    :param seed: (int) The initial seed for training
    """
    with tf_util.single_threaded_session():
        rank = MPI.COMM_WORLD.Get_rank()
        log_path = './experiments/'+str(env_id)+'./updated_nkappa_x7_ent_0.01_new/'+str(kappa)+'_'+str(vf_phi_update_interval)+'_'+str(run)
        if not log:
            if rank == 0:
                logger.configure(log_path)
            else:
                logger.configure(log_path, format_strs=[])
                logger.set_level(logger.DISABLED)
        else:
            if rank == 0:
                logger.configure()
            else:
                logger.configure(format_strs=[])
                logger.set_level(logger.DISABLED)
        seed = run
        workerseed = seed + 10000 * MPI.COMM_WORLD.Get_rank()

        env = make_mujoco_env(env_id, workerseed)
        test_env = None#make_mujoco_env(env_id, workerseed)
        model = TRPO(MlpPolicy, env, test_env=test_env, timesteps_per_batch=1024, max_kl=0.01, cg_iters=10, cg_damping=0.1, entcoeff=0.01,
                     gamma=0.99, kappa=kappa, vf_iters=5, vf_stepsize=1e-3, verbose=1, vf_phi_update_interval=vf_phi_update_interval, seed=run)
        model.learn(total_timesteps=num_timesteps, seed=run)
        #model.save("./"+str(env_id)+"./models/"+str(kappa)+"_"+str(run)+'_final_nkappa_x7_ent_0.01_'+str(vf_phi_update_interval)+'.pkl')
        env.close()


def main():
    """
    Runs the test
    """
    args = mujoco_arg_parser().parse_args()
    train(args.env, num_timesteps=args.num_timesteps, run=args.run, kappa=args.kappa, vf_phi_update_interval=args.vf_phi_update_interval, log=args.log)


if __name__ == '__main__':
    main()
