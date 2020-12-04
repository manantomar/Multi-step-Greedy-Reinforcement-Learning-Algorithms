import time
from functools import partial
from contextlib import contextmanager
from collections import deque

import tensorflow as tf
import numpy as np
import gym

from stable_baselines import logger, deepq_kvi
from stable_baselines.common import tf_util, OffPolicyRLModel, SetVerbosity, TensorboardWriter, colorize
from stable_baselines.common.vec_env import VecEnv
from stable_baselines.common.schedules import LinearSchedule, PiecewiseSchedule
from stable_baselines.deepq_kvi.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from stable_baselines.deepq_kvi.policies import DQNPolicy
from stable_baselines.a2c.utils import total_episode_reward_logger


class DQN(OffPolicyRLModel):
    """
    The DQN model class. DQN paper: https://arxiv.org/pdf/1312.5602.pdf

    :param policy: (DQNPolicy or str) The policy model to use (MlpPolicy, CnnPolicy, LnMlpPolicy, ...)
    :param env: (Gym environment or str) The environment to learn from (if registered in Gym, can be str)
    :param gamma: (float) discount factor
    :param learning_rate: (float) learning rate for adam optimizer
    :param buffer_size: (int) size of the replay buffer
    :param exploration_fraction: (float) fraction of entire training period over which the exploration rate is
            annealed
    :param exploration_final_eps: (float) final value of random action probability
    :param train_freq: (int) update the model every `train_freq` steps. set to None to disable printing
    :param batch_size: (int) size of a batched sampled from replay buffer for training
    :param checkpoint_freq: (int) how often to save the model. This is so that the best version is restored at the
            end of the training. If you do not wish to restore the best version
            at the end of the training set this variable to None.
    :param checkpoint_path: (str) replacement path used if you need to log to somewhere else than a temporary
            directory.
    :param learning_starts: (int) how many steps of the model to collect transitions for before learning starts
    :param target_network_update_freq: (int) update the target network every `target_network_update_freq` steps.
    :param param_noise: (bool) Whether or not to apply noise to the parameters of the policy.
    :param verbose: (int) the verbosity level: 0 none, 1 training information, 2 tensorflow debug
    :param _init_setup_model: (bool) Whether or not to build the network at the creation of the instance
        WARNING: this logging can take a lot of space quickly
    """

    def __init__(self, policy, env, test_env=None, gamma=0.99, kappa=1.0, learning_rate=5e-4, buffer_size=50000, exploration_fraction=0.1,
                 exploration_final_eps=0.02, train_freq=1, batch_size=32, checkpoint_freq=10000, checkpoint_path=None,
                 learning_starts=1000, target_network_update_freq=500, phi_grad_update_freq=1, seed=0,
                 eval_episodes=5, param_noise=False, verbose=0, policy_phi=None, policy_phi_kwargs=None,
                 _init_setup_model=True, policy_kwargs=None):

        # TODO: replay_buffer refactoring
        super(DQN, self).__init__(policy=policy, env=env, replay_buffer=None, verbose=verbose, policy_base=DQNPolicy,
                                  requires_vec_env=False, policy_kwargs=policy_kwargs, policy_phi=policy_phi, policy_phi_kwargs=policy_phi_kwargs)

        self.checkpoint_path = checkpoint_path
        self.param_noise = param_noise
        self.learning_starts = learning_starts
        self.train_freq = train_freq
        self.batch_size = batch_size
        self.target_network_update_freq = target_network_update_freq
        self.checkpoint_freq = checkpoint_freq
        self.exploration_final_eps = exploration_final_eps
        self.exploration_fraction = exploration_fraction
        self.buffer_size = buffer_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.kappa = kappa
        self.seed = seed

        self.graph = None
        self.sess = None
        self._train_step = None
        self.step_model = None
        self.update_target = None
        self.act = None
        self.proba_step = None
        self.replay_buffer = None
        self.beta_schedule = None
        self.exploration = None
        self.params = None
        self.summary = None
        self.episode_reward = None
        self.test_env= test_env
        self.eval_episodes = eval_episodes
        self.phi_grad_update_freq = phi_grad_update_freq

        if _init_setup_model:
            self.setup_model()

    def _get_pretrain_placeholders(self):
        policy = self.step_model
        return policy.obs_ph, tf.placeholder(tf.int32, [None]), policy.q_values

    def setup_model(self):

        with SetVerbosity(self.verbose):
            assert not isinstance(self.action_space, gym.spaces.Box), \
                "Error: DQN cannot output a gym.spaces.Box action space."

            # If the policy is wrap in functool.partial (e.g. to disable dueling)
            # unwrap it to check the class type
            if isinstance(self.policy, partial):
                test_policy = self.policy.func
            else:
                test_policy = self.policy
            assert issubclass(test_policy, DQNPolicy), "Error: the input policy for the DQN model must be " \
                                                       "an instance of DQNPolicy."

            self.graph = tf.Graph()
            with self.graph.as_default():
                self.sess = tf_util.make_session(graph=self.graph)
                self._setup_learn(self.seed)

                optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
                #optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate, momentum=0.95, epsilon=0.01)

                self.act, self._train_step, self.update_target, self.update_target_phi, self.step_model, _ = deepq_kvi.build_train(
                    q_func=partial(self.policy, **self.policy_kwargs),
                    ob_space=self.observation_space,
                    ac_space=self.action_space,
                    optimizer=optimizer,
                    gamma=self.gamma,
                    kappa=self.kappa,
                    grad_norm_clipping=10,
                    param_noise=self.param_noise,
                    sess=self.sess
                )
                self.proba_step = self.step_model.proba_step
                self.params = tf_util.get_trainable_vars("deepq")

                @contextmanager
                def timed(msg):
                    if self.verbose >= 1:
                        print(colorize(msg, color='magenta'))
                        start_time = time.time()
                        yield
                        print(colorize("done in {:.3f} seconds".format((time.time() - start_time)),
                                           color='magenta'))
                    else:
                        yield 

                # Initialize the parameters and copy them to the target network.
                tf_util.initialize(self.sess)
                self.update_target(sess=self.sess)

                self.timed = timed
                self.summary = tf.summary.merge_all()

    def learn(self, total_timesteps, callback=None, seed=None, log_interval=100, tb_log_name="DQN",
              reset_num_timesteps=True, replay_wrapper=None):

        new_tb_log = self._init_num_timesteps(reset_num_timesteps)
        print("args are", self.kappa, self.phi_grad_update_freq, self.seed, np.random.randint(100))

        with SetVerbosity(self.verbose): 

            # Create the replay buffer
            self.replay_buffer = ReplayBuffer(self.buffer_size)
            self.beta_schedule = None

            # Create the schedule for exploration starting from 1.
            self.exploration = LinearSchedule(schedule_timesteps=int(self.exploration_fraction * total_timesteps),
                                              initial_p=1.0,
                                              final_p=self.exploration_final_eps)
            #self.exploration = PiecewiseSchedule([(0,        1.0), (int(1e6), 0.1), (int(1e7), 0.01)], outside_value=0.01)

            episode_rewards = [0.0]
            episode_successes = []
            #td_errors_mean = []
            #td_phi_errors_mean = []
            obs = self.env.reset()
            reset = True
            self.episode_reward = np.zeros((1,))

            for _ in range(total_timesteps):
                #if callback is not None:
                    # Only stop training if return value is False, not when it is None. This is for backwards
                    # compatibility with callbacks that have no return statement.
                #    if callback(locals(), globals()) is False:
                #        break
                # Take action and update exploration to the newest value
                kwargs = {}
                if not self.param_noise:
                    update_eps = self.exploration.value(self.num_timesteps)
                    update_param_noise_threshold = 0.
                else:
                    update_eps = 0.
                    # Compute the threshold such that the KL divergence between perturbed and non-perturbed
                    # policy is comparable to eps-greedy exploration with eps = exploration.value(t).
                    # See Appendix C.1 in Parameter Space Noise for Exploration, Plappert et al., 2017
                    # for detailed explanation.
                    update_param_noise_threshold = \
                        -np.log(1. - self.exploration.value(self.num_timesteps) +
                                self.exploration.value(self.num_timesteps) / float(self.env.action_space.n))
                    kwargs['reset'] = reset
                    kwargs['update_param_noise_threshold'] = update_param_noise_threshold
                    kwargs['update_param_noise_scale'] = True
                with self.sess.as_default():
                    action = self.act(np.array(obs)[None], update_eps=update_eps, **kwargs)[0]
                env_action = action
                reset = False
                new_obs, rew, done, info = self.env.step(env_action)
                
                # Store transition in the replay buffer.
                self.replay_buffer.add(obs, action, rew, new_obs, float(done))
                obs = new_obs

                episode_rewards[-1] += rew
                if done:
                    maybe_is_success = info.get('is_success')
                    if maybe_is_success is not None:
                        episode_successes.append(float(maybe_is_success))
                    if not isinstance(self.env, VecEnv):
                        obs = self.env.reset()
                    episode_rewards.append(0.0)
                    reset = True

                # Do not train if the warmup phase is not over
                # or if there are not enough samples in the replay buffer
                can_sample = self.replay_buffer.can_sample(self.batch_size)
                if can_sample and self.num_timesteps > self.learning_starts \
                    and self.num_timesteps % self.train_freq == 0:
                    # Minimize the error in Bellman's equation on a batch sampled from replay buffer.
                    obses_t, actions, rewards, obses_tp1, dones = self.replay_buffer.sample(self.batch_size)
                    weights, batch_idxes = np.ones_like(rewards), None

                    # Use v_phi as zero until buffere is filled
                    if self.num_timesteps <= self.buffer_size:
                        weights = np.zeros_like(rewards)

                    with self.sess.as_default():
                        #actions_policy = self.act(obses_t)
                        actions_policy_phi = self.act(obses_tp1)
                    
                    _, td_errors = self._train_step(obses_t, actions, actions_policy_phi, actions_policy_phi, rewards, obses_tp1, obses_tp1, obses_tp1, obses_tp1, dones, weights,
                                                        sess=self.sess)
                    #td_errors_mean.append(np.mean(td_errors))  

                if can_sample and self.num_timesteps > self.learning_starts and \
                        self.num_timesteps % self.target_network_update_freq == 0:
                    # Update target network periodically.
                    self.update_target(sess=self.sess)

                if can_sample and self.kappa != 1.0 and self.num_timesteps >= self.buffer_size and \
                        self.num_timesteps % (self.phi_grad_update_freq * self.train_freq) == 0:
                    # Update target network periodically.
                    self.update_target_phi(sess=self.sess)

                if len(episode_rewards[-101:-1]) == 0:
                    mean_100ep_reward = -np.inf
                else:
                    mean_100ep_reward = round(float(np.mean(episode_rewards[-101:-1])), 1)

                num_episodes = len(episode_rewards)
                if self.verbose >= 1 and done and log_interval is not None and len(episode_rewards) % log_interval == 0:
                    with self.timed("eval time"):
                        if self.test_env is not None and len(episode_rewards) % (10 * log_interval) == 0:
                            eval_return, actual_return = self.evaluate_agent(self.test_env)
                        else:
                            eval_return, actual_return = None, None

                    logger.record_tabular("steps", self.num_timesteps)
                    logger.record_tabular("episodes", num_episodes)
                    if len(episode_successes) > 0:
                        logger.logkv("success rate", np.mean(episode_successes[-100:]))
                    logger.record_tabular("mean 100 episode reward", mean_100ep_reward)
                    logger.record_tabular("eval return", eval_return)
                    logger.record_tabular("actual return", actual_return)
                    #logger.record_tabular("td errors", np.mean(td_errors_mean))
                    #logger.record_tabular("td errors phi", np.mean(td_phi_errors_mean))
                    logger.record_tabular("% time spent exploring",
                                          int(100 * self.exploration.value(self.num_timesteps)))
                    logger.dump_tabular()

                    #td_errors_mean = []
                    #td_phi_errors_mean = []

                if self.checkpoint_path is not None and self.num_timesteps % self.checkpoint_freq == 0:
                    self.save(self.checkpoint_path)

                self.num_timesteps += 1

        return self

    def predict(self, observation, state=None, mask=None, deterministic=True):
        observation = np.array(observation)
        vectorized_env = self._is_vectorized_observation(observation, self.observation_space)

        observation = observation.reshape((-1,) + self.observation_space.shape)
        with self.sess.as_default():
            actions, _, _ = self.step_model.step(observation, deterministic=deterministic)

        if not vectorized_env:
            actions = actions[0]

        return actions, None

    def evaluate_agent(self, test_env):
        test_reward = []
        test_actual_reward = []
        for _ in range(1): #10):
            episode_actual_rew = 0
            for _ in range(self.eval_episodes):
                obs_eval, done_eval = test_env.reset(), False
                episode_rew = 0
                while not done_eval:
                    action_eval, _ = self.predict(obs_eval)
                    obs_eval, rew_eval, done_eval, _ = test_env.step(action_eval)
                    episode_rew += rew_eval
                    episode_actual_rew += test_env.get_actual_reward() 
                test_reward.append(episode_rew)
            test_actual_reward.append(episode_actual_rew)
            obs_eval = test_env.reset()
        
        # random test
        #observation = np.array(obs_eval)
        #observation = observation.reshape((-1,) + self.observation_space.shape)
        #with self.sess.as_default():
        #    actions, _, _ = self.step_model.step(observation, deterministic=True)
        #    print("action is", actions)
        #    actions = self.act(observation, stochastic=False)
        #    print("new action is", actions)

        return np.mean(test_reward), np.mean(test_actual_reward)

    def action_probability(self, observation, state=None, mask=None, actions=None, logp=False):
        observation = np.array(observation)
        vectorized_env = self._is_vectorized_observation(observation, self.observation_space)

        observation = observation.reshape((-1,) + self.observation_space.shape)
        actions_proba = self.proba_step(observation, state, mask)

        if actions is not None:  # comparing the action distribution, to given actions
            actions = np.array([actions])
            assert isinstance(self.action_space, gym.spaces.Discrete)
            actions = actions.reshape((-1,))
            assert observation.shape[0] == actions.shape[0], "Error: batch sizes differ for actions and observations."
            actions_proba = actions_proba[np.arange(actions.shape[0]), actions]
            # normalize action proba shape
            actions_proba = actions_proba.reshape((-1, 1))
            if logp:
                actions_proba = np.log(actions_proba)

        if not vectorized_env:
            if state is not None:
                raise ValueError("Error: The environment must be vectorized when using recurrent policies.")
            actions_proba = actions_proba[0]

        return actions_proba

    def get_parameter_list(self):
        return self.params

    def save(self, save_path):
        # params
        data = {
            "checkpoint_path": self.checkpoint_path,
            "param_noise": self.param_noise,
            "learning_starts": self.learning_starts,
            "train_freq": self.train_freq,
            "batch_size": self.batch_size,
            "target_network_update_freq": self.target_network_update_freq,
            "checkpoint_freq": self.checkpoint_freq,
            "exploration_final_eps": self.exploration_final_eps,
            "exploration_fraction": self.exploration_fraction,
            "learning_rate": self.learning_rate,
            "gamma": self.gamma,
            "verbose": self.verbose,
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "policy": self.policy,
            "n_envs": self.n_envs,
            "_vectorize_action": self._vectorize_action,
            "policy_kwargs": self.policy_kwargs,
            "eval_episodes": self.eval_episodes,
            "phi_grad_update_freq": self.phi_grad_update_freq
        }

        params_to_save = self.get_parameters()

        self._save_to_file(save_path, data=data, params=params_to_save)
