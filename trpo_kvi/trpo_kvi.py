import time
from contextlib import contextmanager
from collections import deque

import gym
from mpi4py import MPI
import tensorflow as tf
import numpy as np

import stable_baselines.common.tf_util as tf_util
from stable_baselines.common import explained_variance, zipsame, dataset, fmt_row, colorize, ActorCriticRLModel, \
    SetVerbosity, TensorboardWriter
from stable_baselines import logger
from stable_baselines.common.mpi_adam import MpiAdam
from stable_baselines.common.cg import conjugate_gradient
from stable_baselines.common.policies import ActorCriticPolicy
from stable_baselines.a2c.utils import total_episode_reward_logger
from stable_baselines.trpo_kvi.utils import traj_segment_generator, add_vtarg_and_adv, flatten_lists


class TRPO(ActorCriticRLModel):
    """
    Trust Region Policy Optimization (https://arxiv.org/abs/1502.05477)

    :param policy: (ActorCriticPolicy or str) The policy model to use (MlpPolicy, CnnPolicy, CnnLstmPolicy, ...)
    :param env: (Gym environment or str) The environment to learn from (if registered in Gym, can be str)
    :param gamma: (float) the discount value
    :param timesteps_per_batch: (int) the number of timesteps to run per batch (horizon)
    :param max_kl: (float) the Kullback-Leibler loss threshold
    :param cg_iters: (int) the number of iterations for the conjugate gradient calculation
    :param lam: (float) GAE factor
    :param entcoeff: (float) the weight for the entropy loss
    :param cg_damping: (float) the compute gradient dampening factor
    :param vf_stepsize: (float) the value function stepsize
    :param vf_iters: (int) the value function's number iterations for learning
    :param verbose: (int) the verbosity level: 0 none, 1 training information, 2 tensorflow debug
    :param _init_setup_model: (bool) Whether or not to build the network at the creation of the instance
    :param policy_kwargs: (dict) additional arguments to be passed to the policy on creation
        WARNING: this logging can take a lot of space quickly
    """

    def __init__(self, policy, env, test_env=None, gamma=0.99, timesteps_per_batch=1024, max_kl=0.01, cg_iters=10, kappa=1.0,
                 entcoeff=0.0, cg_damping=1e-2, vf_stepsize=3e-4, vf_iters=3, vf_phi_update_interval=1, verbose=0,
                 _init_setup_model=True, policy_kwargs=None, eval_freq=5, seed=0):
        super(TRPO, self).__init__(policy=policy, env=env, verbose=verbose, requires_vec_env=False,
                                   _init_setup_model=_init_setup_model, policy_kwargs=policy_kwargs)

        self.timesteps_per_batch = timesteps_per_batch
        self.cg_iters = cg_iters
        self.cg_damping = cg_damping
        self.gamma = gamma
        self.kappa = kappa
        self.max_kl = max_kl
        self.vf_iters = vf_iters
        self.vf_phi_update_interval = vf_phi_update_interval
        self.vf_stepsize = vf_stepsize
        self.entcoeff = entcoeff

        # GAIL Params
        self.g_step = 1

        self.graph = None
        self.sess = None
        self.policy_pi = None
        self.loss_names = None
        self.assign_old_eq_new = None
        self.compute_losses = None
        self.compute_lossandgrad = None
        self.compute_fvp = None
        self.compute_vflossandgrad = None
        self.d_adam = None
        self.vfadam = None
        self.get_flat = None
        self.set_from_flat = None
        self.timed = None
        self.allmean = None
        self.nworkers = None
        self.rank = None
        self.reward_giver = None
        self.step = None
        self.proba_step = None
        self.initial_state = None
        self.params = None
        self.summary = None
        self.episode_reward = None
        self.test_env = test_env
        self.eval_freq = eval_freq
        self.eval_reward = None
        self.seed = seed
        self.phi_seed = seed

        if _init_setup_model:
            self.setup_model()

    def _get_pretrain_placeholders(self):
        policy = self.policy_pi
        action_ph = policy.pdtype.sample_placeholder([None])
        if isinstance(self.action_space, gym.spaces.Discrete):
            return policy.obs_ph, action_ph, policy.policy
        return policy.obs_ph, action_ph, policy.deterministic_action

    def setup_model(self):
        # prevent import loops

        with SetVerbosity(self.verbose):

            assert issubclass(self.policy, ActorCriticPolicy), "Error: the input policy for the TRPO model must be " \
                                                               "an instance of common.policies.ActorCriticPolicy."

            self.nworkers = MPI.COMM_WORLD.Get_size()
            print("number of workers are", self.nworkers)
            self.rank = MPI.COMM_WORLD.Get_rank()
            np.set_printoptions(precision=3)

            self.graph = tf.Graph()
            with self.graph.as_default():
                self.sess = tf_util.single_threaded_session(graph=self.graph)
                self._setup_learn(self.seed)

                # Construct network for new policy
                self.policy_pi = self.policy(self.sess, self.observation_space, self.action_space, self.n_envs, 1,
                                             None, reuse=False, **self.policy_kwargs)

                # Network for old policy
                with tf.variable_scope("oldpi", reuse=False):
                    old_policy = self.policy(self.sess, self.observation_space, self.action_space, self.n_envs, 1,
                                             None, reuse=False, **self.policy_kwargs)

                # Network for phi old
                with tf.variable_scope("oldphi", reuse=False):
                    self.policy_phi_old = self.policy(self.sess, self.observation_space, self.action_space, self.n_envs, 1,
                                             None, reuse=False, **self.policy_kwargs)

                with tf.variable_scope("loss", reuse=False):
                    atarg = tf.placeholder(dtype=tf.float32, shape=[None])  # Target advantage function (if applicable)
                    ret = tf.placeholder(dtype=tf.float32, shape=[None])  # Empirical return 

                    observation = self.policy_pi.obs_ph
                    action = self.policy_pi.pdtype.sample_placeholder([None])

                    kloldnew = old_policy.proba_distribution.kl(self.policy_pi.proba_distribution)
                    ent = self.policy_pi.proba_distribution.entropy()
                    meankl = tf.reduce_mean(kloldnew)
                    meanent = tf.reduce_mean(ent)
                    entbonus = self.entcoeff * meanent

                    vferr = tf.reduce_mean(tf.square(self.policy_pi.value_flat - ret))

                    # advantage * pnew / pold
                    ratio = tf.exp(self.policy_pi.proba_distribution.logp(action) -
                                   old_policy.proba_distribution.logp(action))
                    surrgain = tf.reduce_mean(ratio * atarg)

                    optimgain = surrgain + entbonus
                    losses = [optimgain, meankl, entbonus, surrgain, meanent]
                    self.loss_names = ["optimgain", "meankl", "entloss", "surrgain", "entropy"]

                    dist = meankl

                    all_var_list = tf_util.get_trainable_vars("model")
                    var_list = [v for v in all_var_list if "/vf" not in v.name and "/q/" not in v.name]
                    vf_var_list = [v for v in all_var_list if "/pi" not in v.name and "/logstd" not in v.name]
                    all_var_oldpi_list = tf_util.get_trainable_vars("oldpi")
                    var_oldpi_list = [v for v in all_var_oldpi_list if "/vf" not in v.name and "/q/" not in v.name]
                    
                    vf_only_var_list = [v for v in all_var_list if "/pi" not in v.name and "/logstd" not in v.name and "/q" not in v.name]
                    all_var_phi_old_list = tf_util.get_trainable_vars("oldphi")
                    vf_phi_old_var_list = [v for v in all_var_phi_old_list if "/pi" not in v.name and "/logstd" not in v.name and "/q" not in v.name]
                    
                    self.policy_vars = all_var_list
                    self.oldpolicy_vars = all_var_oldpi_list
                    print("pi vars", vf_var_list)
                    print("phi old vars", vf_phi_old_var_list)

                    self.get_flat = tf_util.GetFlat(var_list, sess=self.sess)
                    self.set_from_flat = tf_util.SetFromFlat(var_list, sess=self.sess)

                    klgrads = tf.gradients(dist, var_list)
                    flat_tangent = tf.placeholder(dtype=tf.float32, shape=[None], name="flat_tan")
                    shapes = [var.get_shape().as_list() for var in var_list]
                    start = 0
                    tangents = []
                    for shape in shapes:
                        var_size = tf_util.intprod(shape)
                        tangents.append(tf.reshape(flat_tangent[start: start + var_size], shape))
                        start += var_size
                    gvp = tf.add_n([tf.reduce_sum(grad * tangent)
                                    for (grad, tangent) in zipsame(klgrads, tangents)])  # pylint: disable=E1111
                    fvp = tf_util.flatgrad(gvp, var_list)

                    tf.summary.scalar('entropy_loss', meanent)
                    tf.summary.scalar('policy_gradient_loss', optimgain)
                    tf.summary.scalar('value_function_loss', surrgain)
                    tf.summary.scalar('approximate_kullback-leibler', meankl)
                    tf.summary.scalar('loss', optimgain + meankl + entbonus + surrgain + meanent)

                    self.assign_old_eq_new = \
                        tf_util.function([], [], updates=[tf.assign(oldv, newv) for (oldv, newv) in
                                                          zipsame(tf_util.get_globals_vars("oldpi"),
                                                                  tf_util.get_globals_vars("model"))])
                    self.compute_losses = tf_util.function([observation, old_policy.obs_ph, action, atarg], losses)
                    self.compute_fvp = tf_util.function([flat_tangent, observation, old_policy.obs_ph, action, atarg],
                                                        fvp)
                    self.compute_vflossandgrad = tf_util.function([observation, old_policy.obs_ph, ret],
                                                                  tf_util.flatgrad(vferr, vf_var_list))
                    self.compute_vf_loss = tf_util.function([observation, old_policy.obs_ph, ret], vferr)

                    update_phi_old_expr = []
                    for var, var_target in zip(sorted(vf_only_var_list, key=lambda v: v.name),
                                   sorted(vf_phi_old_var_list, key=lambda v: v.name)):
                        update_phi_old_expr.append(var_target.assign(var))
                    update_phi_old_expr = tf.group(*update_phi_old_expr)

                    self.update_phi_old = tf_util.function([], [], updates=[update_phi_old_expr])

                    @contextmanager
                    def timed(msg):
                        if self.rank == 0 and self.verbose >= 1:
                            print(colorize(msg, color='magenta'))
                            start_time = time.time()
                            yield
                            print(colorize("done in {:.3f} seconds".format((time.time() - start_time)),
                                           color='magenta'))
                        else:
                            yield

                    @contextmanager
                    def temp_seed(seed):
                        state = np.random.get_state()
                        np.random.seed(seed)
                        try:
                            yield
                        finally:
                            np.random.set_state(state)

                    def allmean(arr):
                        assert isinstance(arr, np.ndarray)
                        out = np.empty_like(arr)
                        MPI.COMM_WORLD.Allreduce(arr, out, op=MPI.SUM)
                        out /= self.nworkers
                        return out

                    tf_util.initialize(sess=self.sess)

                    th_init = self.get_flat()
                    MPI.COMM_WORLD.Bcast(th_init, root=0)
                    self.set_from_flat(th_init)

                with tf.variable_scope("Adam_mpi", reuse=False):
                    self.vfadam = MpiAdam(vf_var_list, sess=self.sess)
                    #self.vf_phi_adam = MpiAdam(vf_phi_var_list, sess=self.sess)
                    self.vfadam.sync()
                    #self.vf_phi_adam.sync()

                with tf.variable_scope("input_info", reuse=False):
                    tf.summary.scalar('discounted_rewards', tf.reduce_mean(ret))
                    tf.summary.scalar('learning_rate', tf.reduce_mean(self.vf_stepsize))
                    tf.summary.scalar('advantage', tf.reduce_mean(atarg))
                    tf.summary.scalar('kl_clip_range', tf.reduce_mean(self.max_kl))

                self.timed = timed
                self.allmean = allmean
                self.temp_seed = temp_seed

                self.step = self.policy_pi.step
                self.proba_step = self.policy_pi.proba_step
                self.initial_state = self.policy_pi.initial_state

                self.params = tf_util.get_trainable_vars("model") + tf_util.get_trainable_vars("oldpi")

                self.summary = tf.summary.merge_all()

                self.compute_lossandgrad = \
                    tf_util.function([observation, old_policy.obs_ph, action, atarg, ret],
                                     [self.summary, tf_util.flatgrad(optimgain, var_list)] + losses)

    def learn(self, total_timesteps, callback=None, seed=None, log_interval=100, tb_log_name="TRPO",
              reset_num_timesteps=True):

        new_tb_log = self._init_num_timesteps(reset_num_timesteps)
        print("args", self.kappa, self.vf_phi_update_interval, seed)

        with SetVerbosity(self.verbose):
            #self._setup_learn(seed)

            with self.sess.as_default():
                seg_gen = traj_segment_generator(self.policy_pi, self.policy_phi_old, self.env, self.timesteps_per_batch, self.gamma, self.kappa,
                                                 reward_giver=self.reward_giver)

                episodes_so_far = 0
                timesteps_so_far = 0
                iters_so_far = 0
                t_start = time.time()
                len_buffer = deque(maxlen=40)  # rolling buffer for episode lengths
                reward_buffer = deque(maxlen=40)  # rolling buffer for episode rewards
                vf_loss_buffer = deque(maxlen=80) # rolling buffer for vf loss
                self.episode_reward = np.zeros((self.n_envs,))

                true_reward_buffer = None

                while True:
                    if callback is not None:
                        # Only stop training if return value is False, not when it is None. This is for backwards
                        # compatibility with callbacks that have no return statement.
                        if callback(locals(), globals()) is False:
                            break
                    if total_timesteps and timesteps_so_far >= total_timesteps:
                        break

                    logger.log("********** Iteration %i ************" % iters_so_far)

                    def fisher_vector_product(vec):
                        return self.allmean(self.compute_fvp(vec, *fvpargs, sess=self.sess)) + self.cg_damping * vec

                    # ------------------ Update G ------------------
                    logger.log("Optimizing Policy...")
                    # g_step = 1 when not using GAIL
                    mean_losses = None
                    vpredbefore = None
                    tdlamret = None
                    observation = None
                    action = None
                    seg = None
                    for k in range(self.g_step):
                        with self.timed("sampling"):
                            seg = seg_gen.__next__()
                        add_vtarg_and_adv(seg, self.gamma, self.kappa)
                        # ob, ac, atarg, ret, td1ret = map(np.concatenate, (obs, acs, atargs, rets, td1rets))
                        #iters_val = (2 * int(iters_so_far / self.vf_phi_update_interval) + 1) * self.vf_phi_update_interval * 3 / 4
                        vf_loss = []

                        observation, action = seg["observations"], seg["actions"]
                        atarg, tdlamret = seg["adv"], seg["tdlamret"]

                        vpredbefore = seg["vpred"]  # predicted value function before update
                        atarg = (atarg - atarg.mean()) / atarg.std()  # standardized advantage function estimate

                        args = seg["observations"], seg["observations"], seg["actions"], atarg
                        # Subsampling: see p40-42 of John Schulman thesis
                        # http://joschu.net/docs/thesis.pdf
                        fvpargs = [arr[::5] for arr in args]

                        self.assign_old_eq_new(sess=self.sess)

                        with self.timed("computegrad"):
                            steps = self.num_timesteps + (k + 1) * (seg["total_timestep"] / self.g_step)
                            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                            run_metadata = None
                            # run loss backprop with summary, and save the metadata (memory, compute time, ...)
                            _, grad, *lossbefore = self.compute_lossandgrad(*args, tdlamret, sess=self.sess,
                                                                                options=run_options,
                                                                                run_metadata=run_metadata)

                        lossbefore = self.allmean(np.array(lossbefore))
                        grad = self.allmean(grad)
                        if np.allclose(grad, 0):
                            logger.log("Got zero gradient. not updating")
                        else:
                            with self.timed("conjugate_gradient"):
                                stepdir = conjugate_gradient(fisher_vector_product, grad, cg_iters=self.cg_iters,
                                                            verbose=self.rank == 0 and self.verbose >= 1)
                            assert np.isfinite(stepdir).all()
                            shs = .5 * stepdir.dot(fisher_vector_product(stepdir))
                            # abs(shs) to avoid taking square root of negative values
                            lagrange_multiplier = np.sqrt(abs(shs) / self.max_kl)
                            # logger.log("lagrange multiplier:", lm, "gnorm:", np.linalg.norm(g))
                            fullstep = stepdir / lagrange_multiplier
                            expectedimprove = grad.dot(fullstep)
                            surrbefore = lossbefore[0]
                            stepsize = 1.0
                            thbefore = self.get_flat()
                            thnew = None
                            for _ in range(10):
                                thnew = thbefore + fullstep * stepsize
                                self.set_from_flat(thnew)
                                mean_losses = surr, kl_loss, *_ = self.allmean(
                                    np.array(self.compute_losses(*args, sess=self.sess)))
                                improve = surr - surrbefore
                                logger.log("Expected: %.3f Actual: %.3f" % (expectedimprove, improve))
                                if not np.isfinite(mean_losses).all():
                                    logger.log("Got non-finite value of losses -- bad!")
                                elif kl_loss > self.max_kl * 1.5:
                                    logger.log("violated KL constraint. shrinking step.")
                                elif improve < 0:
                                    logger.log("surrogate didn't improve. shrinking step.")
                                else:
                                    logger.log("Stepsize OK!")
                                    break
                                stepsize *= .5
                            else:
                                logger.log("couldn't compute a good step")
                                self.set_from_flat(thbefore)
                            if self.nworkers > 1 and iters_so_far % 20 == 0:
                                # list of tuples
                                paramsums = MPI.COMM_WORLD.allgather((thnew.sum(), self.vfadam.getflat().sum()))
                                assert all(np.allclose(ps, paramsums[0]) for ps in paramsums[1:])

                        with self.timed("vf"):
                            vf_loss = []
                            for _ in range(self.vf_iters):
                                # NOTE: for recurrent policies, use shuffle=False?
                                for (mbob, mbret) in dataset.iterbatches((seg["observations"], seg["tdlamret"]),
                                                                        include_final_partial_batch=False,
                                                                        batch_size=128,
                                                                        shuffle=True):
                                    grad = self.allmean(self.compute_vflossandgrad(mbob, mbob, mbret, sess=self.sess))
                                    self.vfadam.update(grad, self.vf_stepsize)
                                    vf_loss.append(self.compute_vf_loss(mbob, mbob, mbret, sess=self.sess))
                        
                        if iters_so_far % self.vf_phi_update_interval == 0:
                            with self.timed("vf_phi_old_update"):
                                self.update_phi_old(sess=self.sess)
                        
                        if iters_so_far % self.eval_freq == 0:
                            if self.test_env is not None:
                                self.eval_reward = self.evaluate_agent(self.test_env, self.eval_freq)

                    if mean_losses is None:
                        mean_losses = [None, None, None, None, None]
                    for (loss_name, loss_val) in zip(self.loss_names, mean_losses):
                        logger.record_tabular(loss_name, loss_val)

                    if vpredbefore is not None:
                        logger.record_tabular("explained_variance_tdlam_before",
                                            explained_variance(vpredbefore, tdlamret))
                    else:
                        logger.record_tabular("explained_variance_tdlam_before", None)

                    # lr: lengths and rewards
                    lr_local = (seg["ep_lens"], seg["ep_rets"])  # local values
                    list_lr_pairs = MPI.COMM_WORLD.allgather(lr_local)  # list of tuples
                    lens, rews = map(flatten_lists, zip(*list_lr_pairs))
                    len_buffer.extend(lens)
                    reward_buffer.extend(rews)
                    vf_loss_buffer.extend(vf_loss)

                    if len(len_buffer) > 0:
                        logger.record_tabular("EpLenMean", np.mean(len_buffer))
                        logger.record_tabular("EpRewMean", np.mean(reward_buffer))
                        logger.record_tabular("VfLossThisIter", np.mean(vf_loss))
                    #    logger.record_tabular("VfPhiLossThisIter", np.mean(vf_phi_loss))
                    logger.record_tabular("EpThisIter", len(lens))
                    episodes_so_far += len(lens)
                    current_it_timesteps = MPI.COMM_WORLD.allreduce(seg["total_timestep"])
                    timesteps_so_far += current_it_timesteps
                    self.num_timesteps += current_it_timesteps
                    iters_so_far += 1

                    logger.record_tabular("EpisodesSoFar", episodes_so_far)
                    logger.record_tabular("TimestepsSoFar", self.num_timesteps)
                    logger.record_tabular("TimeElapsed", time.time() - t_start)
                    logger.record_tabular("EvaluationScore", self.eval_reward)

                    if self.verbose >= 1 and self.rank == 0:
                        logger.dump_tabular()

        return self

    def evaluate_agent(self, test_env, test_episodes):

        step = 0
        games = 0
        total_rewards = 0.0
        reset = test_env.reset()

        while (games < test_episodes):
            terminal = False
            state = reset
            while not terminal:
                action, _, _, _ = self.policy_pi.step(state.reshape(-1, *state.shape), deterministic=True)
                state, reward, terminal, info = test_env.step(action[0])

                total_rewards += reward

                step += 1

            games += 1
            reset = test_env.reset()

        return total_rewards / games

    def save(self, save_path):
        data = {
            "gamma": self.gamma,
            "kappa": self.kappa,
            "timesteps_per_batch": self.timesteps_per_batch,
            "max_kl": self.max_kl,
            "cg_iters": self.cg_iters,
            "entcoeff": self.entcoeff,
            "cg_damping": self.cg_damping,
            "vf_stepsize": self.vf_stepsize,
            "vf_iters": self.vf_iters,
            "g_step": self.g_step,
            "verbose": self.verbose,
            "policy": self.policy,
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "n_envs": self.n_envs,
            "_vectorize_action": self._vectorize_action,
            "policy_kwargs": self.policy_kwargs,
            "eval_freq": self.eval_freq,
            "vf_phi_update_interval": self.vf_phi_update_interval
        }

        params_to_save = self.get_parameters()

        self._save_to_file(save_path, data=data, params=params_to_save)
