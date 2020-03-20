import sys
import re
import multiprocessing
import os.path as osp
import gym
from collections import defaultdict
import tensorflow as tf
import numpy as np

from baselines.common.vec_env import VecFrameStack, VecNormalize, VecEnv, DummyVecEnv
from baselines.common.vec_env.vec_video_recorder import VecVideoRecorder
from baselines.common.cmd_util import common_arg_parser, parse_unknown_args, make_vec_env, make_env
from baselines.common.tf_util import get_session
from baselines import logger
from importlib import import_module
from contextlib import ExitStack
import time

try:
    from mpi4py import MPI
except ImportError:
    MPI = None

try:
    import pybullet_envs
except ImportError:
    pybullet_envs = None

try:
    import roboschool
except ImportError:
    roboschool = None

_game_envs = defaultdict(set)
for env in gym.envs.registry.all():
    # TODO: solve this with regexes
    env_type = env.entry_point.split(':')[0].split('.')[-1]
    _game_envs[env_type].add(env.id)

# reading benchmark names directly from retro requires
# importing retro here, and for some reason that crashes tensorflow
# in ubuntu
_game_envs['retro'] = {
    'BubbleBobble-Nes',
    'SuperMarioBros-Nes',
    'TwinBee3PokoPokoDaimaou-Nes',
    'SpaceHarrier-Nes',
    'SonicTheHedgehog-Genesis',
    'Vectorman-Genesis',
    'FinalFight-Snes',
    'SpaceInvaders-Snes',
}


def train(args, extra_args, env_kwargs=None):
    env_type, env_id = get_env_type(args)
    print('env_type: {}'.format(env_type))

    total_timesteps = int(args.num_timesteps)
    seed = args.seed

    learn = get_learn_function(args.alg)
    alg_kwargs = get_learn_function_defaults(args.alg, env_type)
    alg_kwargs.update(extra_args)

    env = build_env(args, env_kwargs=env_kwargs)
    if args.save_video_interval != 0:
        env = VecVideoRecorder(env, osp.join(logger.get_dir(), "videos"), record_video_trigger=lambda x: x % args.save_video_interval == 0, video_length=args.save_video_length)

    if args.network:
        alg_kwargs['network'] = args.network
    else:
        if alg_kwargs.get('network') is None:
            alg_kwargs['network'] = get_default_network(env_type)
            
    if args.init_logstd:
        alg_kwargs['init_logstd'] = args.init_logstd

    print('Training {} on {}:{} with arguments \n{}'.format(args.alg, env_type, env_id, alg_kwargs))

    model = learn(
        env=env,
        seed=seed,
        total_timesteps=total_timesteps,
        **alg_kwargs
    )

    return model, env


def build_env(args, env_kwargs):
    ncpu = multiprocessing.cpu_count()
    if sys.platform == 'darwin': ncpu //= 2
    nenv = args.num_env or ncpu
    alg = args.alg
    seed = args.seed

    env_type, env_id = get_env_type(args)

    if env_type in {'atari', 'retro'}:
        if alg == 'deepq':
            env = make_env(env_id, env_type, seed=seed, env_kwargs=env_kwargs, wrapper_kwargs={'frame_stack': True})
        elif alg == 'trpo_mpi':
            env = make_env(env_id, env_type, seed=seed, env_kwargs=env_kwargs)
        else:
            frame_stack_size = 4
            env = make_vec_env(env_id, env_type, nenv, seed, env_kwargs=env_kwargs, gamestate=args.gamestate, reward_scale=args.reward_scale)
            env = VecFrameStack(env, frame_stack_size)

    else:
        config = tf.ConfigProto(allow_soft_placement=True,
                               intra_op_parallelism_threads=1,
                               inter_op_parallelism_threads=1)
        config.gpu_options.allow_growth = True
        get_session(config=config)

        flatten_dict_observations = alg not in {'her'}
        env = make_vec_env(env_id, env_type, args.num_env or 1, seed, env_kwargs=env_kwargs, reward_scale=args.reward_scale, flatten_dict_observations=flatten_dict_observations)

        if env_type == 'mujoco':
            env = VecNormalize(env, use_tf=True)

    return env


def get_env_type(args):
    env_id = args.env

    if args.env_type is not None:
        return args.env_type, env_id

    # Re-parse the gym registry, since we could have new envs since last time.
    for env in gym.envs.registry.all():
        env_type = env.entry_point.split(':')[0].split('.')[-1]
        _game_envs[env_type].add(env.id)  # This is a set so add is idempotent

    if env_id in _game_envs.keys():
        env_type = env_id
        env_id = [g for g in _game_envs[env_type]][0]
    else:
        env_type = None
        for g, e in _game_envs.items():
            if env_id in e:
                env_type = g
                break
        if ':' in env_id:
            env_type = re.sub(r':.*', '', env_id)
        assert env_type is not None, 'env_id {} is not recognized in env types'.format(env_id, _game_envs.keys())

    return env_type, env_id


def get_default_network(env_type):
    if env_type in {'atari', 'retro'}:
        return 'cnn'
    else:
        return 'mlp'


def get_alg_module(alg, submodule=None):
    submodule = submodule or alg
    try:
        # first try to import the alg module from baselines
        alg_module = import_module('.'.join(['baselines', alg, submodule]))
    except ImportError:
        # then from rl_algs
        alg_module = import_module('.'.join(['rl_' + 'algs', alg, submodule]))

    return alg_module


def get_learn_function(alg):
    return get_alg_module(alg).learn


def get_learn_function_defaults(alg, env_type):
    try:
        alg_defaults = get_alg_module(alg, 'defaults')
        kwargs = getattr(alg_defaults, env_type)()
    except (ImportError, AttributeError):
        kwargs = {}
    return kwargs


def parse_cmdline_kwargs(args):
    '''
    convert a list of '='-spaced command-line arguments to a dictionary, evaluating python objects when possible
    '''
    def parse(v):

        assert isinstance(v, str)
        try:
            return eval(v)
        except (NameError, SyntaxError):
            return v

    return {k: parse(v) for k, v in parse_unknown_args(args).items()}


def configure_logger(log_path, **kwargs):
    if log_path is not None:
        logger.configure(log_path)
    else:
        logger.configure(**kwargs)


def set_train_noise(model, log_value):
    try:
        with tf.variable_scope('ppo2_model', reuse=True):
            logstd = tf.get_variable(name='pi/logstd')
            ass = tf.assign(logstd, np.full(logstd.shape, log_value))
            model.sess.run(ass)
    except:
        pass

def remove_train_noise(model):
    # set_train_noise(model, -np.inf)
    if hasattr(model, 'act_model'):
        model.act_model.pd.remove_noise = True
        model.train_model.pd.remove_noise = True
        model.act_model.action = model.act_model.pd.sample()
        model.train_model.action = model.train_model.pd.sample()

def main(args):
    # configure logger, disable logging in child MPI processes (with rank > 0)

    arg_parser = common_arg_parser()
    args, unknown_args = arg_parser.parse_known_args(args)
    extra_args = parse_cmdline_kwargs(unknown_args)

    if MPI is None or MPI.COMM_WORLD.Get_rank() == 0:
        rank = 0
        configure_logger(args.log_path)
    else:
        rank = MPI.COMM_WORLD.Get_rank()
        configure_logger(args.log_path, format_strs=[])

    model, env = train(args, extra_args)

    if args.save_path is not None and rank == 0:
        save_path = osp.expanduser(args.save_path)
        model.save(save_path)

    if args.play_episodes > 0:
        logger.log("Running trained model")
        start_time = time.time()
        remove_train_noise(model)

        with ExitStack() as stack:  # handling orderly resource closure

            f = None
            if args.policy_path is not None:
               f = stack.enter_context(open(args.policy_path, 'w+'))

            obs = env.reset()

            state = model.initial_state if hasattr(model, 'initial_state') else None
            dones = np.zeros((1,))

            episode_rew = 0.       # batch average reward
            mean_rew = 0.          # average reward
            mean_sq_batch_avg = 0. # second moment of batch averages
            mean_batch_var = 0.    # mean of batch variances

            episode_counter = 0
            cycle_length = 0
            template_message = 'episodes {} to {} out of {}: mean reward={}'
            while episode_counter < args.play_episodes:
                if state is not None:
                    actions, _, state, _ = model.step(obs, S=state, M=dones)
                else:
                    actions, _, _, _ = model.step(obs)

                obs, rew, done, _ = env.step(actions)
                if args.policy_path is not None:
                   obs_str = ['%f,' % o for o in obs[0]] + ['%.6f,' % a for a in actions[0]]
                   line = ','.join(obs_str)
                   f.write(line + '\n')
                episode_rew += rew if isinstance(env, VecEnv) else rew
                done = done.any() if isinstance(done, np.ndarray) else done
                if done:
                    episode_counter += rew.size
                    cycle_length += 1
                    starting_episode = episode_counter < args.print_episodes
                    periodic_episode = args.print_period > 0 and cycle_length % args.print_period == 0
                    batch_mean = np.mean(episode_rew)
                    batch_var = np.var(episode_rew) if isinstance(env, VecEnv) else 0.
                    if starting_episode or periodic_episode:
                       print(template_message.format(episode_counter+1-rew.shape[0], episode_counter+1, args.play_episodes, batch_mean))
                       print('_________________________________________________________________')
                    mean_rew += batch_mean
                    mean_batch_var += batch_var
                    mean_sq_batch_avg += batch_mean * batch_mean
                    episode_rew = 0.
                    obs = env.reset()
                elif rew.shape[0] == 1 and episode_counter < args.print_episodes:
                    env.render()

            mean_rew = mean_rew / cycle_length
            mean_batch_var = mean_batch_var / cycle_length
            mean_sq_batch_avg = mean_sq_batch_avg / cycle_length

        print('\n' * int(args.print_episodes > 0))

        if isinstance(env, DummyVecEnv):
            print('*** TH price = ', env.theoretical_price())
        if args.play_episodes > 0:
            total_var_rew = mean_sq_batch_avg - mean_rew * mean_rew + mean_batch_var  # variance of means plus mean of variances
            print('*** MC price = ', mean_rew)
            print('*** MC error = ', np.sqrt(total_var_rew / args.play_episodes))
            print('*** MC wall time = %.2f seconds' % (time.time() - start_time))

    env.close()

    return model


if __name__ == '__main__':
    main(sys.argv)
