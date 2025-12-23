"""Training script for Recall2Imagine (R2I) on BoxMovingEnv.

This script provides the main entry point for training the goal-conditioned R2I agent
on the BoxMovingEnv. It follows the structure of the original R2I training script
but adapts it for the box moving environment.

Usage:
    python train_r2i.py --configs box_moving
    python train_r2i.py --configs box_moving small --logdir ./runs/experiment1
    python train_r2i.py --configs box_moving_debug  # For quick testing
"""

import importlib
import logging
import pathlib
import sys
import warnings
from functools import partial as bind
import traceback

# Setup paths FIRST before any other imports
# This is critical for Docker and local environments to work correctly
directory = pathlib.Path(__file__).resolve().parent
project_root = directory.parent
r2i_path = project_root / 'recall2imagine'

# Add project root so recall2imagine can be imported as a package
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
# Add src directory for local imports
if str(directory) not in sys.path:
    sys.path.insert(0, str(directory))
# Add recall2imagine for embodied and direct imports
if str(r2i_path) not in sys.path:
    sys.path.insert(0, str(r2i_path))

warnings.filterwarnings('ignore', '.*box bound precision lowered.*')
warnings.filterwarnings('ignore', '.*using stateful random seeds*')
warnings.filterwarnings('ignore', '.*is a deprecated alias for.*')
warnings.filterwarnings('ignore', '.*truncated to dtype int32.*')

# Allow host-to-device transfers for environment operations
# The jaxagent will set this to 'disallow' later for training, but we need
# it to be permissive during environment initialization
import jax
jax.config.update('jax_transfer_guard', 'allow')
import jax.numpy as jnp

import numpy as np

import embodied
from embodied import wrappers

from envs.embodied_wrapper import EmbodiedBoxMovingWrapper, make_box_moving_env
from envs.block_moving.env_types import BoxMovingConfig
from impls.utils.datasets import Dataset, GCDataset
from impls.utils.log_utils import CsvLogger


logger = logging.getLogger(__name__)


class CsvOutput:
    """Embodied logger output that mirrors scalar metrics into a CSV file."""

    def __init__(self, logdir, filename='metrics.csv'):
        self._path = pathlib.Path(logdir) / filename
        self._writer = CsvLogger(str(self._path))

    def __call__(self, summaries):
        if not summaries:
            return
        step = max(step for step, _, _ in summaries)
        scalars = {}
        for _, name, value in summaries:
            if len(value.shape) == 0:
                scalars[name] = float(value)
        if scalars:
            self._writer.log(scalars, step=step)

    def close(self):
        self._writer.close()


DEFAULT_HER_CONFIG = {
    'enabled': True,
    'discount': 0.99,
    'value_p_curgoal': 0.1,
    'value_p_trajgoal': 0.8,
    'value_p_randomgoal': 0.1,
    'value_geom_sample': True,
    'actor_p_curgoal': 0.1,
    'actor_p_trajgoal': 0.8,
    'actor_p_randomgoal': 0.1,
    'actor_geom_sample': True,
    'gc_negative': False,
    'p_aug': None,
    'frame_stack': None,
}


def _extract_config_dict(config, key):
    try:
        nested = getattr(config, key)
    except AttributeError:
        return {}
    return dict(nested)


def _resolve_her_config(config):
    cfg = DEFAULT_HER_CONFIG.copy()
    cfg.update(_extract_config_dict(config, 'her'))
    cfg['enabled'] = bool(cfg.get('enabled', True))
    return cfg


def _to_numpy(value):
    if isinstance(value, np.ndarray):
        return value
    return np.asarray(jax.device_get(value))


def _restore_array(original, new_value):
    dtype = getattr(original, 'dtype', new_value.dtype)
    array = new_value.astype(dtype)
    if hasattr(original, 'device'):
        return jax.device_put(array, original.device())
    if hasattr(original, 'devices'):
        return jax.device_put(array)
    return array


def _compute_terminal_mask(is_first):
    """Return a flat mask that marks the last transition of every trajectory."""
    batch, horizon = is_first.shape
    terminals = np.zeros((batch, horizon), dtype=np.float32)
    if horizon > 1:
        # When the next step starts a new episode, the current one is terminal.
        terminals[:, :-1] = np.where(is_first[:, 1:], 1.0, 0.0)
    # Always treat the final collected step in the sequence as terminal.
    terminals[:, -1] = 1.0
    return terminals.reshape(-1)


def _apply_her_to_batch(batch, her_cfg):
    if not her_cfg.get('enabled', True):
        return batch, {}
    required_keys = ('grid', 'goal', 'is_first')
    if any(key not in batch for key in required_keys):
        return batch, {}

    grid_np = _to_numpy(batch['grid'])
    goal_np = _to_numpy(batch['goal'])
    is_first_np = _to_numpy(batch['is_first']).astype(bool)

    batch_size, horizon = grid_np.shape[:2]
    obs_tree = {'grid': grid_np.reshape(batch_size * horizon, -1)}
    if 'grid_2d' in batch:
        grid2d = _to_numpy(batch['grid_2d'])
        obs_tree['grid_2d'] = grid2d.reshape(batch_size * horizon, *grid2d.shape[2:])

    terminals = _compute_terminal_mask(is_first_np.reshape(batch_size, horizon))
    num_transitions = obs_tree['grid'].shape[0]
    dataset = Dataset.create(
        freeze=False,
        observations=obs_tree['grid'],
        terminals=terminals,
        valids=1.0 - terminals,
    )
    gc_dataset = GCDataset(dataset=dataset, config=her_cfg, preprocess_frame_stack=False)
    idxs = np.arange(num_transitions)
    her_batch = gc_dataset.sample(num_transitions, idxs=idxs, evaluation=True)

    new_goal_flat = her_batch['value_goals']
    new_goal = new_goal_flat.reshape(goal_np.shape)
    diff = np.abs(new_goal - goal_np)
    stats = {
        'her/relabel_fraction': float((diff > 1e-6).mean()) if diff.size else 0.0,
    }
    batch['goal'] = _restore_array(batch['goal'], new_goal)

    if 'goal_2d' in batch:
        goal2d = new_goal_flat.reshape(_to_numpy(batch['goal_2d']).shape)
        batch['goal_2d'] = _restore_array(batch['goal_2d'], goal2d)

    return batch, stats


def _wrap_agent_dataset_with_her(agent, her_cfg):
    if not her_cfg.get('enabled', True):
        return

    def postprocess(batch):
        batch, _ = _apply_her_to_batch(batch, her_cfg)
        return agent._convert_inps(batch, agent.train_devices)

    def dataset(source, shared_memory=True):
        if shared_memory:
            batcher = embodied.BatcherSM(
                replay=source,
                workers=agent.data_loaders,
                batch_size=agent.batch_size,
                batch_sequence_len=agent.batch_length,
                postprocess=postprocess,
                prefetch_source=4,
                prefetch_batch=agent.num_buffers,
            )
        else:
            batcher = embodied.Batcher(
                sources=[source] * agent.batch_size,
                workers=agent.data_loaders,
                postprocess=postprocess,
                prefetch_source=4,
                prefetch_batch=1,
            )
        return batcher()

    agent.dataset = dataset


def main(argv=None):
    """Main training function."""
    from impls.agents.r2i import GCR2IAgent
    
    # Parse command line arguments
    parsed, other = embodied.Flags(configs=['defaults']).parse_known(argv)
    config = embodied.Config(GCR2IAgent.configs['defaults'])
    for name in parsed.configs:
        if name in GCR2IAgent.configs:
            config = config.update(GCR2IAgent.configs[name])
        else:
            print(f"Warning: Config '{name}' not found, skipping")
    config = embodied.Flags(config).parse(other)
    her_config = _resolve_her_config(config)
    np.random.seed(config.seed)
    
    args = embodied.Config(
        **config.run,
        logdir=config.logdir,
        replay_dir=config.replay_dir,
        checkpoint_dir=config.checkpoint_dir,
        batch_steps=config.batch_size * config.batch_length,
    )
    print(config)
    
    # Setup directories
    logdir = embodied.Path(args.logdir)
    logdir.mkdirs()
    if args.replay_dir == 'none':
        replay_dir = logdir / 'replay'
    else:
        replay_dir = embodied.Path(args.replay_dir)
    replay_dir.mkdirs()
    config.save(logdir / 'config.yaml')
    
    # Setup step counter and logger
    step = embodied.Counter()
    logger = make_logger(parsed, logdir, step, config)
    
    cleanup = []
    try:
        if args.script == 'train':
            replay = make_replay(config, replay_dir)
            env = make_envs(config)
            cleanup.append(env)
            agent = GCR2IAgent(env.obs_space, env.act_space, step, config)
            _wrap_agent_dataset_with_her(agent, her_config)
            replay.set_agent(agent)
            embodied.run.train(agent, env, replay, logger, args, config)
        
        elif args.script == 'train_eval':
            replay = make_replay(config, logdir / 'replay')
            eval_replay = make_replay(config, logdir / 'eval_replay', is_eval=True)
            env = make_envs(config)
            eval_env = make_envs(config)
            cleanup += [env, eval_env]
            agent = GCR2IAgent(env.obs_space, env.act_space, step, config)
            _wrap_agent_dataset_with_her(agent, her_config)
            embodied.run.train_eval(
                agent, env, eval_env, replay, eval_replay, logger, args)
        
        elif args.script == 'eval_only':
            env = make_envs(config)
            cleanup.append(env)
            agent = GCR2IAgent(env.obs_space, env.act_space, step, config)
            embodied.run.eval_only(agent, env, logger, args)
        
        else:
            raise NotImplementedError(f"Unknown script: {args.script}")
    
    finally:
        for obj in cleanup:
            obj.close()


def make_logger(parsed, logdir, step, config):
    """Create the logger with appropriate outputs."""
    multiplier = 1  # No frame repeat for box moving
    logger = embodied.Logger(step, [
        embodied.logger.TerminalOutput(config.filter),
        embodied.logger.JSONLOutput(logdir, 'metrics.jsonl'),
        embodied.logger.JSONLOutput(logdir, 'scores.jsonl', 'episode/score'),
        embodied.logger.WandBOutput('.*', str(logdir), config),
        CsvOutput(logdir),
    ], multiplier)
    return logger


def make_replay(config, directory=None, is_eval=False, rate_limit=False, **kwargs):
    """Create the replay buffer."""
    assert config.replay in ['uniform', 'lfs'] or not rate_limit
    length = config.batch_length
    size = config.replay_size // 10 if is_eval else config.replay_size
    
    if config.replay == 'uniform' or is_eval:
        kw = {'online': config.replay_online}
        if rate_limit and config.run.train_ratio > 0:
            kw['samples_per_insert'] = config.run.train_ratio / config.batch_length
            kw['tolerance'] = 10 * config.batch_size
            kw['min_size'] = config.batch_size
        replay = embodied.replay.Uniform(length, size, directory, **kw)
    elif config.replay == 'lfs':
        kw = {}
        if rate_limit and config.run.train_ratio > 0:
            kw['samples_per_insert'] = config.run.train_ratio / config.batch_length
            kw['tolerance'] = 10 * config.batch_size
            kw['min_size'] = max(config.batch_size, config.envs.amount) * 2
        kw['batch_size'] = config.batch_size
        kw['num_buffers'] = config.num_buffers
        kw['lfs_directory'] = config.lfs_dir
        kw['use_lfs'] = config.use_lfs
        kw['unlocked_sampling'] = config.unlocked_sampling
        replay = embodied.replay.FIFO_LFS(directory, length, size, **kw)
    else:
        raise NotImplementedError(config.replay)
    
    return replay


def make_envs(config, **overrides):
    """Create batched environments."""
    ctors = []
    for index in range(config.envs.amount):
        ctor = lambda: make_env(config, **overrides)
        if config.envs.parallel != 'none':
            ctor = bind(embodied.Parallel, ctor, config.envs.parallel)
        if config.envs.restart:
            ctor = bind(wrappers.RestartOnException, ctor)
        ctors.append(ctor)
    envs = [ctor() for ctor in ctors]
    return embodied.BatchEnv(envs, parallel=(config.envs.parallel != 'none'))


def make_env(config, **overrides):
    """Create a single BoxMovingEnv instance wrapped for R2I.
    
    This function handles the task routing and creates the appropriate
    environment based on the config.task setting.
    """
    task = config.task
    
    if task == 'box_moving':
        # Create BoxMovingEnv with embodied wrapper
        env = make_box_moving_env(config, **overrides)
        return wrap_env(env, config)
    else:
        # Fall back to standard R2I environment creation for other tasks
        # This allows using the same script for other environments if needed
        return _make_standard_env(config, task, **overrides)


def _make_standard_env(config, task, **overrides):
    """Create a standard R2I environment (fallback for non-box_moving tasks)."""
    import ruamel.yaml as yaml
    
    suite, task_name = task.split('_', 1)
    ctor = {
        'dummy': 'embodied.envs.dummy:Dummy',
        'gym': 'embodied.envs.from_gym:FromGym',
        'dm': 'embodied.envs.from_dm:FromDM',
        'crafter': 'embodied.envs.crafter:Crafter',
        'dmc': 'embodied.envs.dmc:DMC',
        'atari': 'embodied.envs.atari:Atari',
    }.get(suite)
    
    if ctor is None:
        raise ValueError(f"Unknown environment suite: {suite}")
    
    if isinstance(ctor, str):
        module, cls = ctor.split(':')
        module = importlib.import_module(module)
        ctor = getattr(module, cls)
    
    if config.env.kwargs != '{}':
        kwargs = yaml.YAML(typ='safe').load(config.env.kwargs)
    else:
        kwargs = config.env.get(suite, {})
    kwargs.update(overrides)
    
    env = ctor(task_name, **kwargs)
    return wrap_env(env, config)


def wrap_env(env, config):
    """Apply standard wrappers to the environment."""
    args = config.wrapper
    
    for name, space in env.act_space.items():
        if name == 'reset':
            continue
        elif space.discrete:
            env = wrappers.OneHotAction(env, name)
        elif args.discretize:
            env = wrappers.DiscretizeAction(env, name, args.discretize)
        else:
            env = wrappers.NormalizeAction(env, name)
    
    env = wrappers.ExpandScalars(env)
    
    if args.length:
        env = wrappers.TimeLimit(env, args.length, args.reset)
    
    if args.checks:
        env = wrappers.CheckSpaces(env)
    
    for name, space in env.act_space.items():
        if not space.discrete:
            env = wrappers.ClipAction(env, name)
    
    env = wrappers.CountSteps(env)
    
    return env


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        traceback.print_exc()
        print('Training failed with error:', e)
        sys.exit(1)

