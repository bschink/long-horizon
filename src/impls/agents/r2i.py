"""Goal-conditioned Recall2Imagine (R2I) agent for BoxMovingEnv.

This module provides a goal-conditioned variant of the R2I agent that:
- Handles separate 'grid' and 'goal' observation keys
- Supports `reconstruct_only_obs` to skip goal reconstruction in the decoder loss
- Can be used with both MLP and CNN encoders
"""

import sys
import os
from pathlib import Path

# Add recall2imagine's PARENT directory to path so it can be imported as a package
# This allows relative imports within recall2imagine to work correctly
_r2i_parent = Path(__file__).resolve().parent.parent.parent.parent
_r2i_path = _r2i_parent / 'recall2imagine'
if _r2i_path.exists():
    if str(_r2i_parent) not in sys.path:
        sys.path.insert(0, str(_r2i_parent))
    # Also add recall2imagine itself for embodied imports
    if str(_r2i_path) not in sys.path:
        sys.path.insert(0, str(_r2i_path))

import jax
import jax.numpy as jnp
import ruamel.yaml as yaml

# Import embodied (it's at recall2imagine/embodied)
import embodied

# Import recall2imagine modules - they use relative imports so we import the package
from recall2imagine import behaviors
from recall2imagine import jaxagent
from recall2imagine import jaxutils
from recall2imagine import nets
from recall2imagine import ssm_nets
from recall2imagine import ninjax as nj

tree_map = jax.tree_util.tree_map
sg = lambda x: tree_map(jax.lax.stop_gradient, x)


# Load configs from src/configs.yaml
_CONFIGS_PATH = Path(__file__).parent.parent.parent / 'configs.yaml'


def load_configs():
    """Load R2I configs from src/configs.yaml."""
    if _CONFIGS_PATH.exists():
        return yaml.YAML(typ='safe').load(_CONFIGS_PATH.read_text())
    else:
        # Fallback to recall2imagine configs
        r2i_path = Path(__file__).parent.parent.parent.parent / 'recall2imagine' / 'configs.yaml'
        return yaml.YAML(typ='safe').load(r2i_path.read_text())


@jaxagent.Wrapper
class GCR2IAgent(nj.Module):
    """Goal-conditioned R2I Agent.
    
    This agent extends the standard R2I agent to handle goal-conditioned
    observations with separate 'grid' and 'goal' keys.
    """
    
    configs = load_configs()
    
    def __init__(self, obs_space, act_space, step, config):
        self.config = config
        self.obs_space = obs_space
        self.act_space = act_space['action']
        self.step = step
        self.wm = GCWorldModel(obs_space, act_space, config, name='wm')
        self.task_behavior = getattr(behaviors, config.task_behavior)(
            self.wm, self.act_space, self.config, name='task_behavior')
        if config.expl_behavior == 'None':
            self.expl_behavior = self.task_behavior
        else:
            self.expl_behavior = getattr(behaviors, config.expl_behavior)(
                self.wm, self.act_space, self.config, name='expl_behavior')
    
    def policy_initial(self, batch_size):
        return (
            self.wm.initial(batch_size),
            self.task_behavior.initial(batch_size),
            self.expl_behavior.initial(batch_size))
    
    def train_initial(self, batch_size):
        return self.wm.initial(batch_size)
    
    def policy(self, obs, state, mode='train'):
        self.config.jax.jit and print('Tracing policy function.')
        obs = self.preprocess(obs)
        (prev_latent, prev_action), task_state, expl_state = state
        embed = self.wm.encoder(obs)
        latent, _ = self.wm.rssm.obs_step(
            prev_latent, prev_action, embed, obs['is_first'])
        self.expl_behavior.policy(latent, expl_state)
        task_outs, task_state = self.task_behavior.policy(latent, task_state)
        expl_outs, expl_state = self.expl_behavior.policy(latent, expl_state)
        if mode == 'eval':
            outs = task_outs
            outs['action'] = outs['action'].sample(seed=nj.rng())
            outs['log_entropy'] = jnp.zeros(outs['action'].shape[:1])
        elif mode == 'explore':
            outs = expl_outs
            outs['log_entropy'] = outs['action'].entropy()
            outs['action'] = outs['action'].sample(seed=nj.rng())
        elif mode == 'train':
            outs = task_outs
            outs['log_entropy'] = outs['action'].entropy()
            outs['action'] = outs['action'].sample(seed=nj.rng())
        state = ((latent, outs['action']), task_state, expl_state)
        return outs, state
    
    def train(self, data, state):
        self.config.jax.jit and print('Tracing train function.')
        metrics = {}
        data = self.preprocess(data)
        state, wm_outs, mets = self.wm.train(data, state)
        metrics.update(mets)
        context = {**data, **wm_outs['post']}
        start = tree_map(lambda x: x.reshape([-1] + list(x.shape[2:])), context)
        _, mets = self.task_behavior.train(self.wm.imagine, start, context)
        metrics.update(mets)
        if self.config.expl_behavior != 'None':
            _, mets = self.expl_behavior.train(self.wm.imagine, start, context)
            metrics.update({'expl_' + key: value for key, value in mets.items()})
        outs = {}
        return outs, state, metrics
    
    def report(self, data):
        self.config.jax.jit and print('Tracing report function.')
        data = self.preprocess(data)
        report = {}
        report.update(self.wm.report(data))
        mets = self.task_behavior.report(data)
        report.update({f'task_{k}': v for k, v in mets.items()})
        if self.expl_behavior is not self.task_behavior:
            mets = self.expl_behavior.report(data)
            report.update({f'expl_{k}': v for k, v in mets.items()})
        return report
    
    def preprocess(self, obs):
        obs = obs.copy()
        for key, value in obs.items():
            if key.startswith('log_') or key in ('key',):
                continue
            if len(value.shape) > 3 and value.dtype == jnp.uint8:
                value = jaxutils.cast_to_compute(value) / 255.0
            else:
                value = value.astype(jnp.float32)
            obs[key] = value
        obs['cont'] = 1.0 - obs['is_terminal'].astype(jnp.float32)
        return obs


class GCWorldModel(nj.Module):
    """Goal-conditioned World Model.
    
    Extends the standard R2I world model with support for:
    - Separate grid and goal observations
    - reconstruct_only_obs option to skip goal reconstruction loss
    """
    
    def __init__(self, obs_space, act_space, config):
        self.obs_space = obs_space
        self.act_space = act_space['action']
        self.config = config
        
        # Get shapes for encoder/decoder
        shapes = {k: tuple(v.shape) for k, v in obs_space.items()}
        shapes = {k: v for k, v in shapes.items() if not k.startswith('log_')}
        self.shapes = shapes
        
        # Determine which keys to reconstruct
        self.reconstruct_only_obs = getattr(config, 'reconstruct_only_obs', False)
        self._obs_keys = self._get_obs_keys()
        
        # Create encoder (uses all observation keys)
        self.encoder = nets.MultiEncoder(shapes, **config.encoder, name='enc')
        
        # Create decoder
        if self.reconstruct_only_obs:
            # Only reconstruct observation keys (grid, grid_2d), not goals
            decoder_shapes = {k: v for k, v in shapes.items() if k in self._obs_keys}
        else:
            decoder_shapes = shapes
        self.decoder_shapes = decoder_shapes
        
        # Setup RSSM
        rssms = {
            'gru': nets.GRU_RSSM,
            'siso': ssm_nets.S3M,
            'mimo': ssm_nets.S3M,
        }
        kws = {
            'gru': dict(**config.rssm),
            'siso': dict(**config.rssm, ssm_kwargs=config.ssm, ssm=config.ssm_type),
            'mimo': dict(**config.rssm, ssm_kwargs=dict(**config.ssm, **config.ssm_cell), ssm=config.ssm_type),
        }
        self.rssm = rssms[config.ssm_type](**kws[config.ssm_type], name='rssm')
        
        # Setup heads
        self.heads = {
            'decoder': nets.MultiDecoder(decoder_shapes, **config.decoder, name='dec'),
            'reward': nets.MLP((), **config.reward_head, name='rew'),
            'cont': nets.MLP((), **config.cont_head, name='cont'),
        }
        
        # Setup optimizer
        opt_groups = {}
        for gr in config.model_opt_group_keys:
            keys = config.model_opt_group_keys[gr].split('|')
            for key in keys:
                opt_groups[key] = dict(config.model_opt_groups[gr])
        self.opt = jaxutils.Optimizer(name='model_opt', **config.model_opt, layer_opts=opt_groups)
        
        # Setup loss scales
        scales = self.config.loss_scales.copy()
        image, vector = scales.pop('image'), scales.pop('vector')
        scales.update({k: image for k in self.heads['decoder'].cnn_shapes})
        scales.update({k: vector for k in self.heads['decoder'].mlp_shapes})
        self.scales = scales
    
    def _get_obs_keys(self):
        """Get observation keys (excluding goals) for reconstruct_only_obs."""
        obs_keys = set()
        for key in self.shapes.keys():
            # Keys that represent observations (not goals)
            if key in ('grid', 'grid_2d'):
                obs_keys.add(key)
        return obs_keys
    
    def initial(self, batch_size):
        prev_latent = self.rssm.initial(batch_size)
        prev_action = jnp.zeros((batch_size, *self.act_space.shape))
        return prev_latent, prev_action
    
    def train(self, data, state):
        modules = [self.encoder, self.rssm, *self.heads.values()]
        mets, (state, outs, metrics) = self.opt(
            modules, self.loss, data, state, has_aux=True)
        metrics.update(mets)
        return state, outs, metrics
    
    def loss(self, data, state):
        embed = self.encoder(data)
        prev_latent, prev_action = state
        prev_actions = jnp.concatenate([
            prev_action[:, None], data['action'][:, :-1]], 1)
        post, prior = self.rssm.observe(
            embed, prev_actions, data['is_first'], prev_latent)
        
        dists = {}
        feats = {**post, 'embed': embed}
        for name, head in self.heads.items():
            out = head(feats if name in self.config.grad_heads else sg(feats))
            out = out if isinstance(out, dict) else {name: out}
            dists.update(out)
        
        losses = {}
        losses['dyn'] = self.rssm.dyn_loss(post, prior, **self.config.dyn_loss)
        losses['rep'] = self.rssm.rep_loss(post, prior, **self.config.rep_loss)
        
        # Compute reconstruction losses
        for key, dist in dists.items():
            if key in data:
                loss = -dist.log_prob(data[key].astype(jnp.float32))
                assert loss.shape == embed.shape[:2], (key, loss.shape)
                losses[key] = loss
        
        scaled = {k: v * self.scales.get(k, 1.0) for k, v in losses.items()}
        model_loss = sum(scaled.values())
        
        out = {'embed': embed, 'post': post, 'prior': prior}
        out.update({f'{k}_loss': v for k, v in losses.items()})
        
        last_latent = {k: v[:, -1] for k, v in post.items()}
        last_action = data['action'][:, -1]
        state = last_latent, last_action
        
        metrics = self._metrics(data, dists, post, prior, losses, model_loss)
        return model_loss.mean(), (state, out, metrics)
    
    def imagine(self, policy, start, horizon):
        first_cont = (1.0 - start['is_terminal']).astype(jnp.float32)
        keys = list(self.rssm.initial(1).keys())
        start = {k: v for k, v in start.items() if k in keys}
        start['action'] = policy(start)
        
        def step(prev, _):
            prev = prev.copy()
            state = self.rssm.img_step(prev, prev.pop('action'))
            return {**state, 'action': policy(state)}
        
        traj = jaxutils.scan(
            step, jnp.arange(horizon), start, self.config.imag_unroll)
        traj = {
            k: jnp.concatenate([start[k][None], v], 0) for k, v in traj.items()}
        cont = self.heads['cont'](traj).mode()
        traj['cont'] = jnp.concatenate([first_cont[None], cont[1:]], 0)
        discount = 1 - 1 / self.config.horizon
        traj['weight'] = jnp.cumprod(discount * traj['cont'], 0) / discount
        return traj
    
    def report(self, data):
        state = self.initial(len(data['is_first']))
        report = {}
        report.update(self.loss(data, state)[-1][-1])
        return report
    
    def _metrics(self, data, dists, post, prior, losses, model_loss):
        entropy = lambda feat: self.rssm.get_dist(feat).entropy()
        metrics = {}
        metrics.update(jaxutils.tensorstats(entropy(prior), 'prior_ent'))
        metrics.update(jaxutils.tensorstats(entropy(post), 'post_ent'))
        metrics.update({f'{k}_loss_mean': v.mean() for k, v in losses.items()})
        metrics.update({f'{k}_loss_std': v.std() for k, v in losses.items()})
        metrics['model_loss_mean'] = model_loss.mean()
        metrics['model_loss_std'] = model_loss.std()
        metrics['reward_max_data'] = jnp.abs(data['reward']).max()
        metrics['reward_max_pred'] = jnp.abs(dists['reward'].mean()).max()
        if 'reward' in dists and not self.config.jax.debug_nans:
            stats = jaxutils.balance_stats(dists['reward'], data['reward'], 0.1)
            metrics.update({f'reward_{k}': v for k, v in stats.items()})
        if 'cont' in dists and not self.config.jax.debug_nans:
            stats = jaxutils.balance_stats(dists['cont'], data['cont'], 0.5)
            metrics.update({f'cont_{k}': v for k, v in stats.items()})
        return metrics


# Alias for compatibility
Agent = GCR2IAgent

