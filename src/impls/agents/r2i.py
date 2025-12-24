"""Goal-conditioned Recall2Imagine (R2I) agent for BoxMovingEnv.

This module provides a goal-conditioned variant of the R2I agent that:
- Encodes goals separately with frozen encoder weights (detached gradients)
- Re-encodes goals every forward pass (policy step) with detached gradients
- Concatenates goal embedding with latent state (deter, stoch) for actor/critic
- The input to actor/critic is z_g = [deter, stoch, goal_embedding]
- Only reconstructs observations in decoder (goals are never reconstructed)
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
import numpy as np
import ruamel.yaml as yaml

# Import embodied (it's at recall2imagine/embodied)
import embodied

# Import recall2imagine modules - they use relative imports so we import the package
from recall2imagine import agent as r2i_agent
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


def _broadcast_goal_embed(goal_embed, target_shape):
    """Broadcast goal_embed to match target_shape prefix (batch dimensions).
    
    This handles the common case where:
    - goal_embed is (Batch, Dim) and target is (Horizon, Batch, Dim)
    - goal_embed is (Batch, Seq, Dim) and target is (Batch*Seq, Dim)
    
    Args:
        goal_embed: Goal embedding array
        target_shape: Target shape to match (typically deter.shape)
        
    Returns:
        goal_embed broadcast to match target_shape prefix
    """
    target_prefix = target_shape[:-1]  # All dims except feature dim
    
    # First, flatten any extra dimensions in goal_embed beyond 2D
    while len(goal_embed.shape) > len(target_shape):
        # Flatten middle dimensions: (A, B, C, D) -> (A, B*C, D)
        goal_embed = goal_embed.reshape(
            goal_embed.shape[:1] + (-1,) + goal_embed.shape[-1:]
        )
    
    # Expand dimensions if goal_embed has fewer dims than target
    diff_dims = len(target_shape) - len(goal_embed.shape)
    if diff_dims > 0:
        # Prepend singleton dimensions: (B, D) -> (1, B, D) for (H, B, D) alignment
        goal_embed = jnp.expand_dims(goal_embed, axis=tuple(range(diff_dims)))
    
    # Now broadcast to match target prefix
    goal_embed = jnp.broadcast_to(
        goal_embed, 
        target_prefix + goal_embed.shape[-1:]
    )
    
    return goal_embed


class GCRewardHead(nj.Module):
    """Goal-conditioned reward head: reward = f(deter, stoch, goal_embed).
    
    This head takes the latent state (deter, stoch) concatenated with the goal
    embedding to predict rewards. This allows the reward model to learn the
    relationship between current state and goal.
    
    For HER training, this head will receive constant reward targets when
    the current state matches the relabeled goal (positive samples).
    """
    
    def __init__(self, config):
        self.config = config
        # Get reward_head config, use defaults if not specified
        reward_cfg = dict(config.reward_head) if hasattr(config, 'reward_head') else {}
        self._layers = reward_cfg.get('layers', 2)
        self._units = reward_cfg.get('units', 64)
        self._act = reward_cfg.get('act', 'silu')
        self._norm = reward_cfg.get('norm', 'layer')
        self._dist = reward_cfg.get('dist', 'symlog_disc')
        self._outscale = reward_cfg.get('outscale', 0.0)
        self._outnorm = reward_cfg.get('outnorm', False)
        self._bins = reward_cfg.get('bins', 255)
        
        # Create the underlying MLP
        self._mlp = nets.MLP(
            shape=(),
            layers=self._layers,
            units=self._units,
            act=self._act,
            norm=self._norm,
            dist=self._dist,
            outscale=self._outscale,
            outnorm=self._outnorm,
            bins=self._bins,
            inputs=['tensor'],  # We'll manually concatenate inputs
            name='gc_reward_mlp'
        )
    
    def __call__(self, feats):
        """Compute goal-conditioned reward.
        
        Args:
            feats: Dict containing 'deter', 'stoch', and 'goal_embed'.
        
        Returns:
            Reward distribution.
            
        Raises:
            ValueError: If 'goal_embed' is not present in feats.
        """
        # Get latent state components
        deter = feats['deter']
        stoch = feats['stoch']
        
        # Flatten stoch if it has categorical structure
        if len(stoch.shape) > len(deter.shape):
            stoch = stoch.reshape(stoch.shape[:-2] + (-1,))
        
        # Get goal embedding - REQUIRED for goal-conditioned reward
        if 'goal_embed' not in feats:
            raise ValueError(
                f"GCRewardHead requires 'goal_embed' in feats but it was not found. "
                f"Available keys: {list(feats.keys())}. "
                f"Ensure goal embeddings are passed to the reward head during training."
            )
        
        goal_embed = _broadcast_goal_embed(feats['goal_embed'], deter.shape)
        
        # Concatenate all inputs: [deter, stoch, goal_embed]
        # This is the z_g representation
        combined = jnp.concatenate([deter, stoch, goal_embed], axis=-1)
        
        return self._mlp({'tensor': combined})


class GCVFunction(nj.Module):
    """Goal-conditioned Value Function: V(s, g).
    
    This value function takes [deter, stoch, goal_embed] as input to estimate
    the goal-conditioned value. It follows the same interface as r2i_agent.VFunction
    but properly handles the goal embedding.
    """
    
    def __init__(self, rewfn, config):
        self.rewfn = rewfn
        self.config = config
        # Note: We use inputs=['tensor'] to handle our manually concatenated input
        # instead of dims='deter' which would only use the deter key
        # Filter out 'inputs' from critic config since we override it with 'tensor'
        critic_config = {k: v for k, v in dict(self.config.critic).items() if k != 'inputs'}
        self.net = nets.MLP((), name='net', inputs=['tensor'], **critic_config)
        self.slow = nets.MLP((), name='slow', inputs=['tensor'], **critic_config)
        self.updater = jaxutils.SlowUpdater(
            self.net, self.slow,
            self.config.slow_critic_fraction,
            self.config.slow_critic_update)
        self.opt = jaxutils.Optimizer(name='critic_opt', **self.config.critic_opt)
    
    def _get_input(self, traj):
        """Concatenate deter, stoch, and goal_embed into a single tensor."""
        deter = traj['deter']
        stoch = traj['stoch']
        
        # Flatten stoch if it has categorical structure
        if len(stoch.shape) > len(deter.shape):
            stoch = stoch.reshape(stoch.shape[:-2] + (-1,))
        
        # Get goal embedding - it MUST be present for goal-conditioned RL
        if 'goal_embed' not in traj:
            raise ValueError(
                f"goal_embed required for GCVFunction but not found. "
                f"Available keys: {list(traj.keys())}"
            )
        
        goal_embed = _broadcast_goal_embed(traj['goal_embed'], deter.shape)
        
        # Concatenate: z_g = [deter, stoch, goal_embed]
        return jnp.concatenate([deter, stoch, goal_embed], axis=-1)
    
    def train(self, traj, actor):
        target = sg(self.score(traj)[1])
        mets, metrics = self.opt(self.net, self.loss, traj, target, has_aux=True)
        metrics.update(mets)
        self.updater()
        return metrics
    
    def loss(self, traj, target):
        metrics = {}
        traj = {k: v[:-1] for k, v in traj.items()}
        inp = self._get_input(traj)
        dist = self.net({'tensor': inp})
        loss = -dist.log_prob(sg(target))
        if self.config.critic_slowreg == 'logprob':
            slow_inp = self._get_input(traj)
            reg = -dist.log_prob(sg(self.slow({'tensor': slow_inp}).mean()))
        elif self.config.critic_slowreg == 'xent':
            slow_inp = self._get_input(traj)
            reg = -jnp.einsum(
                '...i,...i->...',
                sg(self.slow({'tensor': slow_inp}).probs),
                jnp.log(dist.probs))
        else:
            raise NotImplementedError(self.config.critic_slowreg)
        loss += self.config.loss_scales.slowreg * reg
        loss = (loss * sg(traj['weight'])).mean()
        loss *= self.config.loss_scales.critic
        metrics = jaxutils.tensorstats(dist.mean())
        return loss, metrics
    
    def score(self, traj, actor=None):
        rew = self.rewfn(traj)
        assert len(rew) == len(traj['action']) - 1, (
            'should provide rewards for all but last action')
        discount = 1 - 1 / self.config.horizon
        disc = traj['cont'][1:] * discount
        inp = self._get_input(traj)
        value = self.net({'tensor': inp}).mean()
        vals = [value[-1]]
        interm = rew + disc * value[1:] * (1 - self.config.return_lambda)
        for t in reversed(range(len(disc))):
            vals.append(interm[t] + disc[t] * self.config.return_lambda * vals[-1])
        ret = jnp.stack(list(reversed(vals))[:-1])
        return rew, ret, value[:-1]


class GCGreedy(nj.Module):
    """Goal-conditioned Greedy behavior that uses goal embeddings."""
    
    def __init__(self, wm, act_space, config):
        self.wm = wm
        self.config = config
        rewfn = lambda s: wm.heads['reward'](s).mean()[1:]
        if config.critic_type == 'vfunction':
            # Use goal-conditioned VFunction that properly handles goal_embed
            critics = {'extr': GCVFunction(rewfn, config, name='critic')}
        else:
            raise NotImplementedError(config.critic_type)
        self.ac = GCImagActorCritic(
            wm, critics, {'extr': 1.0}, act_space, config, name='ac')
    
    def initial(self, batch_size):
        return self.ac.initial(batch_size)
    
    def policy(self, latent, state):
        return self.ac.policy(latent, state)
    
    def train(self, imagine, start, data):
        return self.ac.train(imagine, start, data)
    
    def report(self, data):
        return {}


class GCImagActorCritic(nj.Module):
    """Goal-conditioned Actor-Critic that uses z_g = [latent_state, goal_embedding].
    
    The goal embedding is computed by passing goal observations through the frozen
    encoder (with stop_gradient applied to encoder outputs for goals).
    """
    
    def __init__(self, wm, critics, scales, act_space, config):
        self.wm = wm
        critics = {k: v for k, v in critics.items() if scales[k]}
        for key, scale in scales.items():
            assert not scale or key in critics, key
        self.critics = {k: v for k, v in critics.items() if scales[k]}
        self.scales = scales
        self.act_space = act_space
        self.config = config
        disc = act_space.discrete
        self.grad = config.actor_grad_disc if disc else config.actor_grad_cont
        
        # Actor takes goal-augmented state: [deter, stoch, goal_embedding]
        # We use custom dims to handle the goal-augmented input
        self.actor = nets.MLP(
            name='actor', dims='deter', shape=act_space.shape, **config.actor,
            dist=config.actor_dist_disc if disc else config.actor_dist_cont)
        self.retnorms = {
            k: jaxutils.Moments(**config.retnorm, name=f'retnorm_{k}')
            for k in critics}
        self.opt = jaxutils.Optimizer(name='actor_opt', **config.actor_opt)
    
    def initial(self, batch_size):
        return {}
    
    def _get_goal_embedding(self, data_or_state):
        """Compute goal embedding with frozen encoder weights.
        
        The goal is encoded using the same encoder as observations, but with
        stop_gradient applied so gradients don't flow back through the encoder.
        
        Raises:
            ValueError: If no goal data (goal or goal_2d) is found in input.
        """
        # Check if we have goal data available - raise error if missing
        if 'goal' not in data_or_state and 'goal_2d' not in data_or_state:
            raise ValueError(
                f"Goal data required but not found. Expected 'goal' or 'goal_2d' key, "
                f"got keys: {list(data_or_state.keys())}"
            )
        
        # Build goal-only data dict for encoder
        goal_data = {}
        for key in data_or_state:
            if key.startswith('goal'):
                # Rename goal keys to match observation keys for encoder
                if key == 'goal':
                    goal_data['grid'] = data_or_state[key]
                elif key == 'goal_2d':
                    goal_data['grid_2d'] = data_or_state[key]
                else:
                    goal_data[key] = data_or_state[key]
        
        if not goal_data:
            raise ValueError(
                "Goal keys found but no valid goal data could be extracted. "
                "Check that goal/goal_2d keys contain valid data."
            )
        
        # Encode goals with frozen encoder (stop_gradient on output)
        goal_embed = sg(self.wm.encoder(goal_data))
        return goal_embed
    
    def _augment_state_with_goal(self, state, goal_embed):
        """Augment latent state with goal embedding: z_g = [deter, stoch, e_g].
        
        Uses robust JAX broadcasting to handle all shape combinations properly.
        This is multi-GPU safe as it uses pure JAX operations.
        
        Raises:
            ValueError: If goal_embed is None.
        """
        if goal_embed is None:
            raise ValueError("goal_embed cannot be None - goals are required for goal-conditioned RL")
        
        deter = state['deter']
        stoch = state['stoch']
        
        # Flatten stoch if it has categorical structure (e.g., [batch, stoch_dim, classes])
        # We need to match the number of dimensions with deter
        if len(stoch.shape) > len(deter.shape):
            # stoch has shape [..., stoch_dim, classes], flatten last two dims
            stoch = stoch.reshape(stoch.shape[:-2] + (-1,))
        
        # Use robust broadcasting to match goal_embed to deter shape
        goal_embed = _broadcast_goal_embed(goal_embed, deter.shape)
        
        # Concatenate deter, stoch, and goal_embed
        # This creates z_g = [deter, stoch, goal_embedding]
        augmented = state.copy()
        augmented['deter'] = jnp.concatenate([deter, stoch, goal_embed], axis=-1)
        return augmented
    
    def policy(self, latent, carry):
        """Policy function that uses goal-augmented state."""
        # Get goal embedding if available in latent (during policy execution)
        goal_embed = latent.get('goal_embed', None)
        
        if goal_embed is not None:
            augmented = self._augment_state_with_goal(latent, goal_embed)
        else:
            augmented = latent
        
        return {'action': self.actor(augmented)}, carry
    
    def train(self, imagine, start, context):
        """Train with goal-augmented states."""
        # Get goal embedding from context (frozen encoder)
        goal_embed = self._get_goal_embedding(context)
        
        # Flatten goal_embed to match flattened batch dimensions
        # context has (batch, seq, ...) but start is flattened to (batch*seq, ...)
        # goal_embed has shape (batch, seq, embed_dim) -> flatten to (batch*seq, embed_dim)
        if len(goal_embed.shape) > 2:
            goal_embed = goal_embed.reshape((-1,) + goal_embed.shape[-1:])
        
        def loss(start):
            # Create policy that augments state with goal embedding
            def goal_policy(s):
                aug_s = self._augment_state_with_goal(s, goal_embed)
                return self.actor(sg(aug_s)).sample(seed=nj.rng())
            
            traj = imagine(goal_policy, start, self.config.imag_horizon)
            
            # Store goal_embed in traj for critic
            # traj has shape [horizon, batch, ...]
            # goal_embed has shape [batch, embed_dim] (already flattened)
            traj_goal_embed = jnp.broadcast_to(
                goal_embed[None], 
                (traj['deter'].shape[0],) + goal_embed.shape
            )
            traj['goal_embed'] = traj_goal_embed
            
            loss, metrics = self.loss(traj, goal_embed)
            return loss, (traj, metrics)
        
        mets, (traj, metrics) = self.opt(self.actor, loss, start, has_aux=True)
        metrics.update(mets)
        for key, critic in self.critics.items():
            mets = critic.train(traj, self.actor)
            metrics.update({f'{key}_critic_{k}': v for k, v in mets.items()})
        return traj, metrics
    
    def loss(self, traj, goal_embed=None):
        """Compute actor loss with goal-augmented states."""
        metrics = {}
        advs = []
        total = sum(self.scales[k] for k in self.critics)
        for key, critic in self.critics.items():
            rew, ret, base = critic.score(traj, self.actor)
            offset, invscale = self.retnorms[key](ret)
            normed_ret = (ret - offset) / invscale
            normed_base = (base - offset) / invscale
            advs.append((normed_ret - normed_base) * self.scales[key] / total)
            metrics.update(jaxutils.tensorstats(rew, f'{key}_reward'))
            metrics.update(jaxutils.tensorstats(ret, f'{key}_return_raw'))
            metrics.update(jaxutils.tensorstats(normed_ret, f'{key}_return_normed'))
            metrics[f'{key}_return_rate'] = (jnp.abs(ret) >= 0.5).mean()
        adv = jnp.stack(advs).sum(0)
        
        # Broadcast goal_embed to match trajectory dimensions
        # traj has shape [horizon, batch, ...], goal_embed has [batch, embed_dim]
        # We need [horizon, batch, embed_dim]
        if goal_embed is not None and len(goal_embed.shape) < len(traj['deter'].shape):
            goal_embed_broadcast = jnp.broadcast_to(
                goal_embed[None], 
                (traj['deter'].shape[0],) + goal_embed.shape
            )
        else:
            goal_embed_broadcast = goal_embed
        
        # Use goal-augmented state for policy
        aug_traj = self._augment_state_with_goal(traj, goal_embed_broadcast)
        policy = self.actor(sg(aug_traj))
        logpi = policy.log_prob(sg(traj['action']))[:-1]
        loss = {'backprop': -adv, 'reinforce': -logpi * sg(adv)}[self.grad]
        ent = policy.entropy()[:-1]
        loss -= self.config.actent * ent
        loss *= sg(traj['weight'])[:-1]
        loss *= self.config.loss_scales.actor
        metrics.update(self._metrics(traj, policy, logpi, ent, adv))
        return loss.mean(), metrics
    
    def _metrics(self, traj, policy, logpi, ent, adv):
        metrics = {}
        ent = policy.entropy()[:-1]
        rand = (ent - policy.minent) / (policy.maxent - policy.minent)
        rand = rand.mean(range(2, len(rand.shape)))
        act = traj['action']
        act = jnp.argmax(act, -1) if self.act_space.discrete else act
        metrics.update(jaxutils.tensorstats(act, 'action'))
        metrics.update(jaxutils.tensorstats(rand, 'policy_randomness'))
        metrics.update(jaxutils.tensorstats(ent, 'policy_entropy'))
        metrics.update(jaxutils.tensorstats(logpi, 'policy_logprob'))
        metrics.update(jaxutils.tensorstats(adv, 'adv'))
        metrics['imag_weight_dist'] = jaxutils.subsample(traj['weight'])
        return metrics


@jaxagent.Wrapper
class GCR2IAgent(nj.Module):
    """Goal-conditioned R2I Agent.
    
    This agent extends the standard R2I agent with:
    - Separate goal embedding with frozen encoder weights
    - Goal embedding concatenated with latent state for actor/critic
    - Input to actor/critic: z_g = [deter, stoch, goal_embedding]
    """
    
    configs = load_configs()
    
    def __init__(self, obs_space, act_space, step, config):
        self.config = config
        self.obs_space = obs_space
        self.act_space = act_space['action']
        self.step = step
        self.wm = GCWorldModel(obs_space, act_space, config, name='wm')
        
        # Use goal-conditioned behavior that handles goal embeddings
        if config.task_behavior == 'Greedy':
            self.task_behavior = GCGreedy(
                self.wm, self.act_space, self.config, name='task_behavior')
        else:
            # Fallback to standard behavior for non-Greedy
            from recall2imagine import behaviors
            self.task_behavior = getattr(behaviors, config.task_behavior)(
                self.wm, self.act_space, self.config, name='task_behavior')
        
        if config.expl_behavior == 'None':
            self.expl_behavior = self.task_behavior
        else:
            from recall2imagine import behaviors
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
        """Forward pass for policy.
        
        Each forward pass:
        1. (Re-)encodes the goal with frozen encoder weights (detached gradients)
        2. Encodes observations and updates the latent state via RSSM
        3. Passes z_g = [deter, stoch, goal_embedding] to actor/critic
        4. Returns action to be appended to buffer
        """
        self.config.jax.jit and print('Tracing policy function.')
        obs = self.preprocess(obs)
        (prev_latent, prev_action), task_state, expl_state = state
        
        # Step 1: Re-encode goals with frozen encoder (detached gradients)
        # This happens every forward pass to ensure fresh goal embeddings
        goal_embed = self.wm.encode_goal_frozen(obs)
        
        # Step 2: Encode observation (excluding goals) for world model
        embed = self.wm.encode_obs(obs)
        latent, _ = self.wm.rssm.obs_step(
            prev_latent, prev_action, embed, obs['is_first'])
        
        # Attach goal embedding to latent state for actor/critic
        # goal_embed is always valid (encode_goal_frozen raises if missing)
        # IMPORTANT: Create separate dict for actor/critic, don't modify latent
        # because RSSM state must not contain goal_embed key
        latent_for_actor = {**latent, 'goal_embed': goal_embed}
        
        self.expl_behavior.policy(latent_for_actor, expl_state)
        task_outs, task_state = self.task_behavior.policy(latent_for_actor, task_state)
        expl_outs, expl_state = self.expl_behavior.policy(latent_for_actor, expl_state)
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
        # Use original latent (without goal_embed) for RSSM state
        state = ((latent, outs['action']), task_state, expl_state)
        return outs, state
    
    def train(self, data, state):
        self.config.jax.jit and print('Tracing train function.')
        
        # Relabel data (HER)
        data = self._relabel_data(data)
        
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

    def _relabel_data(self, data):
        """Relabel data with goals sampled from future states (HER).
        
        This method:
        1. Samples future time steps t' > t geometrically (or uniformly).
        2. Sets goal at t to obs at t'.
        3. Updates reward at t to 1.0 if t == t' else 0.0.
        """
        # Only apply relabeling if we have goal keys
        goal_keys = [k for k in data.keys() if k.startswith('goal')]
        if not goal_keys:
            return data
            
        # Ensure we have time dimension [batch, length, ...]
        if len(data['is_first'].shape) < 2:
            return data
            
        batch_size, seq_len = data['is_first'].shape[:2]
        rng = nj.rng()
        
        # Geometric sampling parameters
        discount = self.config.get('goal_relabel_discount', 0.99)
        
        # Sample offsets: floor(log(u) / log(discount))
        # u ~ U(0,1)
        u = jax.random.uniform(rng, (batch_size, seq_len))
        # We want offsets >= 0. log(u) is negative, log(discount) is negative.
        # This gives a geometric distribution.
        offsets = jnp.floor(jnp.log(u) / jnp.log(discount)).astype(jnp.int32)
        
        # Compute target indices: t' = t + offset
        indices = jnp.arange(seq_len)[None, :]
        target_indices = indices + offsets
        
        # Clamp to sequence length
        target_indices = jnp.minimum(target_indices, seq_len - 1)
        
        # Validate that t' is in the same episode as t
        # We use segment IDs derived from is_first
        segment_ids = jnp.cumsum(data['is_first'].astype(jnp.int32), axis=1)
        
        # Helper to gather values at target indices
        def gather_time(arr, time_idxs):
            # arr: (B, T, ...)
            # time_idxs: (B, T)
            # returns: (B, T, ...)
            batch_idxs = jnp.arange(batch_size)[:, None]
            return arr[batch_idxs, time_idxs]
            
        target_segments = gather_time(segment_ids, target_indices)
        current_segments = segment_ids
        
        # Valid if segments match (same episode)
        valid_mask = (target_segments == current_segments)
        
        new_data = data.copy()
        
        # Update goals
        for goal_key in goal_keys:
            # Determine corresponding observation key using world model's mapping
            obs_key = self.wm._goal_key_to_obs_key(goal_key)
            
            if obs_key not in data:
                raise ValueError(
                    f"Cannot relabel goal '{goal_key}': corresponding observation key "
                    f"'{obs_key}' not found in data. Available keys: {list(data.keys())}"
                )
            
            future_obs = gather_time(data[obs_key], target_indices)
            current_goal = data[goal_key]
            
            # Expand mask to match goal shape
            mask = valid_mask
            while len(mask.shape) < len(current_goal.shape):
                mask = mask[..., None]
            
            # Replace goal with future obs where valid, otherwise keep original
            new_goal = jnp.where(mask, future_obs, current_goal)
            new_data[goal_key] = new_goal
        
        # Update rewards
        # Reward is 1.0 (or constant) if t == t', else 0.0
        # If offset=0, then t=t'.
        is_match = (indices == target_indices) & valid_mask
        
        reward_constant = self.config.get('reward_constant', 1.0)
        new_reward = jnp.where(is_match, reward_constant, 0.0).astype(jnp.float32)
        
        new_data['reward'] = new_reward
        
        return new_data

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
    
    Key features:
    - Encodes observations (grid) and goals separately
    - Goals are ALWAYS encoded with frozen encoder weights (stop_gradient)
    - Goals are re-encoded every forward pass with detached gradients
    - Only observation encodings are used for RSSM posterior
    - Decoder only reconstructs observations, NEVER goals
    """
    
    def __init__(self, obs_space, act_space, config):
        self.obs_space = obs_space
        self.act_space = act_space['action']
        self.config = config
        
        # Get shapes for encoder/decoder
        shapes = {k: tuple(v.shape) for k, v in obs_space.items()}
        shapes = {k: v for k, v in shapes.items() if not k.startswith('log_')}
        self.shapes = shapes
        
        # Identify observation and goal keys
        self._obs_keys = self._get_obs_keys()
        self._goal_keys = self._get_goal_keys()
        
        # Shapes for observation-only encoder/decoder (goals are never reconstructed)
        self._obs_shapes = {k: v for k, v in shapes.items() if k in self._obs_keys}
        self._goal_shapes = {k: v for k, v in shapes.items() if k in self._goal_keys}
        
        # Create encoder for observations only (goals encoded separately with frozen weights)
        self.encoder = nets.MultiEncoder(self._obs_shapes, **config.encoder, name='enc')
        
        # Decoder only reconstructs observations, never goals
        # Goals are always detached from the world model
        decoder_shapes = self._obs_shapes
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
        # Use goal-conditioned reward head that takes [deter, stoch, goal_embed]
        self.heads = {
            'decoder': nets.MultiDecoder(decoder_shapes, **config.decoder, name='dec'),
            'reward': GCRewardHead(config, name='rew'),  # Goal-conditioned reward
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
    
    # Keys that are NOT observations (metadata, goals, actions, etc.)
    _NON_OBS_KEYS = frozenset({
        'reward', 'is_first', 'is_last', 'is_terminal', 'cont',
        'action', 'reset', 'key', 'log_entropy',
    })
    
    def _get_obs_keys(self):
        """Get observation keys (excluding goals and metadata).
        
        Dynamically detects observation keys by excluding:
        - Keys starting with 'goal' (goal observations)
        - Keys starting with 'log_' (logging metadata)
        - Known metadata keys (reward, is_first, is_last, etc.)
        
        Raises:
            ValueError: If no observation keys are found.
        """
        obs_keys = set()
        for key in self.shapes.keys():
            # Skip goal keys
            if key.startswith('goal'):
                continue
            # Skip log keys
            if key.startswith('log_'):
                continue
            # Skip known non-observation keys
            if key in self._NON_OBS_KEYS:
                continue
            obs_keys.add(key)
        
        if not obs_keys:
            raise ValueError(
                f"No observation keys found in obs_space. "
                f"Available keys: {list(self.shapes.keys())}. "
                f"Expected at least one of: 'grid', 'grid_2d', 'image', etc."
            )
        return obs_keys
    
    def _get_goal_keys(self):
        """Get goal keys."""
        goal_keys = set()
        for key in self.shapes.keys():
            if key.startswith('goal'):
                goal_keys.add(key)
        return goal_keys
    
    def _goal_key_to_obs_key(self, goal_key):
        """Map a goal key to its corresponding observation key.
        
        Mapping rules:
        - 'goal' -> 'grid' (if exists) or 'image' (if exists)
        - 'goal_2d' -> 'grid_2d' (if exists) or 'image' (if exists)
        - 'goal_image' -> 'image'
        - 'goal_<x>' -> '<x>' if '<x>' exists in obs_keys
        
        Args:
            goal_key: The goal key to map
            
        Returns:
            The corresponding observation key
            
        Raises:
            ValueError: If no matching observation key is found
        """
        # Direct mapping attempts
        if goal_key == 'goal':
            # Try 'grid' first, then 'image'
            if 'grid' in self._obs_keys:
                return 'grid'
            if 'image' in self._obs_keys:
                return 'image'
        elif goal_key == 'goal_2d':
            if 'grid_2d' in self._obs_keys:
                return 'grid_2d'
            if 'image' in self._obs_keys:
                return 'image'
        elif goal_key == 'goal_image':
            if 'image' in self._obs_keys:
                return 'image'
        else:
            # Try stripping 'goal_' prefix
            stripped = goal_key[5:] if goal_key.startswith('goal_') else goal_key
            if stripped in self._obs_keys:
                return stripped
        
        raise ValueError(
            f"Cannot map goal key '{goal_key}' to observation key. "
            f"Available obs_keys: {self._obs_keys}"
        )
    
    def encode_obs(self, data):
        """Encode only observations (not goals) for world model."""
        obs_data = {k: v for k, v in data.items() if k in self._obs_keys}
        return self.encoder(obs_data)
    
    def encode_goal_frozen(self, data):
        """Encode goals with frozen encoder weights (stop_gradient).
        
        The goal is encoded using the same encoder architecture, but with
        stop_gradient applied so gradients don't flow through the encoder.
        
        Raises:
            ValueError: If no goal data found or goal data cannot be mapped.
        """
        # Build goal data dict, mapping goal keys to obs keys for encoder
        goal_data = {}
        for key in data:
            if key in self._goal_keys:
                obs_key = self._goal_key_to_obs_key(key)
                goal_data[obs_key] = data[key]
        
        if not goal_data:
            raise ValueError(
                f"Goal data required but not found. "
                f"Expected keys starting with 'goal' (e.g., 'goal', 'goal_2d', 'goal_image'). "
                f"Got keys: {list(data.keys())}, goal_keys: {self._goal_keys}"
            )
        
        # Encode with frozen weights (stop_gradient on output)
        goal_embed = sg(self.encoder(goal_data))
        return goal_embed
    
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
        # Encode only observations for world model
        embed = self.encode_obs(data)
        
        # Check if any goal keys are present in the data
        data_goal_keys = [k for k in data.keys() if k in self._goal_keys]
        
        # Encode goals (frozen) - REQUIRED for goal-conditioned RL
        if not data_goal_keys:
            raise ValueError(
                f"Goal data required for goal-conditioned world model training. "
                f"Expected keys from {self._goal_keys} but found none in data. "
                f"Available keys: {list(data.keys())}"
            )
        
        # This returns (batch, seq, embed) if data has time dim
        goal_embed = self.encode_goal_frozen(data)
            
        prev_latent, prev_action = state
        prev_actions = jnp.concatenate([
            prev_action[:, None], data['action'][:, :-1]], 1)
        post, prior = self.rssm.observe(
            embed, prev_actions, data['is_first'], prev_latent)
        
        dists = {}
        feats = {**post, 'embed': embed, 'goal_embed': goal_embed}
            
        for name, head in self.heads.items():
            out = head(feats if name in self.config.grad_heads else sg(feats))
            out = out if isinstance(out, dict) else {name: out}
            dists.update(out)
        
        losses = {}
        losses['dyn'] = self.rssm.dyn_loss(post, prior, **self.config.dyn_loss)
        losses['rep'] = self.rssm.rep_loss(post, prior, **self.config.rep_loss)
        
        # Compute reconstruction losses (only for decoder shapes)
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

