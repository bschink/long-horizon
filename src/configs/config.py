import os
from dataclasses import dataclass
from envs import legal_envs
import ml_collections
from ml_collections import FrozenConfigDict
from typing import Literal

    
@dataclass
class ExpConfig:
    # wandb logging
    name: str = "default_config"
    project: str = "long-horizon"
    mode: str = "online"
    entity: str = "uhh_machine_learning"

    # Replay buffer and batch size and seed
    num_envs: int = 1024
    batch_size: int = 256
    seed: int = 0
    max_replay_size: int = 10000

    # Number of updates etc
    epochs: int = 10
    intervals_per_epoch: int = 100
    updates_per_rollout: int = 1000

    # Miscellaneous
    use_targets: bool = False
    gamma: float = 0.99
    use_future_and_random_goals: bool = False  # noqa: E501 Whether to use environment goals, default - geometric sampling of future states

    # Evaluation settings
    eval_different_box_numbers: bool = False
    eval_special: bool = False
    """In addition to standard evaluation also evaluate in 'special' mode. The specifics of special mode depend on level generator used."""

    # Filtering settings
    filtering: Literal["horizontal", "vertical", "quarter"] | None = None
    """Type of filtering used during TRAINING.'horizontal' and 'vertical' doesn't allow to cross respctive board symmetry. 'quarter' only allows the agent to be in the quarter with boxes or targets"""

    # Gifs and
    num_gifs: int = 1
    save_dir: str | None = None
    gif_every: int = 10


agent_config = ml_collections.FrozenConfigDict(
        dict(
            # Agent hyperparameters.
            agent_name='crl',  # Available agent names: crl, crl_search, gciql, gciql_search, gcdqn, clearn_search
            lr=3e-4,  # Learning rate.
            batch_size=256,  # Batch size.
            actor_hidden_dims=(256, 256),  # Actor network hidden dimensions.
            value_hidden_dims=(256, 256),  # Value network hidden dimensions.
            latent_dim=64,
            net_arch='mlp',
            layer_norm=True,  # Whether to use layer normalization.
            discount=0.99,  # Discount factor.
            contrastive_loss = 'binary',
            energy_fn = 'dot',
            logsumexp_coeff = 0.0, # Coefficient for logsumexp loss in critic loss
            actor_loss='awr',  # Actor loss type ('awr' or 'ddpgbc').
            alpha=0.1,  # Temperature in AWR or BC coefficient in DDPG+BC.
            tau=0.005,  # Target network update rate.
            expectile=0.9, # IQL expectile.
            actor_log_q=True,  # Whether to maximize log Q (True) or Q itself (False) in the actor loss.
            const_std=True,  # Whether to use constant standard deviation for the actor.
            discrete=True,  # Whether the action space is discrete.
            encoder=ml_collections.config_dict.placeholder(str),  # Visual encoder name (None, 'impala_small', etc.).
            # Dataset hyperparameters.
            use_next_obs=False, #TODO: This is not used anymore, we should remove it 
            target_entropy_multiplier=0.5,  # Multiplier for the target entropy (used in SAC-like agents).
            target_entropy=-1.1,  # Default target entropy for agents (-ln(|A|/2))
            use_discounted_mc_rewards=False,  # Whether to use discounted Monte Carlo rewards.
            action_sampling='softmax',
            is_td=False,
        )
    )

@dataclass
class Config():
    exp: ExpConfig
    env: legal_envs
    agent: FrozenConfigDict = agent_config



