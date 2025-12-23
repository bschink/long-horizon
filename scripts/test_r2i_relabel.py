import sys
import os
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np

# Add src to path
project_root = Path(__file__).resolve().parent.parent
src_path = project_root / 'src'
sys.path.insert(0, str(src_path))
sys.path.insert(0, str(project_root)) # For recall2imagine relative imports if needed

# Import r2i
from impls.agents.r2i import GCR2IAgent, load_configs
from recall2imagine import ninjax as nj

import ml_collections

def main():
    print("Testing GCR2IAgent._relabel_data logic...")
    
    # 1. Config
    # Create simple config dict directly to avoid loading issues if file not found
    config = ml_collections.ConfigDict({
        'jax': {'jit': False, 'debug_nans': False},
        'goal_relabel_discount': 0.5, # Low discount to encourage short-term goals
        'reward_constant': 5.0
    })

    # 2. Mock Agent
    # GCR2IAgent is a Wrapped class (JAXAgent). We want to test the inner logic.
    InnerAgentClass = GCR2IAgent.inner
    
    class TestAgent(InnerAgentClass):
        def __init__(self, config):
            self.config = config
            # nj.Module init
            self._path = 'agent'
            self._submodules = {}
            
    agent = TestAgent(config, name='test_agent')
    print("Agent attributes:", dir(agent))
    if hasattr(agent, '_relabel_data'):
        print("YES: _relabel_data exists")
    else:
        print("NO: _relabel_data missing")
        
    # 3. Create Dummy Data
    B, T = 2, 10
    H, W, C = 1, 1, 1 # Minimal spatial dims
    
    # Create distinguishable grids
    # grid[b, t] = t + 100*b (scalar broadcasted)
    grid = np.zeros((B, T, H, W, C), dtype=np.float32)
    for b in range(B):
        for t in range(T):
            grid[b, t] = t + 100 * (b + 1)
            
    is_first = np.zeros((B, T), dtype=bool)
    is_first[:, 0] = True
    # Let's break episode in middle for batch 1
    is_first[1, 5] = True 
    
    # Initial goals (zeros - strictly different from grid values which are >= 100)
    goal = np.zeros_like(grid)
    reward = np.zeros((B, T), dtype=np.float32)
    
    data = {
        'grid': jnp.array(grid),
        'goal': jnp.array(goal),
        'is_first': jnp.array(is_first),
        'reward': jnp.array(reward)
    }
    
    # 4. Run Relabeling
    # Need nj context
    def run_relabel(data):
        return agent._relabel_data(data)
    
    # Wrap with nj.pure to provide context
    run_relabel_pure = nj.pure(run_relabel)
    
    # Run: (out, state) = pure_fn(state, rng, *args)
    init_state = {}
    rng_key = jax.random.PRNGKey(0)
    (relabeled_data, _state) = run_relabel_pure(init_state, rng_key, data)
    
    # 5. Verification
    
    new_goal = np.array(relabeled_data['goal'])
    new_reward = np.array(relabeled_data['reward'])
    
    print("\n--- Verification ---")
    
    success = True

    # Check Batch 0 (Single Episode)
    b = 0
    print(f"\nBatch {b} (Single Episode):")
    for t in range(T):
        orig_val = grid[b, t, 0, 0, 0]
        goal_val = new_goal[b, t, 0, 0, 0]
        
        # Verify goal comes from future or current
        # Valid values are indices + 100*(b+1)
        # If goal_val is 0, it wasn't replaced (masked out or error)
        
        target_t = int(goal_val - 100 * (b + 1))
        
        print(f"t={t}, goal_val={goal_val:.1f} (target_t={target_t}), reward={new_reward[b, t]}")
        
        if goal_val == 0:
            # Should not happen for single episode batch 0 unless geometric sample offset too large > boundary?
            # Clamp logic: target_indices = min(t+offset, seq_len-1).
            # So it should always find a valid future in range [t, T-1].
            print(f"FAILURE: Goal was not updated (remained 0) at t={t}")
            success = False
            continue

        if target_t < t:
            print(f"FAILURE: Goal target {target_t} is in the past of {t}!")
            success = False
        if target_t >= T:
            print(f"FAILURE: Goal target {target_t} out of bounds!")
            success = False
            
        # Verify reward
        expected_reward = 5.0 if target_t == t else 0.0
        if not np.isclose(new_reward[b, t], expected_reward):
            print(f"FAILURE: Reward {new_reward[b, t]} != expected {expected_reward}")
            success = False
            
    # Check Batch 1 (Broken Episode at t=5)
    b = 1
    print(f"\nBatch {b} (Episode break at t=5):")
    # Segments: [0, 1, 2, 3, 4] and [5, 6, 7, 8, 9]
    
    for t in range(T):
        orig_val = grid[b, t, 0, 0, 0]
        goal_val = new_goal[b, t, 0, 0, 0]
        target_t = int(goal_val - 100 * (b + 1))
        
        print(f"t={t}, goal_val={goal_val:.1f} (target_t={target_t}), reward={new_reward[b, t]}")
        
        # Determine segment
        seg = 0 if t < 5 else 1
        target_seg = 0 if target_t < 5 else 1
        
        if goal_val == 0:
            # Goal kept as original (0). This implies invalid relabeling attempt (cross segment).
            # This is expected behavior if geometric sample jumped too far.
            # But wait, if offset=0, it should be valid.
            # So 0 IS possible if offset was large.
            # We just verify that IF it kept 0, it was indeed cross-segment (or we force large offsets to test).
            # With disc=0.5, offsets are small.
            pass
        else:
            # Check validity
            if seg != target_seg:
                 print(f"FAILURE: Goal {target_t} crossed segment boundary from {t}!")
                 success = False
            if target_t < t:
                 print(f"FAILURE: Goal target {target_t} is in the past of {t}!")
                 success = False
    
    if success:
        print("\nALL CHECKS PASSED.")
    else:
        print("\nSOME CHECKS FAILED.")

if __name__ == "__main__":
    main()
