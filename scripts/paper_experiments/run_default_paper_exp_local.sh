#!/bin/bash

# Exit immediately if any command fails or Ctrl+C is pressed
set -e
trap 'echo "Interrupted by user"; exit 130' SIGINT

GRID=6
BOX_NUM=4

# Set local paths if SCRATCH is not defined
SAVE_DIR=${SCRATCH:-.}/crl_subgoal/runs
mkdir -p "$SAVE_DIR"

export XLA_PYTHON_CLIENT_PREALLOCATE=false

for KIND in mc td;
do
    for ARCH in deep; #big small
    do
        for AGENT in gcdqn clearn_search crl_search;
        do
            # Logic for flags
            if [ "$AGENT" = "gcdqn" ]; then
                GOALS_FLAG="--exp.use_future_and_random_goals"
            else
                GOALS_FLAG="--exp.no_use_future_and_random_goals"
            fi

            if [ "$KIND" = "mc" ]; then
                KIND_FLAG="--agent.use_discounted_mc_rewards"
                IS_TD_FLAG="--agent.no_is_td"
            else
                KIND_FLAG="--agent.no_use_discounted_mc_rewards"
                IS_TD_FLAG="--agent.is_td"
                if [ "$AGENT" = "crl_search" ]; then continue; fi
            fi

            if [ "$ARCH" = "deep" ]; then
                ARCH_FLAG="res_block"
                SIZE_FLAG="1024 1024 1024 1024 1024 1024 1024 1024 1024 1024 1024 1024 1024 1024 1024 1024 1024 1024 1024 1024 1024 1024"  # 1 inital layer + 1 output layer + 22 ResidualBlocks × 4 Dense layers = 90 layers
            else
                ARCH_FLAG="mlp"
                SIZE_FLAG="256 256"
            fi

            echo "Starting: $KIND $AGENT $ARCH"

            # Run seeds for this specific configuration
            for SEED in 1;
            do
                # use python not uv for dosgpu
                python src/train.py \
                        --agent.agent_name "$AGENT" \
                        --exp.name "$KIND"_"$AGENT"_"$ARCH"_"$BOX_NUM"_boxes_"$GRID"_grid_90layers_512envs_64batch \
                        --exp.gamma 0.99 \
                        --exp.seed "$SEED" \
                        --exp.project long-horizon \
                        --exp.entity uhh_machine_learning \
                        --exp.epochs 50 \
                        --exp.gif_every 10 \
                        --agent.alpha 0.1 \
                        --exp.max_replay_size 10000 \
                        --exp.num_envs 512 \
                        --exp.batch_size 64 \
                        --agent.value_hidden_dims $SIZE_FLAG \
                        --agent.net_arch "$ARCH_FLAG" \
                        --agent.target_entropy "-1.1" \
                        --exp.save_dir "$SAVE_DIR" \
                        $GOALS_FLAG \
                        $IS_TD_FLAG \
                        $KIND_FLAG \
                        env:box-moving \
                        --env.number_of_boxes_min "$BOX_NUM" \
                        --env.number_of_boxes_max "$BOX_NUM" \
                        --env.number_of_moving_boxes_max "$BOX_NUM" \
                        --env.grid_size "$GRID" \
                        --env.episode_length 100 \
                        --env.level-generator default
            done
        done
    done
done