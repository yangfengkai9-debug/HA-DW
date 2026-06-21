#!/usr/bin/env bash
set -xeuo pipefail

# Example:
# bash examples/hadw_trainer/run_hadw.sh
#
# Optional overrides:
# MODEL_PATH=Qwen/Qwen2.5-3B-Instruct \
# TRAIN_FILE=/data/gsm8k/train.parquet \
# VAL_FILE=/data/gsm8k/test.parquet \
# REWARD_WINDOW_K=32 \
# bash examples/hadw_trainer/run_hadw.sh

MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-4B-Base}
TRAIN_FILE=${TRAIN_FILE:-/data/gsm8k/train.parquet}
VAL_FILE=${VAL_FILE:-/data/gsm8k/test.parquet}
REWARD_WINDOW_K=${REWARD_WINDOW_K:-10}

adv_estimator=ha-dw
loss_mode=grpo
rollout_engine=vllm
rollout_mode=async

train_batch_size=256
n_resp_per_prompt=8
ppo_mini_batch_size=16
ppo_micro_batch_size_per_gpu=4
max_prompt_length=2048
max_response_length=4096
total_epochs=3
total_training_steps=200
test_freq=20
save_freq=20

project_name='RL-GRPO-HA-DW'
exp_name="hadw-grpo-$(basename "${MODEL_PATH}")"
CKPTS_DIR=${CKPTS_DIR:-"./ckpts/${exp_name}"}

export VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}
export HYDRA_FULL_ERROR=1

python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=${adv_estimator} \
  actor_rollout_ref.actor.policy_loss.loss_mode=${loss_mode} \
  trainer.reward_window_k=${REWARD_WINDOW_K} \
  data.train_files="['${TRAIN_FILE}']" \
  data.val_files="['${VAL_FILE}']" \
  data.prompt_key=prompt \
  data.truncation=error \
  data.filter_overlong_prompts=true \
  data.train_batch_size=${train_batch_size} \
  data.max_prompt_length=${max_prompt_length} \
  data.max_response_length=${max_response_length} \
  actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
  actor_rollout_ref.rollout.name=${rollout_engine} \
  actor_rollout_ref.rollout.mode=${rollout_mode} \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size} \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${ppo_micro_batch_size_per_gpu} \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.actor.grad_clip=1.0 \
  reward.reward_manager.name=dapo \
  trainer.project_name="${project_name}" \
  trainer.experiment_name="${exp_name}" \
  trainer.logger='["console"]' \
  trainer.val_before_train=false \
  trainer.test_freq=${test_freq} \
  trainer.save_freq=${save_freq} \
  trainer.total_epochs=${total_epochs} \
  trainer.total_training_steps=${total_training_steps} \
  trainer.default_local_dir="${CKPTS_DIR}" \
  "$@"
