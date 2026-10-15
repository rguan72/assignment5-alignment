import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# rl_utils.py lives at the repo root, which isn't on sys.path when this file is run directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import rl_utils  # noqa: E402
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from cs336_alignment.vllm_utils import VLLMServer


@dataclass
class DataRow:
    question: str
    ground_truth: str

# inputs
training_data_fname = "data/gsm8k/train.jsonl"
validation_data_fname = "data/gsm8k/test.jsonl"
r1_zero_prompt_fname = "cs336_alignment/prompts/r1_zero.prompt"
reward_fn = r1_zero_reward_fn
validation_period = 5

# hyperparameters
seed = 0
n_train_examples = 6400
n_val_examples = 1024
num_rollout_steps = 200
learning_rate = 1e-5
rollout_batch_size = train_batch_size = 256
group_size = 8
gradient_accumulation_steps = 32
sampling_temperature = 1.0
sampling_max_tokens = 512
max_grad_norm = 1.0
# batch sizes count responses, not prompts: 256 rollouts = 32 questions x 8 rollouts each
n_prompts_per_rollout_batch = rollout_batch_size // group_size

# load model and datasets
# train on gpu 0; VLLMServer runs inference on gpu 1 by default
policy_device = "cuda:0"
policy = AutoModelForCausalLM.from_pretrained("allenai/OLMo-2-0425-1B").to(policy_device)
tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-2-0425-1B")
if tokenizer is None:
    raise ValueError("Tokenizer was not initialized correctly and is None.")
optimizer = torch.optim.AdamW(
    policy.parameters(), lr=learning_rate, betas=(0.9, 0.95), weight_decay=0.0
)
server = VLLMServer(model_id="allenai/OLMo-2-0425-1B", seed=seed)
server.start()
server.init_weight_sync(policy_device)

with open(r1_zero_prompt_fname) as pfile:
    prompt = pfile.read()

val_rows = []
with open(validation_data_fname) as vfile:
    for line in vfile:
        if len(val_rows) >= n_val_examples:
            break
        if line:
            data = json.loads(line)
            ground_truth = data["answer"].split("####")[1].strip()
            val_row = DataRow(question=data["question"], ground_truth=ground_truth)
            val_rows.append(val_row)

train_rows = []
with open(training_data_fname) as tfile:
    for line in tfile:
        if len(train_rows) >= n_train_examples:
            break
        if line:
            data = json.loads(line)
            ground_truth = data["answer"].split("####")[1].strip()
            train_row = DataRow(question=data["question"], ground_truth=ground_truth)
            train_rows.append(train_row)
# shuffle once, then each rollout step consumes the next n_prompts_per_rollout_batch questions
# (6400 examples / 32 questions per step = one pass over the data in 200 steps)
random.Random(seed).shuffle(train_rows)

val_sampling_params = {
    "top_p": 1.0,
    "temperature": sampling_temperature,
    "max_tokens": sampling_max_tokens,
    "seed": seed,
    "n": 1,
    "stop": "</answer>",
    "include_stop_str_in_output": True,
}
# vLLM returns the n samples for each prompt contiguously (the group layout
# compute_group_normalized_rewards_grpo expects) and gives each sample its own seed
train_sampling_params = {**val_sampling_params, "n": group_size}

val_prompts = [prompt.format(question=val_row.question) for val_row in val_rows]


def evaluate(step: int) -> None:
    vllm_completions = server.generate_completions(val_prompts, val_sampling_params)
    rewards = [
        reward_fn(vllm_completion.text, val_row.ground_truth)
        for val_row, vllm_completion in zip(val_rows, vllm_completions)
    ]
    val_reward = sum(reward["reward"] for reward in rewards) / len(rewards)
    val_format_reward = sum(reward["format_reward"] for reward in rewards) / len(rewards)
    print(f"step {step}: val reward {val_reward:.3f}, val format reward {val_format_reward:.3f}")


for i in range(num_rollout_steps):
    server.sync_policy_weights(policy)
    if i % validation_period == 0:
        evaluate(i)
    start = (i * n_prompts_per_rollout_batch) % len(train_rows)
    batch_rows = train_rows[start : start + n_prompts_per_rollout_batch]
    prompts = [prompt.format(question=train_row.question) for train_row in batch_rows]
    vllm_completions = server.generate_completions(prompts, train_sampling_params)
    rollouts = [vllm_completion.text for vllm_completion in vllm_completions]
    # repeat each prompt / ground truth group_size times so they line up with the rollouts
    repeated_prompts = [p for p in prompts for _ in range(group_size)]
    repeated_ground_truths = [train_row.ground_truth for train_row in batch_rows for _ in range(group_size)]
    assert len(rollouts) == len(repeated_prompts) == rollout_batch_size
    rl_utils.grpo_train_step(
        policy,
        tokenizer,
        optimizer,
        gradient_accumulation_steps,
        max_grad_norm,
        reward_fn,
        repeated_prompts,
        rollouts,
        repeated_ground_truths,
        group_size,
    )

# evaluate the policy after its final update
server.sync_policy_weights(policy)
evaluate(num_rollout_steps)
