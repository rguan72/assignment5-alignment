from collections.abc import Callable
from typing import Literal

import torch
from torch.nn import utils
from torch.nn.utils import clip_grad_norm_
from torch.optim import Optimizer
from transformers import PreTrainedModel, PreTrainedTokenizerBase


def tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer: PreTrainedTokenizerBase
) -> dict[str, torch.Tensor]:
    prompt_tokens: list[list[int]] = [tokenizer.encode(prompt) for prompt in prompt_strs]
    output_tokens: list[list[int]] = [tokenizer.encode(output) for output in output_strs]
    input_ids_unpadded = [torch.tensor(prompt + output, dtype=torch.long) for (prompt, output) in zip(prompt_tokens, output_tokens)]
    response_mask_unpadded = [torch.tensor([0] * len(prompt) + [1] * len(output)) for (prompt, output) in zip(prompt_tokens, output_tokens)]
    input_ids = utils.rnn.pad_sequence(input_ids_unpadded, batch_first=True, padding_value=0)[..., :-1]
    labels = utils.rnn.pad_sequence(input_ids_unpadded, batch_first=True, padding_value=0)[..., 1:]
    response_mask = utils.rnn.pad_sequence(response_mask_unpadded, batch_first=True, padding_value=0)[..., 1:]
    return {
        "input_ids": input_ids,
        "labels": labels,
        "response_mask": response_mask
    }

def get_response_log_probs(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool = False,
) -> dict[str, torch.Tensor]:
    logits = model(input_ids).logits # b x t x c
    # labels: b x t: already aligned with logits
    # use labels to index into logits for log_softmax computation (label selects numerator in softmax)
    logsoftmax = torch.log_softmax(logits, dim=-1)
    log_probs = torch.gather(input=logsoftmax, index=labels.unsqueeze(-1), dim=-1).squeeze(-1) # b x t
    if not return_token_entropy:
        return { "log_probs": log_probs }
    # -> that gives us our b x t response log probs
    # token entropy: can be computed directly on b x t x c logits
    softmax = torch.exp(logsoftmax)
    token_entropy = torch.sum(-softmax * logsoftmax, dim=-1)
    return { "log_probs": log_probs, "token_entropy": token_entropy }
    # output will be b x t: we need softmax and log softmax over channel dimension, multiply them elementwise, sum over channel dim

def compute_rollout_rewards(
    reward_fn: Callable[[str, str], dict[str, float]],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
) -> tuple[torch.Tensor, dict[str, float]]:
    # for each idx i, call reward_fn((rollout_responses[i], repeated_ground_truths[i])) -> {} of reward
    # [{} of reward] -> flatten on "reward" key, convert to 1D tensor (first return val)
    # [{} of reward] -> flatten on other keys and compute statistics 
    reward_details: list[dict[str, float]] = [reward_fn(rollout_response, ground_truth) for (rollout_response, ground_truth) in zip(rollout_responses, repeated_ground_truths)]
    raw_rewards = torch.tensor([reward_detail["reward"] for reward_detail in reward_details])
    mean_total_reward = torch.mean(raw_rewards).item()
    mean_format_reward = torch.mean(torch.tensor([reward_detail["format_reward"] for reward_detail in reward_details])).item()
    metadata = { "mean_total_reward": mean_total_reward, "mean_format_reward": mean_format_reward }
    return (raw_rewards, metadata)

def compute_group_normalized_rewards_grpo(
    raw_rewards: torch.Tensor,
    group_size: int,
    baseline: Literal["mean", "none"] = "mean",
    advantage_eps: float = 1e-6,
    advantage_normalizer: Literal["std", "none", "mean"] = "std",
) -> tuple[torch.Tensor, dict[str, float]]:
    if (advantage_normalizer in ("mean")):
        raise NotImplementedError
    # rightmost varies fastest, so layout a_1 a_2 ... a_g b_1 b_2 ... b_g c_1 c_2 ... c_g
    raw_rewards_grouped = raw_rewards.reshape(-1, group_size) # b x g
    if baseline == "mean":
        reward_baselines = torch.mean(raw_rewards_grouped, dim=-1, keepdim=True) # b x 1
    else:
        reward_baselines = 0
    if advantage_normalizer == "std":
        reward_normalizers = torch.std(raw_rewards_grouped, dim=-1, keepdim=True) + advantage_eps # b x 1
    elif advantage_normalizer == "none":
        reward_normalizers = 1
    rewards_grouped = (raw_rewards_grouped - reward_baselines) / reward_normalizers # b x g
    rewards = rewards_grouped.flatten()
    mean_reward = torch.mean(raw_rewards).item()
    std_reward = torch.std(raw_rewards).item()
    max_reward = torch.max(raw_rewards).item()
    min_reward = torch.min(raw_rewards).item()
    return (rewards, {
        "mean_reward": mean_reward,
        "std_reward": std_reward,
        "max_reward": max_reward,
        "min_reward": min_reward,
    })
    
def compute_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    importance_reweighting_method: Literal["none", "noclip", "grpo", "gspo"] = "none",
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
    response_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if (importance_reweighting_method == "gspo") or (response_mask is not None):
        raise NotImplementedError
    raw_rewards_or_advantages = raw_rewards_or_advantages.reshape(raw_rewards_or_advantages.shape[0], 1) # b x 1
    if importance_reweighting_method in ("grpo", "noclip"):
        assert old_log_probs is not None, "need to pass old_log_probs for importance reweighting"
        importance_weights = torch.exp(policy_log_probs) / torch.exp(old_log_probs) # b x t
        if importance_reweighting_method == "noclip":
            per_token_policy_gradient_loss = raw_rewards_or_advantages * importance_weights
        else:
            assert cliprange is not None, "need to pass cliprange for grpo importance reweighting"
            per_token_policy_gradient_loss = torch.min(raw_rewards_or_advantages * importance_weights, raw_rewards_or_advantages * torch.clip(importance_weights, min=1-cliprange, max=1+cliprange))
    elif importance_reweighting_method == "none":
        per_token_policy_gradient_loss = raw_rewards_or_advantages * policy_log_probs # b x t
    return (-per_token_policy_gradient_loss, {}) #  negative for gradient ascent,

def aggregate_loss_across_microbatch(
    per_token_policy_gradient_loss: torch.Tensor, # b x t
    mask: torch.Tensor, # b x t
    loss_normalization: Literal["sequence", "constant"] = "sequence",
    normalization_constant: int | None = None,
) -> torch.Tensor:
    sum_over_seq = torch.sum(per_token_policy_gradient_loss * mask, dim=-1)
    if loss_normalization == "sequence":
        count_over_seq = torch.sum(mask, dim=-1)
        return torch.mean(sum_over_seq / count_over_seq)
    assert normalization_constant is not None, "normalization constant must be provided when loss normalization = constant"
    # normalization constant already BGL so no need to do means over batch dimension
    return torch.sum(sum_over_seq / normalization_constant)

def grpo_train_step(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    optimizer: Optimizer,
    gradient_accumulation_steps: int,
    max_grad_norm: float | None,
    reward_fn: Callable[[str, str], dict[str, float]],
    repeated_prompts: list[str],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    # Reward normalization
    baseline: Literal["mean", "none"] = "mean",
    advantage_eps: float = 1e-6,
    advantage_normalizer: Literal["std", "none", "mean"] = "std",
    # Importance reweighting and clipping
    importance_reweighting_method: Literal["none", "noclip", "grpo", "gspo"] = "none",
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
    # Loss normalization
    loss_normalization: Literal["sequence", "constant"] = "sequence",
    normalization_constant: int | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor | float]]:
    if (baseline != "mean") or (advantage_normalizer != "std") or (importance_reweighting_method != "none") or (old_log_probs is not None) or (cliprange is not None) or (loss_normalization != "sequence") or (normalization_constant is not None):
        raise NotImplementedError
    # tokenize
    # optimize over mini batch
        # sample
        # compute reward
        # normalize reward
        # compute policy gradient
    tokenized = tokenize_prompt_and_output(repeated_prompts, rollout_responses, tokenizer)
    input_ids = tokenized["input_ids"]
    labels = tokenized["labels"]
    response_mask = tokenized["response_mask"]
    raw_rewards, rollout_metadata = compute_rollout_rewards(reward_fn, rollout_responses, repeated_ground_truths)
    normalized_rewards, normalized_metadata = compute_group_normalized_rewards_grpo(raw_rewards, group_size, baseline, advantage_eps, advantage_normalizer)
    batch_size = len(repeated_prompts)
    mini_batch_size = batch_size // gradient_accumulation_steps
    aggregate_pg_loss = 0.0
    for i in range(0, batch_size, mini_batch_size):
        true_mini_batch_size = min(mini_batch_size, batch_size - i)
        mini_batch_input_ids = input_ids[i:i+true_mini_batch_size]
        mini_batch_labels = labels[i:i+true_mini_batch_size]
        mini_batch_response_mask = response_mask[i:i+true_mini_batch_size]
        policy_logprobs = get_response_log_probs(model, mini_batch_input_ids, mini_batch_labels)["log_probs"]
        per_token_policy_gradient_loss, pg_metadata = compute_policy_gradient_loss(normalized_rewards[i:i+true_mini_batch_size], policy_logprobs, importance_reweighting_method, old_log_probs, cliprange, response_mask=None)
        per_token_policy_gradient_loss_scaled = per_token_policy_gradient_loss * true_mini_batch_size / batch_size
        aggregate_microbatch_pg_loss = aggregate_loss_across_microbatch(per_token_policy_gradient_loss_scaled, mini_batch_response_mask, loss_normalization, normalization_constant)
        aggregate_pg_loss += aggregate_microbatch_pg_loss
        aggregate_microbatch_pg_loss.backward()
    # do we accumulate aggregate_pg_loss over batch? instead of setting it
    if max_grad_norm is not None:
        clip_grad_norm_(model.parameters(), max_grad_norm)
    optimizer.step()
    optimizer.zero_grad()
    return (torch.tensor(aggregate_pg_loss), {})
    
        
    