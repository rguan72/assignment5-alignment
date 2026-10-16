"""Standard on-policy GRPO, locally or on two Modal B200s.

Run from the repository root:
    uv run modal run --detach experiments/grpo_experiments_standard_on_policy.py --num-rollout-steps 50

The Modal secret ``wandb`` must contain WANDB_API_KEY. See README.md for setup.
"""

import argparse
import json
import os
import random
import sys
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import modal

# Modal can relocate this entrypoint to /root/<filename>. Do not infer the
# container's project directory from the entrypoint's parent directories.
REPO_ROOT = Path(__file__).resolve().parent.parent if modal.is_local() else Path("/root")
DATA_ROOT = REPO_ROOT / "data" if modal.is_local() else Path("/data")
sys.path.insert(0, str(REPO_ROOT))

# Match the assignment's modal_utils.py environment without requiring its SUNET_ID
# placeholder to be edited. The training implementation also needs root-level rl_utils.py.
app = modal.App("grpo-experiments-standard-on-policy")
image = (
    modal.Image.from_registry("nvidia/cuda:12.9.1-devel-ubuntu22.04", add_python="3.12")
    .uv_sync(uv_project_dir=str(REPO_ROOT), extras=["gpu"])
    .workdir("/root")
    .add_local_dir(REPO_ROOT / "cs336_alignment", "/root/cs336_alignment")
    .add_local_dir(REPO_ROOT / "data", "/data")
    .add_local_dir(REPO_ROOT / "experiments", "/root/experiments")
    .add_local_file(REPO_ROOT / "rl_utils.py", "/root/rl_utils.py")
)


@dataclass
class Config:
    model_id: str = "allenai/OLMo-2-0425-1B"
    training_data_fname: str = "data/gsm8k/train.jsonl"
    validation_data_fname: str = "data/gsm8k/test.jsonl"
    prompt_fname: str = "cs336_alignment/prompts/r1_zero.prompt"
    seed: int = 0
    n_train_examples: int = 6400
    n_val_examples: int = 1024
    num_rollout_steps: int = 200
    learning_rate: float = 1e-5
    rollout_batch_size: int = 256
    group_size: int = 8
    gradient_accumulation_steps: int = 32
    sampling_temperature: float = 1.0
    sampling_max_tokens: int = 512
    max_grad_norm: float = 1.0
    validation_period: int = 5
    rollout_log_period: int = 40
    n_logged_completions: int = 16

    def validate(self) -> None:
        for name in (
            "n_train_examples", "n_val_examples", "num_rollout_steps",
            "rollout_batch_size", "gradient_accumulation_steps", "sampling_max_tokens",
            "validation_period", "rollout_log_period",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.group_size < 2 or self.rollout_batch_size % self.group_size:
            raise ValueError("group_size must be >= 2 and divide rollout_batch_size")
        if self.rollout_batch_size % self.gradient_accumulation_steps:
            raise ValueError("gradient_accumulation_steps must divide rollout_batch_size")
        if self.learning_rate <= 0 or self.max_grad_norm <= 0 or self.sampling_temperature < 0:
            raise ValueError("learning rate/gradient norm must be positive; temperature must be nonnegative")
        if self.n_logged_completions < 0:
            raise ValueError("n_logged_completions must be nonnegative")


@dataclass
class DataRow:
    question: str
    ground_truth: str


def resolve_input_path(filename: str) -> Path:
    path = Path(filename)
    if path.is_absolute():
        return path
    if path.parts and path.parts[0] == "data":
        return DATA_ROOT / path.relative_to("data")
    return REPO_ROOT / path


def load_rows(filename: str, limit: int) -> list[DataRow]:
    rows = []
    with resolve_input_path(filename).open() as infile:
        for line in infile:
            if not line.strip():
                continue
            data = json.loads(line)
            rows.append(DataRow(data["question"], data["answer"].split("####", 1)[1].strip()))
            if len(rows) == limit:
                break
    if not rows:
        raise ValueError(f"No examples found in {filename}")
    return rows


def train(config: Config, wandb_project: str, wandb_entity: str | None = None,
          run_name: str | None = None) -> str | None:
    # Keep GPU/model/W&B initialization out of module import on the submitting laptop.
    import torch
    import wandb

    import rl_utils
    from cs336_alignment.checkpoint import get_model_and_tokenizer
    from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
    from cs336_alignment.vllm_utils import VLLMServer

    config.validate()
    if torch.cuda.device_count() < 2:
        raise RuntimeError("This experiment needs two CUDA GPUs: policy on cuda:0, vLLM on cuda:1")
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    train_rows = load_rows(config.training_data_fname, config.n_train_examples)
    val_rows = load_rows(config.validation_data_fname, config.n_val_examples)
    random.Random(config.seed).shuffle(train_rows)
    prompt = resolve_input_path(config.prompt_fname).read_text()
    val_prompts = [prompt.format(question=row.question) for row in val_rows]
    prompts_per_batch = config.rollout_batch_size // config.group_size
    sampling_params = {
        "top_p": 1.0,
        "temperature": config.sampling_temperature,
        "max_tokens": config.sampling_max_tokens,
        "seed": config.seed,
        "n": 1,
        "stop": "</answer>",
        "include_stop_str_in_output": True,
    }
    server = VLLMServer(model_id=config.model_id, gpu=1, seed=config.seed)
    with wandb.init(
        project=wandb_project,
        entity=wandb_entity,
        name=run_name or f"grpo-standard-on-policy-seed-{config.seed}",
        group="grpo_experiments_standard_on_policy",
        config={**asdict(config), "train_batch_size": config.rollout_batch_size,
                "policy_device": "cuda:0", "inference_gpu": 1,
                "dtype": "bfloat16", "attn_implementation": "flash_attention_2",
                "actual_train_examples": len(train_rows), "actual_val_examples": len(val_rows)},
    ) as run:
        run.define_metric("rollout_step")
        run.define_metric("train/*", step_metric="rollout_step")
        run.define_metric("val/*", step_metric="rollout_step")
        print(f"W&B run: {run.url}", flush=True)

        def completion_table(prompts, completions, ground_truths):
            table = wandb.Table(columns=["prompt", "response", "ground_truth", "reward", "format_reward"])
            for p, completion, truth in list(zip(prompts, completions, ground_truths, strict=True))[:config.n_logged_completions]:
                reward = r1_zero_reward_fn(completion.text, truth)
                table.add_data(p, completion.text, truth, reward["reward"], reward["format_reward"])
            return table

        def evaluate(step: int) -> None:
            completions = server.generate_completions(val_prompts, sampling_params)
            rewards = [r1_zero_reward_fn(completion.text, row.ground_truth)
                       for row, completion in zip(val_rows, completions, strict=True)]
            metrics = {
                "rollout_step": step,
                "val/reward": sum(r["reward"] for r in rewards) / len(rewards),
                "val/format_reward": sum(r["format_reward"] for r in rewards) / len(rewards),
                "val/response_length": sum(len(c.token_ids) for c in completions) / len(completions),
            }
            if config.n_logged_completions:
                metrics["val/completions"] = completion_table(
                    val_prompts, completions, [row.ground_truth for row in val_rows])
            run.log(metrics)
            print(f"step {step}: val reward {metrics['val/reward']:.3f}, "
                  f"val format reward {metrics['val/format_reward']:.3f}", flush=True)

        try:
            # Assignment helper loads directly onto GPU 0 in bf16 with FlashAttention-2.
            policy, tokenizer = get_model_and_tokenizer(config.model_id, "cuda:0")
            policy.train()
            policy.config.use_cache = False
            optimizer = torch.optim.AdamW(
                policy.parameters(), lr=config.learning_rate, betas=(0.9, 0.95), weight_decay=0.0)
            server.start()
            server.init_weight_sync("cuda:0")
            for i in range(config.num_rollout_steps):
                started = time.monotonic()
                server.sync_policy_weights(policy)
                if i % config.validation_period == 0:
                    evaluate(i)
                # Wrap across the dataset boundary while retaining a full response batch.
                start = i * prompts_per_batch
                batch_rows = [train_rows[(start + j) % len(train_rows)] for j in range(prompts_per_batch)]
                prompts = [prompt.format(question=row.question) for row in batch_rows]
                completions = server.generate_completions(prompts, {**sampling_params, "n": config.group_size})
                rollouts = [c.text for c in completions]
                repeated_prompts = [p for p in prompts for _ in range(config.group_size)]
                repeated_truths = [row.ground_truth for row in batch_rows for _ in range(config.group_size)]
                if len(rollouts) != config.rollout_batch_size:
                    raise RuntimeError(f"Expected {config.rollout_batch_size} completions, got {len(rollouts)}")
                loss, metadata = rl_utils.grpo_train_step(
                    policy, tokenizer, optimizer, config.gradient_accumulation_steps,
                    config.max_grad_norm, r1_zero_reward_fn, repeated_prompts, rollouts,
                    repeated_truths, config.group_size)
                metrics = {
                    "rollout_step": i + 1,
                    "train/loss": loss.item(),
                    "train/reward": metadata["mean_total_reward"],
                    "train/format_reward": metadata["mean_format_reward"],
                    "train/grad_norm": metadata["grad_norm"],
                    "train/token_entropy": metadata["token_entropy"],
                    "train/response_length": sum(len(c.token_ids) for c in completions) / len(completions),
                    "train/step_seconds": time.monotonic() - started,
                }
                if config.n_logged_completions and (i == 0 or (i + 1) % config.rollout_log_period == 0):
                    metrics["train/completions"] = completion_table(repeated_prompts, completions, repeated_truths)
                run.log(metrics)
                print(f"step {i + 1}: train reward {metrics['train/reward']:.3f}, "
                      f"loss {metrics['train/loss']:.4f}", flush=True)
            server.sync_policy_weights(policy)
            evaluate(config.num_rollout_steps)
            return run.url
        finally:
            server.stop()


@app.function(
    image=image,
    gpu="B200:2",
    timeout=60 * 60,
    max_containers=4,
    secrets=[modal.Secret.from_name("wandb-secret-2", required_keys=["WANDB_API_KEY"])],
)
def run_on_modal(options: dict) -> str | None:
    return run_from_options(options)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    for field in fields(Config):
        parser.add_argument(f"--{field.name.replace('_', '-')}", type=field.type, default=field.default)
    parser.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT", "cs336-assignment5"))
    parser.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY"))
    parser.add_argument("--run-name")
    return parser


def run_from_options(options: dict) -> str | None:
    options = dict(options)
    config = Config(**{field.name: options.pop(field.name) for field in fields(Config)})
    return train(config, **options)


@app.local_entrypoint()
def main(*argv: str) -> None:
    options = vars(make_parser().parse_args(list(argv)))
    Config(**{field.name: options[field.name] for field in fields(Config)}).validate()
    url = run_on_modal.remote(options)
    print(f"Completed. W&B run: {url}", flush=True)


if __name__ == "__main__":
    run_from_options(vars(make_parser().parse_args()))
