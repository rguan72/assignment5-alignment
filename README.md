# CS336 Spring 2026 Assignment 5: Alignment

For a full description of the assignment, see the assignment handout at
[cs336_spring2026_assignment5_alignment.pdf](./cs336_spring2026_assignment5_alignment.pdf)

We will include a supplemental (and completely optional) assignment on safety alignment, instruction tuning, and RLHF at [cs336_spring2026_assignment5_supplement_safety_rlhf.pdf](./cs336_spring2026_assignment5_supplement_safety_rlhf.pdf)

If you see any issues with the assignment handout or code, please feel free to
raise a GitHub issue or open a pull request with a fix.

## Setup

As in previous assignments, we use `uv` to manage dependencies.

1. Install all packages except `flash-attn`, then all packages (`flash-attn` is weird)
```
uv sync --no-install-package flash-attn
uv sync
```

2. Run the required unit tests:

``` sh
uv run pytest tests/test_grpo.py
```

Initially, all tests should fail with `NotImplementedError`s.
To connect your implementation to the tests, complete the
functions in [./tests/adapters.py](./tests/adapters.py).

## Standard on-policy GRPO on Modal

The launcher in `experiments/grpo_experiments_standard_on_policy.py` requests
**two B200 GPUs**: the policy/optimizer use GPU 0 and vLLM uses GPU 1. It matches
the provided Modal helper's CUDA 12.9.1, Python 3.12, GPU dependencies, and one-hour
timeout. The policy uses the assignment's bfloat16/FlashAttention-2 model loader.
This standalone app does not require changing `SUNET_ID` in `modal_utils.py`.

Authenticate Modal once with `uv run modal setup`. In the Modal dashboard, create
a secret named **`wandb`** containing **`WANDB_API_KEY`** for your W&B account.
The GPU image already installs W&B via the `gpu` extra.

From the repository root, start the assignment's initial 50-step check:

```sh
uv run modal run --detach experiments/grpo_experiments_standard_on_policy.py \
  --num-rollout-steps 50 \
  --wandb-project cs336-assignment5 \
  --run-name grpo-standard-seed-0-check
```

For the full 200-step run, omit `--num-rollout-steps`. Use `--seed 1` (etc.) for
independent runs and `--wandb-entity YOUR_TEAM` to select a W&B team. The defaults
retain 6,400 training examples, 1,024 validation examples, 256 responses per batch,
group size 8, and 32 gradient accumulation steps. All configuration options are
listed by `uv run python experiments/grpo_experiments_standard_on_policy.py --help`.
Data/prompt paths must refer to files included in the image. Local `data/` is
mounted at `/data`, so both `data/gsm8k/train.jsonl` and
`/data/gsm8k/train.jsonl` work on Modal. The validation file is mounted at
`/data/gsm8k/test.jsonl`. Code and prompts are mounted under
`/root/cs336_alignment`, experiments under `/root/experiments`, and the training
utilities at `/root/rl_utils.py`. Container paths do not depend on where Modal
places the entrypoint script.

Open the W&B URL printed in the Modal logs to see `train/reward`,
`train/format_reward`, `val/reward`, and `val/format_reward` plotted against
`rollout_step` (completed optimizer updates). Training rewards describe the batch
used for that update. Validation runs before training, every five updates, and
after the final update. Also logged: loss, gradient norm before clipping,
response-token entropy, response lengths in tokens, step duration, and the full
configuration. Completion tables include up to 16 examples at every validation
and at the first/every 40th training batch. Set `--n-logged-completions 0` to disable
tables. A run closes W&B and stops vLLM on success or a Python exception.

`--detach` lets the remote job continue after the submitting terminal disconnects.
The function has a one-hour limit; it logs progress but does not save checkpoints
or resume interrupted training. This launcher does not start a sweep automatically.

On an existing two-GPU Linux machine, the same script can run directly with
`uv run --extra gpu python experiments/grpo_experiments_standard_on_policy.py` after
configuring W&B authentication locally.

