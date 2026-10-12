import json
from dataclasses import dataclass
from typing import Any

import modal
from transformers import AutoModelForCausalLM

from cs336_alignment.drgrpo_grader import question_only_reward_fn, r1_zero_reward_fn
from cs336_alignment.vllm_utils import VLLMServer


@dataclass
class EvalRow:
    prompt: str
    ground_truth: str

app = modal.App("prompting-baselines")
image = (
    modal.Image.from_registry("nvidia/cuda:12.9.1-devel-ubuntu22.04", add_python="3.12")
    .uv_sync(extras=["gpu"])
    .workdir("/root")
    .add_local_dir("cs336_alignment", "/root/cs336_alignment")
    .add_local_dir("data", "/root/data")
)

@app.function(image=image, gpu="B200", timeout=60 * 60)
def main() -> dict[str, dict[str, Any]]:
    validation_data_fname = "data/gsm8k/test.jsonl"
    r1_zero_three_shot_fname = "cs336_alignment/prompts/r1_zero_three_shot_gsm8k.prompt"
    r1_zero_prompt_fname = "cs336_alignment/prompts/r1_zero.prompt"
    question_only_prompt_fname = "cs336_alignment/prompts/question_only.prompt"

    prompt_templates = {}
    prompt_names = ["r1_zero_three_shot", "r1_zero", "question_only"]
    fnames = [r1_zero_three_shot_fname, r1_zero_prompt_fname, question_only_prompt_fname]
    for name, fname in zip(prompt_names, fnames):    
        with open(fname) as pfile:
            prompt_templates[name] = pfile.read()

    eval_rows = []
    with open(validation_data_fname) as vfile:
        for line in vfile:
            if line:
                data = json.loads(line)
                ground_truth = data["answer"].split("####")[1].strip()
                question = data["question"]
                prompt = prompt_templates["r1_zero_three_shot"].format(question=question)
                eval_row = EvalRow(prompt=prompt, ground_truth=ground_truth)
                eval_rows.append(eval_row)

    server = VLLMServer(model_id="allenai/OLMo-2-0425-1B", gpu=0)
    server.start()

    sampling_params = {
        "top_p": 1.0,
        "temperature": 1.0,
        "max_generation_length": 512,
    }
    results: dict[str, dict[str, Any]] = {
        "r1_zero": {
            "full_reward": 0,
            "format_reward": 0,
            "answer_reward": 0,
            "no_reward": 0,
            "text_completions": [],
        },
        "r1_zero_three_shot": {
            "full_reward": 0,
            "format_reward": 0,
            "answer_reward": 0,
            "no_reward": 0,
            "text_completions": [],
        },
        "question_only": {
            "full_reward": 0,
            "format_reward": 0,
            "answer_reward": 0,
            "no_reward": 0,
            "text_completions": [],
        }
    }

    for prompt_name in prompt_names:
        if prompt_name in ["r1_zero", "r1_zero_three_shot"]:
            sampling_params['stop'] = ["</answer>"]
            sampling_params['include_stop_str_in_output'] = True
            reward_fn = r1_zero_reward_fn
        else:
            reward_fn = question_only_reward_fn
        prompts = [eval_row.prompt for eval_row in eval_rows]
        vllm_completions = server.generate_completions(prompts, sampling_params)
        for eval_row, vllm_completion in zip(eval_rows, vllm_completions):
            ground_truth = eval_row.ground_truth
            completion = vllm_completion.text
            reward = reward_fn(ground_truth, completion)
            if reward["format_reward"] == 1.0 and reward["answer_reward"] == 1.0:
                results[prompt_name]["full_reward"] += 1
            elif reward["format_reward"] == 0.0 and reward["answer_reward"] == 1.0:
                results[prompt_name]["answer_reward"] += 1
            elif reward["format_reward"] == 1.0 and reward["answer_reward"] == 0.0:
                results[prompt_name]["format_reward"] += 1
            else:
                results[prompt_name]["no_reward"] += 1
            results[prompt_name]["text_completions"].append(completion)
    return results

@app.local_entrypoint()
def save_results() -> None:
    # save results to a file
    with open("prompting_baselines_results.json", "w") as f:
        json.dump(main.remote(), f, indent=2)
