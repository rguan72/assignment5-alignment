import torch
from transformers import AutoModelForCausalLM

from cs336_alignment.vllm_utils import VLLMServer

# inputs
training_data = "data/gsm8k/train.jsonl"
validation_data = "data/gsm8k/test.jsonl"

# hyperparameters
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

# load model and datasets
policy = AutoModelForCausalLM.from_pretrained("allenai/OLMo-2-0425-1B")
optimizer = torch.optim.AdamW(
    policy.parameters(), lr=learning_rate, betas=(0.9, 0.95), weight_decay=0.0
)
server = VLLMServer(model_id="allenai/OLMo-2-0425-1B")
server.start()


