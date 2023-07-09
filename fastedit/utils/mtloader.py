import torch
from typing import Tuple
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer
)

def load_model_and_tokenizer(model: str) -> Tuple[PreTrainedModel, PreTrainedTokenizer, bool]:

    batch_first = True

    tokenizer = AutoTokenizer.from_pretrained(
        model,
        use_fast=False,
        padding_side="left",
        trust_remote_code=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        model,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        trust_remote_code=True
    ).cuda()

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = 0

    return model, tokenizer, batch_first
