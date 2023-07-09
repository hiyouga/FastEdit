import torch
from typing import Tuple
from transformers import (
    AutoConfig,
    AutoModel,
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

    config = AutoConfig.from_pretrained(model, trust_remote_code=True)

    if hasattr(config, "auto_map") and "AutoModelForCasualLM" not in getattr(config, "auto_map"):
        AutoClass = AutoModel
        if "ChatGLM" in getattr(config, "auto_map")["AutoModel"]:
            batch_first = False
    else:
        AutoClass = AutoModelForCausalLM

    model = AutoClass.from_pretrained(
        model,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        trust_remote_code=True
    ).cuda()

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.unk_token_id

    return model, tokenizer, batch_first
