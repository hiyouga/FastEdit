import torch
from typing import Tuple
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    PretrainedConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase
)

def load_model_and_tokenizer(
    model: str, checkpointing: bool
) -> Tuple[PreTrainedModel, PreTrainedTokenizerBase, bool]:

    batch_first = True

    tokenizer = AutoTokenizer.from_pretrained(
        model,
        use_fast=False,
        padding_side="left",
        trust_remote_code=True
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = 0

    config = AutoConfig.from_pretrained(model)

    model = AutoModelForCausalLM.from_pretrained(
        model,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        trust_remote_code=True
    ).cuda()

    # Register auto class to save the custom code files.
    if isinstance(config, PretrainedConfig) and "AutoConfig" in getattr(config, "auto_map", {}):
        config.__class__.register_for_auto_class()
    if isinstance(model, PreTrainedModel) and "AutoModelForCausalLM" in getattr(config, "auto_map", {}):
        model.__class__.register_for_auto_class()
    if isinstance(tokenizer, PreTrainedTokenizerBase) and "AutoTokenizer" in tokenizer.init_kwargs.get("auto_map", {}):
        tokenizer.__class__.register_for_auto_class()

    if checkpointing:
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    return model, tokenizer, batch_first
