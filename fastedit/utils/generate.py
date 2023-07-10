import torch
from typing import List, Optional
from transformers import PreTrainedModel, PreTrainedTokenizer, TextStreamer

from .template import Template


def generate_interactive(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    template: Template,
    top_k: Optional[int] = 50,
    max_length: Optional[int] = 200
):
    r"""
    Puts generation in a loop. Allows users to repeatedly provide inputs
    with which text is generated.
    """

    print("Enter `exit` to exit the interface.")

    while True:
        query = input("Input: ").strip()

        if query == "exit":
            break

        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        print("Output: ", end="", flush=True)
        generate_fast(model, tokenizer, [query], template, top_k=top_k, max_length=max_length, streamer=streamer)[0]
        print()


def generate_fast(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    queries: List[str],
    template: Template,
    n_gen_per_prompt: Optional[int] = 1,
    top_k: Optional[int] = 50,
    max_length: Optional[int] = 200,
    streamer: Optional[TextStreamer] = None
):
    r"""
    Fast, parallelized auto-regressive text generation with top-k sampling.
    Our custom implementation.
    """

    # Unroll prompts and tokenize
    inp = [template.get_prompt(query) for query in queries for _ in range(n_gen_per_prompt)]
    inp_tok = tokenizer(inp, padding=True, return_token_type_ids=False, return_tensors="pt").to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            **inp_tok,
            temperature=0.1,
            top_k=top_k,
            max_length=max_length,
            do_sample=True,
            streamer=streamer
        )

    responses = tokenizer.batch_decode(
        generated_ids[:, inp_tok["input_ids"].size(1):],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True
    )

    return responses
