import os
import fire
import json
from typing import Optional

from utils.prints import print_loud
from utils.template import Template
from utils.mtloader import load_model_and_tokenizer
from utils.generate import generate_fast, generate_interactive
from rome import ROMEHyperParams, apply_rome_to_model

def main(data: str, model: str, config: str, template: Optional[str] = "default"):

    assert os.path.exists(data), "dataset not found"

    with open(data, "r", encoding="utf-8") as f:
        requests = json.load(f)

    queries = [query for request in requests for query in request["queries"]]

    model, tokenizer, batch_first = load_model_and_tokenizer(model)
    template = Template(name=template)

    print_loud("Retrieving hyperparameters")
    hparams = ROMEHyperParams.from_json(config)
    print(hparams)

    if len(queries) > 0:
        print_loud("Generating pre-update text")
        pre_update_text = generate_fast(model, tokenizer, queries, template, max_length=100)
        print("\n\n".join([queries[i] + " " + pre_update_text[i] for i in range(len(queries))]))

    print_loud(f"Applying rome to model")
    model_new, orig_weights = apply_rome_to_model(
        model,
        tokenizer,
        requests,
        hparams,
        template,
        batch_first,
        return_orig_weights=True
    )

    if len(queries) > 0:
        print_loud("Generating post-update text")
        post_update_text = generate_fast(model_new, tokenizer, queries, template, max_length=100)
        print("\n\n".join([queries[i] + " " + post_update_text[i] for i in range(len(queries))]))

    print_loud("Starting interactively generation interface")
    generate_interactive(model_new, tokenizer, template)


if __name__ == "__main__":
    fire.Fire(main)
