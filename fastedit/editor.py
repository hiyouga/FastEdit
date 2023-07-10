import os
import fire
import json
from typing import Optional

from .rome import ROMEHyperParams, apply_rome_to_model
from .utils.prints import print_loud
from .utils.template import Template
from .utils.mtloader import load_model_and_tokenizer
from .utils.generate import generate_fast, generate_interactive


def test_rome(data: str, model: str, config: str, template: Optional[str] = "default") -> None:
    r"""
    Edits a pre-trained model using model-editing algorithms.

    Args:
        data (`str`):
            The path of the `json` file containing the samples for editing.
        model (`str`):
            The name or path of the pre-trained transformer model to be edited.
        config (`str`):
            The name of the hyper-parameters to use for editing the model.
        template (`str`, *optional*, defaults to `default`):
            The name of the template to use in generation.

    Returns:
        diff_weights (`Dict[str, Tensor]`):
            A dict of diff weights that have been changed.
    """

    assert os.path.exists(data), "data not found"

    with open(data, "r", encoding="utf-8") as f:
        requests = json.load(f)

    queries = [query for request in requests for query in request["queries"]]

    model, tokenizer, batch_first = load_model_and_tokenizer(model)
    template = Template(name=template)

    print_loud("Retrieving hyperparameters")
    hparams = ROMEHyperParams.from_name(config)
    print(hparams)

    if len(queries) > 0:
        print_loud("Generating pre-update text")
        pre_update_text = generate_fast(model, tokenizer, queries, template, max_length=100)
        print("\n\n".join([queries[i] + " " + pre_update_text[i] for i in range(len(queries))]))

    print_loud(f"Applying rome to model")
    model_new, _ = apply_rome_to_model(
        model,
        tokenizer,
        requests,
        hparams,
        batch_first,
        return_diff_weights=False
    )

    if len(queries) > 0:
        print_loud("Generating post-update text")
        post_update_text = generate_fast(model_new, tokenizer, queries, template, max_length=100)
        print("\n\n".join([queries[i] + " " + post_update_text[i] for i in range(len(queries))]))

    print_loud("Starting interactively generation interface")
    generate_interactive(model_new, tokenizer, template)


if __name__ == "__main__":
    fire.Fire(test_rome)
