import torch
from copy import deepcopy
from typing import Dict, List, Optional, Tuple
from transformers import PreTrainedModel, PreTrainedTokenizer

from utils import nethook
from utils.context import get_context_templates
from utils.template import Template
from rome.compute_u import compute_u
from rome.compute_v import compute_v
from rome.rome_hparams import ROMEHyperParams


def apply_rome_to_model(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    requests: List[Dict],
    hparams: ROMEHyperParams,
    template: Template,
    batch_first: Optional[bool] = True,
    copy: Optional[bool] = False,
    return_orig_weights: Optional[bool] = False
) -> Tuple[PreTrainedModel, Dict[str, torch.Tensor]]:
    r"""
    Returns a model with the desired changes.

    :param copy: If true, will preserve the original model while creating a new one to edit.
        Note that you are responsible for deallocating the new model's memory to avoid leaks.

    :return: (1) the updated model, (2) an original copy of the weights that changed
    """

    if copy:
        model = deepcopy(model)

    weights_copy = {}

    for i, request in enumerate(requests):
        deltas = execute_rome(model, tokenizer, request, hparams, template, batch_first)

        with torch.no_grad():
            for w_name, (delta_u, delta_v) in deltas.items():
                upd_matrix = delta_u.unsqueeze(1) @ delta_v.unsqueeze(0)
                w = nethook.get_parameter(model, w_name)
                upd_matrix = upd_matrix_match_shape(upd_matrix, w.shape)

                if return_orig_weights and w_name not in weights_copy:
                    assert i == 0
                    weights_copy[w_name] = w.detach().clone()

                w[...] += upd_matrix

        print(f"New weights successfully inserted into {list(deltas.keys())}")

    return model, weights_copy


def execute_rome(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    request: Dict,
    hparams: ROMEHyperParams,
    template: Template,
    batch_first: Optional[bool] = True
) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Executes the ROME update algorithm for the specified update at the specified layer
    Invariant: model at beginning of function == model at end of function
    """

    # Update target and print info
    request = deepcopy(request)
    request["prompt"] = template.get_prompt(request["prompt"])

    print("Executing ROME algorithm for the update: "
          "[{}] -> [{}]".format(request["prompt"].format(request["subject"]), request["target"]))

    context_templates = get_context_templates()

    # Retrieve weights that user desires to change
    weights = {f"{hparams.rewrite_module_tmp.format(layer)}.weight":
               nethook.get_parameter(model, f"{hparams.rewrite_module_tmp.format(layer)}.weight")
               for layer in hparams.layers}

    # Save old weights for future restoration
    weights_copy = {k: v.detach().clone() for k, v in weights.items()}

    # Update loop: sequentially intervene at each specified layer
    deltas = {}
    for layer in sorted(hparams.layers):
        # Compute rank-1 update matrix
        left_vector: torch.Tensor = compute_u(
            model,
            tokenizer,
            request,
            hparams,
            layer,
            context_templates,
            batch_first
        )
        print("Left vector shape:", left_vector.shape)
        right_vector: torch.Tensor = compute_v(
            model,
            tokenizer,
            request,
            hparams,
            layer,
            left_vector,
            context_templates,
            batch_first
        )
        print("Right vector shape:", right_vector.shape)
        right_vector = right_vector.to(torch.float16)

        with torch.no_grad():
            # Determine correct transposition of delta matrix
            weight_name = f"{hparams.rewrite_module_tmp.format(layer)}.weight"
            upd_matrix = left_vector.unsqueeze(1) @ right_vector.unsqueeze(0)
            upd_matrix = upd_matrix_match_shape(upd_matrix, weights[weight_name].shape)

            # Update model weights and record desired changes in `delta` variable
            weights[weight_name][...] += upd_matrix
            deltas[weight_name] = (
                left_vector.detach(),
                right_vector.detach(),
            )

    # Restore state of original model
    with torch.no_grad():
        for k, v in weights.items():
            v[...] = weights_copy[k]

    print(f"Deltas successfully computed for {list(weights.keys())}")

    return deltas


def upd_matrix_match_shape(matrix: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    """
    GPT-2 and GPT-J have transposed weight representations.
    Returns a matrix that matches the desired shape, else raises a ValueError
    """

    if matrix.shape == shape:
        return matrix
    elif matrix.T.shape == shape:
        return matrix.T
    else:
        raise ValueError("Update matrix computed by ROME does not match original weight shape. "
                         "Check for bugs in the code?")
