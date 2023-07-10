import torch
from typing import Dict, List, Optional
from transformers import PreTrainedModel, PreTrainedTokenizer

from .repr_tools import get_reprs_at_idxs, get_reprs_at_word_tokens
from .rome_hparams import ROMEHyperParams


def get_inv_cov(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    layer_name: str,
    mom2_dataset: str,
    mom2_n_samples: str,
    mom2_dtype: str
) -> torch.Tensor:
    r"""
    Retrieves covariance statistics, then computes the algebraic inverse.
    Caches result for future use.
    """

    raise NotImplementedError


def compute_u(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    request: Dict[str, str],
    hparams: ROMEHyperParams,
    layer: int,
    context_templates: List[str],
    batch_first: Optional[bool] = True
) -> torch.Tensor:
    r"""
    Computes the right vector used in constructing the rank-1 update matrix.
    """

    print("Computing left vector (u)...")

    # Compute projection token
    word_repr_args = dict(
        model=model,
        tokenizer=tokenizer,
        layer=layer,
        module_template=hparams.rewrite_module_tmp,
        track="in",
        batch_first=batch_first
    )
    if "subject_" in hparams.fact_token and hparams.fact_token.index("subject_") == 0:
        word = request["subject"]
        print(f"Selected u projection object {word}")
        cur_repr = get_reprs_at_word_tokens(
            context_templates=[templ.format(request["prompt"]) for templ in context_templates],
            words=[word for _ in range(len(context_templates))],
            subtoken=hparams.fact_token[len("subject_"):],
            **word_repr_args
        ).mean(0)
    elif hparams.fact_token == "last":
        # Heuristic to choose last word. Not a huge deal if there's a minor
        # edge case (e.g. multi-token word) because the function below will
        # take the last token.
        cur_repr = get_reprs_at_idxs(
            contexts=[templ.format(request["prompt"].format(request["subject"])) for templ in context_templates],
            idxs=[[-1] for _ in range(len(context_templates))],
            **word_repr_args
        ).mean(0)
        print("Selected u projection token with last token")
    else:
        raise ValueError(f"fact_token={hparams.fact_token} not recognized")

    # Apply inverse second moment adjustment
    u = cur_repr
    if hparams.mom2_adjustment:
        u = get_inv_cov(
            model,
            tokenizer,
            hparams.rewrite_module_tmp.format(layer),
            hparams.mom2_dataset,
            hparams.mom2_n_samples,
            hparams.mom2_dtype
        ) @ u.unsqueeze(1)
        u = u.squeeze()

    return u / u.norm()
