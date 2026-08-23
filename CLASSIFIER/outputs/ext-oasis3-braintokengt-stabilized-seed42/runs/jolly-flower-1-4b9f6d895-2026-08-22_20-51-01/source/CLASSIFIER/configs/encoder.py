"""
Encoder-arm configuration — the knob behind the "is reconstruction pretraining
worth anything?" ablation.

The GAAE is pretrained with a reconstruction objective and its encoder is then
reused as a feature extractor by the downstream classifiers. ``encoder_init``
names *which* encoder the downstream model gets, so the contribution of the
pretraining, of the architecture, and of the encoder itself can be separated:

===========================  ===========================================================
arm                          what it isolates
===========================  ===========================================================
``pretrained_frozen``        Today's default: GAAE weights loaded, encoder frozen.
                             The reference arm — everything else is compared to it.
``pretrained_finetuned``     GAAE weights loaded, gradients allowed. Difference vs
                             ``pretrained_frozen`` = value of adapting the pretrained
                             features to the classification task.
``random``                   Same encoder architecture, random init, trained end-to-end
                             with the classifier. Difference vs ``pretrained_*`` =
                             value of the *reconstruction pretraining itself* (the
                             architecture prior is held constant).
``none``                     No encoder at all; the pooled raw node features go
                             straight into the downstream head. Difference vs
                             ``random`` = value of the graph encoder as such.
===========================  ===========================================================

Back-compat contract
--------------------
``encoder_init=None`` (the default) means "derive the arm from the legacy
``freeze_encoder`` flag", which is exactly what every existing config does. No
existing run changes behaviour unless ``encoder_init`` is set explicitly. When
both knobs are given and they contradict each other, ``resolve_encoder_init``
raises rather than silently picking one (see ``.claude/rules/errors.md``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

EncoderInit = Literal["pretrained_frozen", "pretrained_finetuned", "random", "none"]

#: Declaration order is the order the ablation table reads in.
ENCODER_INIT_ARMS: tuple[str, ...] = (
    "pretrained_frozen",
    "pretrained_finetuned",
    "random",
    "none",
)


@dataclass(frozen=True)
class EncoderArm:
    """Behavioural flags for one ``encoder_init`` value.

    Single source of truth shared by the model (``GELSTMClassifier``) and the
    adapter, so the two can never disagree about what an arm means.

    Attributes
    ----------
    name : str
        The ``encoder_init`` value this describes.
    has_encoder : bool
        Whether a graph encoder module is constructed at all.
    loads_pretrained : bool
        Whether GAAE reconstruction-pretrained weights are loaded into it.
    trains_encoder : bool
        Whether encoder parameters receive gradients from the classification loss.
    """

    name: str
    has_encoder: bool
    loads_pretrained: bool
    trains_encoder: bool


_ARMS: dict[str, EncoderArm] = {
    "pretrained_frozen": EncoderArm(
        name="pretrained_frozen", has_encoder=True, loads_pretrained=True, trains_encoder=False
    ),
    "pretrained_finetuned": EncoderArm(
        name="pretrained_finetuned", has_encoder=True, loads_pretrained=True, trains_encoder=True
    ),
    "random": EncoderArm(
        name="random", has_encoder=True, loads_pretrained=False, trains_encoder=True
    ),
    "none": EncoderArm(
        name="none", has_encoder=False, loads_pretrained=False, trains_encoder=False
    ),
}


def encoder_arm(name: str) -> EncoderArm:
    """Look up the :class:`EncoderArm` for an ``encoder_init`` value.

    Raises ``ValueError`` listing the valid arms for anything else — a typo must
    never fall back to the default arm, or the ablation silently measures the
    wrong thing.
    """
    key = str(name).strip().lower()
    arm = _ARMS.get(key)
    if arm is None:
        raise ValueError(
            f"Unknown encoder_init={name!r}. Valid arms: {list(ENCODER_INIT_ARMS)}. "
            "See DOCS/reconstruction-value-ablation.md."
        )
    return arm


def resolve_encoder_init(
    encoder_init: Optional[str],
    freeze_encoder: Optional[bool] = None,
) -> str:
    """Resolve the effective arm from the new and legacy knobs.

    Parameters
    ----------
    encoder_init : str or None
        The explicit arm. ``None`` means "not set" and selects the legacy
        behaviour below — this is what keeps every pre-ablation config
        bit-for-bit unchanged.
    freeze_encoder : bool or None
        The legacy flag (``None`` = absent from the config, which historically
        meant ``True``). Used only when ``encoder_init`` is ``None``; otherwise
        it is checked for contradictions.

    Returns
    -------
    str — one of :data:`ENCODER_INIT_ARMS`.

    Raises
    ------
    ValueError
        If ``encoder_init`` is unknown, or if both knobs are set and disagree.
    """
    if encoder_init is None:
        # Legacy path: freeze_encoder absent or True -> frozen pretrained encoder,
        # which is what every config in configs/ does today.
        return "pretrained_frozen" if freeze_encoder in (None, True) else "pretrained_finetuned"

    arm = encoder_arm(encoder_init)

    # Only flag a genuine contradiction. freeze_encoder carries no meaning for the
    # encoder-less arm, so it is ignored there rather than treated as a conflict.
    if freeze_encoder is not None and arm.has_encoder:
        if bool(freeze_encoder) is arm.trains_encoder:
            raise ValueError(
                f"Conflicting encoder configuration: encoder_init={arm.name!r} "
                f"(trains_encoder={arm.trains_encoder}) contradicts the legacy "
                f"freeze_encoder={freeze_encoder!r}. Set encoder_init alone and drop "
                "freeze_encoder from the config."
            )
    return arm.name


__all__ = [
    "EncoderArm",
    "EncoderInit",
    "ENCODER_INIT_ARMS",
    "encoder_arm",
    "resolve_encoder_init",
]
