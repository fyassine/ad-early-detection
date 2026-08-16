"""Tests for the reconstruction-value ablation's encoder arms.

Covers the config-level arm resolution (including the back-compat contract and
the loud failure on contradictory knobs) and the model-level behaviour of each
arm on tiny synthetic graphs: it builds, shapes flow end-to-end, and the
gradient story is what the arm claims it is.

See DOCS/reconstruction-value-ablation.md.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch_geometric.data import Data

from CLASSIFIER.configs.encoder import (
    ENCODER_INIT_ARMS,
    encoder_arm,
    resolve_encoder_init,
)
from CLASSIFIER.configs.gelstm import EvalConfig, GELSTMTrainConfig
from CLASSIFIER.model.GELSTM.models import GELSTMClassifier
from CLASSIFIER.model.GELSTM.train import _eval_cfg_to_dict
from CLASSIFIER.model.GELSTM.utils import encode_batch_sequences
from SHARED.seeding import set_seed

IN_FEATURES = 6
GAAE_HIDDEN = 4
GAAE_LATENT = 3
N_NODES = 5
SEED = 42


# --------------------------------------------------------------------------- #
# Config-level arm resolution
# --------------------------------------------------------------------------- #
def test_default_config_leaves_encoder_init_unset():
    """The dataclass default must not change any existing run's behaviour."""
    assert GELSTMTrainConfig().encoder_init is None
    assert EvalConfig().encoder_grad is False


@pytest.mark.parametrize(
    "freeze_encoder, expected",
    [
        (None, "pretrained_frozen"),  # key absent from the JSON config
        (True, "pretrained_frozen"),  # what every config in configs/ says today
        (False, "pretrained_finetuned"),
    ],
)
def test_legacy_freeze_encoder_maps_to_the_same_arm(freeze_encoder, expected):
    assert resolve_encoder_init(None, freeze_encoder) == expected


@pytest.mark.parametrize("arm", ENCODER_INIT_ARMS)
def test_explicit_arm_wins_when_no_legacy_flag(arm):
    assert resolve_encoder_init(arm, None) == arm


@pytest.mark.parametrize(
    "arm, freeze_encoder",
    [
        ("pretrained_frozen", False),
        ("pretrained_finetuned", True),
        ("random", True),
    ],
)
def test_contradictory_knobs_raise(arm, freeze_encoder):
    with pytest.raises(ValueError, match="Conflicting encoder configuration"):
        resolve_encoder_init(arm, freeze_encoder)


def test_freeze_encoder_is_irrelevant_for_the_encoderless_arm():
    """``none`` has nothing to freeze, so a stale freeze_encoder is not a conflict."""
    assert resolve_encoder_init("none", True) == "none"
    assert resolve_encoder_init("none", False) == "none"


def test_unknown_arm_raises_and_lists_the_valid_ones():
    with pytest.raises(ValueError, match="Unknown encoder_init"):
        encoder_arm("pretrained")  # plausible typo, must not fall back


def test_eval_cfg_roundtrip_preserves_encoder_grad():
    """_eval_cfg_to_dict feeds EvalConfig(**...) rebuilds — encoder_grad must survive."""
    cfg = EvalConfig(encoder_grad=True)
    assert _eval_cfg_to_dict(cfg)["encoder_grad"] is True
    assert EvalConfig(**_eval_cfg_to_dict(cfg)).encoder_grad is True


# --------------------------------------------------------------------------- #
# Model-level behaviour
# --------------------------------------------------------------------------- #
def _make_model(arm: str, *, use_time_delta: bool = True) -> GELSTMClassifier:
    # Seed the weight init here, not in the model (see .claude/rules/seeding.md):
    # a tiny randomly-initialised GAT can land on an all-dead-ReLU draw whose
    # gradients are legitimately zero, which would make the grad assertions below
    # flaky — and pytest-randomly reorders tests, so ambient RNG state is never
    # reproducible. One explicit seed per model makes every test order-independent.
    set_seed(SEED)
    return GELSTMClassifier(
        in_features=IN_FEATURES,
        gaae_hidden=GAAE_HIDDEN,
        gaae_latent=GAAE_LATENT,
        gaae_heads=1,
        gaae_cond_dim=2,
        gaae_dropout=0.0,
        lstm_hidden=4,
        lstm_layers=1,
        lstm_dropout=0.0,
        use_time_delta=use_time_delta,
        classifier_hidden=4,
        encoder_init=arm,
    )


def _make_graph(rng: np.random.Generator) -> Data:
    """A tiny fully-connected-ish graph with the edge_attr the GAT layers expect."""
    src, dst = [], []
    for i in range(N_NODES):
        for j in range(N_NODES):
            if i != j:
                src.append(i)
                dst.append(j)
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    return Data(
        x=torch.tensor(rng.normal(size=(N_NODES, IN_FEATURES)), dtype=torch.float),
        edge_index=edge_index,
        edge_attr=torch.tensor(rng.uniform(size=(edge_index.shape[1], 1)), dtype=torch.float),
    )


def _make_batch(rng: np.random.Generator, n_subjects: int = 2) -> list[dict]:
    batch = []
    for s in range(n_subjects):
        n_visits = 2 + s  # different lengths so packing is exercised
        batch.append(
            {
                "subject_id": f"sub-{s}",
                "graphs": [_make_graph(rng) for _ in range(n_visits)],
                "delta_t": [0.0] + [0.25] * (n_visits - 1),
                "visit_months": list(range(n_visits)),
                "label": float(s % 2),
                "n_scans": n_visits,
            }
        )
    return batch


def _forward_backward(model: GELSTMClassifier, *, encoder_grad: bool) -> torch.Tensor:
    """One train-mode forward+backward pass; returns the logits."""
    rng = np.random.default_rng(0)
    batch = _make_batch(rng)
    model.train()
    packed, labels, _ = encode_batch_sequences(
        batch,
        model,
        device=torch.device("cpu"),
        use_time_delta=model.use_time_delta,
        encoder_grad=encoder_grad,
    )
    logits = model(packed)
    torch.nn.BCEWithLogitsLoss()(logits, labels).backward()
    return logits


@pytest.mark.parametrize("arm", ENCODER_INIT_ARMS)
def test_every_arm_builds_and_shapes_flow_end_to_end(arm):
    model = _make_model(arm)
    expected_embed = IN_FEATURES if arm == "none" else GAAE_LATENT
    assert model.embed_dim == expected_embed
    # The head's input width is derived from the arm, never hardcoded.
    assert model.lstm_input_dim == expected_embed + 1
    assert model.lstm.input_size == expected_embed + 1
    assert model.feat_mean.shape == (expected_embed,)

    logits = _forward_backward(model, encoder_grad=model.encoder_arm.trains_encoder)
    assert logits.shape == (2,)  # one logit per subject
    assert torch.isfinite(logits).all()


def test_arm_none_has_no_encoder_parameters():
    model = _make_model("none")
    assert model.encoder is None
    assert model.encoder_modules() == []
    encoder_params = [n for n, _ in model.named_parameters() if n.startswith("encoder")]
    assert encoder_params == []
    # ... and the arms that do have one are not accidentally empty.
    assert [n for n, _ in _make_model("random").named_parameters() if n.startswith("encoder")]


def test_arm_none_embeds_the_pooled_raw_node_features():
    """No encoder means the per-visit embedding IS the pooled input row."""
    model = _make_model("none")
    g = _make_graph(np.random.default_rng(1))
    z = model.encode_visit(g.x, g.edge_index, g.edge_attr, pool="mean")
    torch.testing.assert_close(z, g.x.mean(dim=0))


def test_pretrained_frozen_has_zero_encoder_grads_after_backward():
    model = _make_model("pretrained_frozen")
    model.freeze_encoder()
    _forward_backward(model, encoder_grad=False)

    enc_params = [p for m in model.encoder_modules() for p in m.parameters()]
    assert enc_params, "frozen arm should still own encoder parameters"
    assert all(not p.requires_grad for p in enc_params)
    assert all(p.grad is None or torch.count_nonzero(p.grad) == 0 for p in enc_params)
    # The head still learns.
    assert any(
        p.grad is not None and torch.count_nonzero(p.grad) > 0 for p in model.classifier.parameters()
    )


def test_random_arm_has_nonzero_encoder_grads_after_backward():
    model = _make_model("random")
    model.unfreeze_encoder()
    _forward_backward(model, encoder_grad=True)

    enc_params = [p for m in model.encoder_modules() for p in m.parameters()]
    assert all(p.requires_grad for p in enc_params)
    assert any(
        p.grad is not None and torch.count_nonzero(p.grad) > 0 for p in enc_params
    ), "a randomly-initialised encoder that never receives gradient cannot train"


def test_encoder_grad_false_starves_a_trainable_encoder():
    """Guard on why encoder_grad exists: without it the 'random' arm is inert."""
    model = _make_model("random")
    model.unfreeze_encoder()
    _forward_backward(model, encoder_grad=False)

    enc_params = [p for m in model.encoder_modules() for p in m.parameters()]
    assert all(p.grad is None for p in enc_params)


def test_pretrained_finetuned_grads_reach_the_encoder():
    model = _make_model("pretrained_finetuned")
    model.freeze_encoder()  # what load_gaae_weights does
    model.unfreeze_encoder()  # ... and what the adapter then undoes for this arm
    _forward_backward(model, encoder_grad=True)

    enc_params = [p for m in model.encoder_modules() for p in m.parameters()]
    assert any(p.grad is not None and torch.count_nonzero(p.grad) > 0 for p in enc_params)


@pytest.mark.parametrize("arm", ["random", "none"])
def test_load_gaae_weights_refuses_the_non_pretrained_arms(arm, tmp_path):
    """Loading pretrained weights into these arms would void the ablation."""
    ckpt = tmp_path / "model_fake.pth"
    torch.save({}, ckpt)
    model = _make_model(arm)
    with pytest.raises(ValueError, match="must not receive reconstruction-pretrained"):
        model.load_gaae_weights(str(ckpt))


def test_default_arm_is_todays_model():
    """Constructed without the new flag, the model is exactly what it was before."""
    model = GELSTMClassifier(
        in_features=IN_FEATURES,
        gaae_hidden=GAAE_HIDDEN,
        gaae_latent=GAAE_LATENT,
        gaae_heads=1,
        gaae_cond_dim=2,
        gaae_dropout=0.0,
        lstm_hidden=4,
        lstm_layers=1,
        lstm_dropout=0.0,
    )
    assert model.encoder_init == "pretrained_frozen"
    assert model.encoder is not None
    assert model.embed_dim == GAAE_LATENT
    assert model.lstm_input_dim == GAAE_LATENT + 1


def test_set_feature_norm_matches_the_arms_embedding_width():
    model = _make_model("none")
    model.set_feature_norm(np.zeros(IN_FEATURES), np.ones(IN_FEATURES))
    with pytest.raises(ValueError, match="do not match embedding dim"):
        model.set_feature_norm(np.zeros(GAAE_LATENT), np.ones(GAAE_LATENT))


# --------------------------------------------------------------------------- #
# Adapter wiring (no GAAE checkpoint, no DELCODE matrices)
# --------------------------------------------------------------------------- #
_GAAE_HP = {"latent_dim": 6, "hidden_dim": 16, "num_heads": 2, "cond_dim": 2, "dropout": 0.3}


def _make_adapter(train_config: dict):
    from CLASSIFIER.adapters import get_adapter

    return get_adapter("gelstm")(
        gaae_ckpt_path="/nonexistent/model.pth",  # loading it would raise
        gaae_hp=_GAAE_HP,
        train_config=train_config,
        data_root="/nonexistent",
        cohorts_csv="/nonexistent/cohorts.csv",
        device="cpu",
        rng=np.random.default_rng(0),
    )


@pytest.mark.parametrize(
    "train_config, expected",
    [
        ({}, "pretrained_frozen"),  # no config at all
        ({"freeze_encoder": True}, "pretrained_frozen"),  # every configs/*.json today
        ({"freeze_encoder": False}, "pretrained_finetuned"),
        ({"encoder_init": "random"}, "random"),
        # An arm set in experiments.yaml on top of a JSON that still says
        # freeze_encoder: true — must resolve, not raise, for the encoderless arm.
        ({"encoder_init": "none", "freeze_encoder": True}, "none"),
    ],
)
def test_adapter_resolves_the_arm(train_config, expected):
    adapter = _make_adapter(train_config)
    assert adapter.encoder_init == expected
    assert adapter.model_config()["encoder_init"] == expected


def test_adapter_raises_on_contradictory_config():
    with pytest.raises(ValueError, match="Conflicting encoder configuration"):
        _make_adapter({"encoder_init": "random", "freeze_encoder": True})


@pytest.mark.parametrize("arm", ["random", "none"])
def test_adapter_builds_non_pretrained_arms_without_reading_the_checkpoint(arm):
    """gaae_ckpt_path points at nothing: building must still succeed for these arms."""
    adapter = _make_adapter({"encoder_init": arm, "lstm_hidden": 4, "classifier_hidden": 4})
    model = adapter._build_model()
    assert model.encoder_init == arm
    assert model.embed_dim == (adapter.in_features if arm == "none" else adapter.gaae_latent)
    assert model.lstm.input_size == model.embed_dim + 1
    # ``random`` must be trainable end-to-end; ``none`` has no encoder to train.
    assert (model.encoder is not None) is (arm == "random")
    if arm == "random":
        assert all(p.requires_grad for m in model.encoder_modules() for p in m.parameters())


def test_adapter_only_opens_encoder_gradients_for_trainable_arms():
    frozen = _make_adapter({})
    assert frozen._eval_cfg(None, encoder_grad=frozen.encoder_arm.trains_encoder).encoder_grad is (
        False
    )
    random_arm = _make_adapter({"encoder_init": "random"})
    assert random_arm._eval_cfg(
        None, encoder_grad=random_arm.encoder_arm.trains_encoder
    ).encoder_grad is True
    # Evaluation paths never open the encoder, whatever the arm.
    assert random_arm._eval_cfg(None).encoder_grad is False
    assert random_arm._eval_cfg(None, 0.5).encoder_grad is False
