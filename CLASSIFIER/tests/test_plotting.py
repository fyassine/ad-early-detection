"""Tests for SHARED.plotting (headless Agg backend)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # no display in CI; must precede pyplot import

import pytest

from SHARED.plotting import (
    add_note,
    append_note_line,
    format_model_runs_note,
    note_center_x,
    run_name_from_checkpoint_path,
)


def test_format_model_runs_note_single_model():
    assert (
        format_model_runs_note({"GAAE": "quiet-fern-12"}) == "Model runs used — GAAE: quiet-fern-12"
    )


def test_format_model_runs_note_multiple_models():
    note = format_model_runs_note({"GAAE": "quiet-fern-12", "GEC": "brave-otter-7"})
    assert note == "Model runs used — GAAE: quiet-fern-12; GEC: brave-otter-7"


def test_format_model_runs_note_truncates_with_max_models():
    runs = {f"model-{i}": f"run-{i}" for i in range(7)}
    note = format_model_runs_note(runs, max_models=5)
    assert note == (
        "Model runs used — model-0: run-0; model-1: run-1; model-2: run-2; "
        "model-3: run-3; model-4: run-4; +2 more"
    )


def test_format_model_runs_note_no_truncation_when_under_max():
    runs = {"GAAE": "quiet-fern-12", "GEC": "brave-otter-7"}
    assert format_model_runs_note(runs, max_models=5) == format_model_runs_note(runs)


def test_format_model_runs_note_requires_at_least_one_run():
    with pytest.raises(ValueError):
        format_model_runs_note({})


def test_append_note_line_appends_to_existing_text():
    assert append_note_line("existing\ntext", "new") == "existing\ntext\nnew"


def test_append_note_line_no_leading_blank_when_empty():
    assert append_note_line("", "new") == "new"


def test_run_name_from_checkpoint_path():
    assert (
        run_name_from_checkpoint_path("/outputs/quiet-fern-12/model_quiet-fern-12.pth")
        == "quiet-fern-12"
    )


def test_add_note_renders_footnote_text():
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2)
    add_note(fig, axes, "Model runs used — GAAE: quiet-fern-12")

    assert len(fig.texts) == 1
    text_artist = fig.texts[0]
    assert text_artist.get_text() == "Model runs used — GAAE: quiet-fern-12"
    assert text_artist.get_style() == "italic"
    assert text_artist.get_color() == "gray"
    assert text_artist.get_fontsize() == 8
    plt.close(fig)


def test_add_note_respects_style_overrides():
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2)
    add_note(fig, axes, "note", color="black", fontsize=10)

    text_artist = fig.texts[0]
    assert text_artist.get_color() == "black"
    assert text_artist.get_fontsize() == 10
    plt.close(fig)


def test_note_center_x_is_between_axes_bounds():
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2)
    fig.canvas.draw()
    x0 = min(ax.get_position().x0 for ax in axes)
    x1 = max(ax.get_position().x1 for ax in axes)

    x = note_center_x(fig, axes)
    assert x0 <= x <= x1
    plt.close(fig)
