"""
Generate W&B-style random run names (``adjective-noun-number``).

The names are generated locally so they can be used for both the run directory
and the W&B display name — matching the style W&B uses for its auto-generated
names (e.g. ``ethereal-planet-16``, ``classic-wind-17``).

The number suffix counts existing runs under the experiment's output directory,
giving a monotonically increasing sequence per experiment (like W&B's per-project
counter).
"""
from __future__ import annotations

import random
from pathlib import Path

# Curated word lists matching W&B's naming style.
_ADJECTIVES = [
    "absurd", "ancient", "apricot", "autumn", "azure",
    "balmy", "brave", "bright", "calm", "celestial",
    "classic", "copper", "cosmic", "crimson", "curious",
    "dainty", "dapper", "dark", "dawn", "desert",
    "divine", "eager", "earnest", "earthy", "electric",
    "ethereal", "exalted", "faint", "fallen", "fiery",
    "floral", "fragrant", "frosty", "gentle", "glad",
    "glorious", "golden", "graceful", "grateful", "happy",
    "hardy", "honest", "icy", "jolly", "kind",
    "laced", "leafy", "light", "lilac", "lively",
    "lucky", "lunar", "magic", "mild", "misty",
    "morning", "noble", "olive", "orange", "patient",
    "peach", "polar", "proud", "quiet", "radiant",
    "resilient", "risen", "rosy", "royal", "ruby",
    "rustic", "sacred", "scarlet", "serene", "silent",
    "silver", "sleek", "smooth", "snowy", "solar",
    "splendid", "stellar", "still", "stoic", "summer",
    "sunny", "swift", "tidal", "trim", "twilight",
    "upbeat", "vernal", "vivid", "warm", "wild",
    "winter", "wise", "woolen", "worthy", "zesty",
]

_NOUNS = [
    "abyss", "atom", "bird", "blaze", "bloom",
    "breeze", "brook", "canyon", "cherry", "cloud",
    "comet", "coral", "cosmos", "creek", "crystal",
    "dawn", "dew", "dream", "dune", "dust",
    "ember", "energy", "feather", "fern", "field",
    "fire", "flame", "flower", "fog", "forest",
    "frost", "galaxy", "garden", "glade", "glimmer",
    "grove", "harbor", "haze", "hill", "horizon",
    "island", "lake", "leaf", "light", "meadow",
    "mist", "moon", "morning", "mountain", "night",
    "oasis", "ocean", "orchid", "paper", "pebble",
    "pine", "planet", "pond", "rain", "reef",
    "resonance", "river", "sea", "shadow", "shape",
    "silence", "sky", "smoke", "snow", "sound",
    "spark", "star", "stone", "storm", "stream",
    "sun", "sunset", "surf", "thunder", "tide",
    "totem", "tree", "valley", "violet", "voice",
    "water", "waterfall", "wave", "whisper", "wildflower",
    "wind", "wood", "yogurt", "dawn", "zenith",
]


def _count_existing_runs(experiment_output_dir: Path) -> int:
    """Count existing run directories under ``<experiment>/runs/``."""
    runs_dir = experiment_output_dir / "runs"
    if not runs_dir.is_dir():
        return 0
    return sum(1 for p in runs_dir.iterdir() if p.is_dir())


def generate_run_name(experiment_output_dir: Path | None = None) -> str:
    """Generate a W&B-style name like ``ethereal-planet-16``.

    Parameters
    ----------
    experiment_output_dir
        Path to ``outputs/<experiment-id>/``. If provided, the numeric suffix
        is the count of existing runs + 1 (monotonically increasing per
        experiment). Otherwise a random number 1–999 is used.

    Returns
    -------
    A string like ``"classic-wind-17"``.
    """
    adj = random.choice(_ADJECTIVES)
    noun = random.choice(_NOUNS)
    if experiment_output_dir is not None:
        num = _count_existing_runs(experiment_output_dir) + 1
    else:
        num = random.randint(1, 999)
    return f"{adj}-{noun}-{num}"
