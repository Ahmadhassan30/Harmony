"""Camelot wheel and Open Key notation utilities."""

from __future__ import annotations

# Camelot Wheel mapping
# Format: (key, mode) -> camelot_code
CAMELOT_MAP: dict[tuple[str, str], str] = {
    # Minor keys (A column)
    ("A", "minor"): "8A",   ("E", "minor"): "9A",   ("B", "minor"): "10A",
    ("F#", "minor"): "11A", ("C#", "minor"): "12A", ("G#", "minor"): "1A",
    ("D#", "minor"): "2A",  ("A#", "minor"): "3A",  ("F", "minor"): "4A",
    ("C", "minor"): "5A",   ("G", "minor"): "6A",   ("D", "minor"): "7A",
    # Major keys (B column)
    ("C", "major"): "8B",   ("G", "major"): "9B",   ("D", "major"): "10B",
    ("A", "major"): "11B",  ("E", "major"): "12B",  ("B", "major"): "1B",
    ("F#", "major"): "2B",  ("C#", "major"): "3B",  ("G#", "major"): "4B",
    ("D#", "major"): "5B",  ("A#", "major"): "6B",  ("F", "major"): "7B",
    # Enharmonic equivalents
    ("Gb", "minor"): "11A", ("Db", "minor"): "12A", ("Ab", "minor"): "1A",
    ("Eb", "minor"): "2A",  ("Bb", "minor"): "3A",
    ("Gb", "major"): "2B",  ("Db", "major"): "3B",  ("Ab", "major"): "4B",
    ("Eb", "major"): "5B",  ("Bb", "major"): "6B",
}


def key_to_camelot(key: str, mode: str) -> str:
    """Convert a musical key to Camelot wheel notation.

    Examples:
        key_to_camelot("A", "minor") -> "8A"
        key_to_camelot("C", "major") -> "8B"
    """
    return CAMELOT_MAP.get((key, mode.lower()), "?")


def compatible_keys(camelot: str) -> list[str]:
    """Return harmonically compatible Camelot codes for mixing.

    Compatible keys are: same position, ±1 position, and parallel (A↔B).
    """
    if len(camelot) < 2:
        return []

    num = int(camelot[:-1])
    letter = camelot[-1]

    compatible = [
        camelot,                            # Same key
        f"{num}{'B' if letter == 'A' else 'A'}",  # Parallel major/minor
        f"{(num % 12) + 1}{letter}",        # +1 (up a fifth)
        f"{((num - 2) % 12) + 1}{letter}",  # -1 (down a fifth)
    ]

    return compatible
