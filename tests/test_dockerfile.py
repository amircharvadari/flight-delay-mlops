"""Static checks of the serving image definition (D-018, D-039, I-24).

The real regression test for I-24 is the CI job "docker", which builds the
image from a checkout with owner-only file modes and starts the container.
These checks run everywhere, also without Docker.
"""
from __future__ import annotations

import re

from flightdelay.config import PROJECT_ROOT

LINES = [ln.strip() for ln in (PROJECT_ROOT / "Dockerfile").read_text().splitlines()]
INSTRUCTIONS = [ln for ln in LINES if ln and not ln.startswith("#")]


def _index(pattern: str) -> int:
    hits = [i for i, ln in enumerate(INSTRUCTIONS) if re.match(pattern, ln)]
    assert hits, pattern
    return hits[-1]


def test_runs_as_a_non_root_user():
    user = INSTRUCTIONS[_index(r"USER\s")].split()[1]
    assert user not in {"root", "0"}
    # USER comes after every COPY and RUN, so nothing switches back to root.
    assert _index(r"USER\s") > max(_index(r"COPY\s"), _index(r"RUN\s"))


def test_code_and_configs_are_made_readable_after_copy():
    """I-24: COPY keeps owner-only modes of the build context."""
    chmod = _index(r"RUN chmod -R u=rwX,go=rX /app/src /app/configs")
    assert chmod > _index(r"COPY src ") and chmod > _index(r"COPY configs ")
    assert chmod < _index(r"USER\s")


def test_model_is_not_baked_into_the_image():
    """D-039: the bundle is mounted at run time, never copied into the image."""
    copies = " ".join(ln for ln in INSTRUCTIONS if ln.startswith("COPY"))
    assert "artifacts" not in copies and "serving" not in copies
    assert any("FLIGHTDELAY_SERVING_DIR=/app/serving" in ln for ln in LINES)


def test_dependencies_come_from_the_lock_file_and_there_is_a_health_check():
    assert all("--locked" in ln for ln in INSTRUCTIONS if "uv sync" in ln)
    assert any(ln.startswith("HEALTHCHECK") for ln in INSTRUCTIONS)
