"""The drawn lesson figures: every DSMA topic has one, and each one is valid, self-contained SVG."""

from __future__ import annotations

import json
import re
from pathlib import Path
from xml.etree import ElementTree

import pytest

from indiclearn.studio.courses import diagrams

COURSES = Path(diagrams.__file__).resolve().parent
TOPICS = [t["id"] for u in json.loads((COURSES / "dsma_syllabus.json").read_text(encoding="utf-8"))["units"] for t in u["topics"]]


def test_every_dsma_topic_has_a_figure():
    missing = [t for t in TOPICS if t not in diagrams.have("dsma")]
    assert not missing, f"topics with no drawn figure: {missing}"


def test_unknown_topic_is_not_an_error():
    assert diagrams.render("dsma", "no-such-topic") is None
    assert diagrams.render("nosuchsubject", "posets") is None


def _figure(topic: str) -> str:
    """The figure, or skip while that unit is still being drawn (one test above tracks what is missing)."""
    s = diagrams.render("dsma", topic)
    if s is None:
        pytest.skip(f"{topic}: not drawn yet")
    return s


@pytest.mark.parametrize("topic", TOPICS)
def test_figure_is_well_formed_svg(topic):
    s = _figure(topic)
    assert s.startswith("<svg"), f"{topic}: not an SVG"
    root = ElementTree.fromstring(s)  # raises if the SVG is malformed
    assert root.tag.endswith("svg")
    assert root.get("viewBox"), f"{topic}: no viewBox"
    assert root.get("aria-label"), f"{topic}: no aria-label for screen readers"


@pytest.mark.parametrize("topic", TOPICS)
def test_figure_is_offline_and_themed(topic):
    s = _figure(topic)
    assert "http://" not in s and "https://" not in s.replace("http://www.w3.org", ""), f"{topic}: external reference"
    assert "<script" not in s.lower(), f"{topic}: scripts do not belong in a figure"
    assert "<image" not in s.lower(), f"{topic}: figures are drawn, not embedded bitmaps"
    # colours come from the theme, so a figure follows the app's palette
    hexes = set(re.findall(r'(?<!,)#[0-9A-Fa-f]{3,6}', re.sub(r"var\([^)]*\)", "", s)))
    assert not hexes, f"{topic}: literal colours {sorted(hexes)} - use the palette constants"


@pytest.mark.parametrize("topic", TOPICS)
def test_figure_is_deterministic(topic):
    assert _figure(topic) == diagrams.render("dsma", topic), f"{topic}: changes between calls"


@pytest.mark.parametrize("topic", TOPICS)
def test_figure_fits_the_lesson_column(topic):
    vb = [float(x) for x in ElementTree.fromstring(_figure(topic)).get("viewBox").split()]
    assert vb[2] <= 760, f"{topic}: {vb[2]}px wide, too wide for the lesson column"
    assert 80 <= vb[3] <= 900, f"{topic}: {vb[3]}px tall"


def test_lesson_response_carries_the_figure(client, auth):
    r = client.get("/api/courses/dsma/lessons/posets", headers=auth).json()
    assert r["figure"].startswith("<svg"), "the lesson page gets the drawn figure, not only the words"
    assert client.get("/api/courses/os/lessons/paging", headers=auth).json()["figure"] is None  # OS has none yet
