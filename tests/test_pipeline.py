"""Test the smoothing / debouncing logic of the two-stage pipeline.

We don't load real models — we instantiate a pipeline directly and only
exercise its private ``_try_emit`` state machine.
"""

import time
from collections import deque

import pytest

from signlang.pipeline.two_stage import TwoStagePipeline


def feed(pipe, letter: str, n: int = 3):
    out = None
    for _ in range(n):
        out = pipe._try_emit(letter)
    return out


@pytest.fixture
def pipe():
    # Build a partially-initialised TwoStagePipeline without loading models.
    p = TwoStagePipeline.__new__(TwoStagePipeline)
    p.smoothing = deque(maxlen=3)
    p.cooldown_ms = 50
    p.require_neutral_between_letters = True
    p.neutral_frames_to_reset = 2
    p._armed = True
    p._neutral_frames = 0
    p._last_emitted_at = 0.0
    p._last_letter = None
    p.text_buffer = ""
    return p


def test_majority_vote(pipe):
    assert pipe._try_emit("A") is None
    assert pipe._try_emit("A") is None
    assert pipe._try_emit("A") == "A"
    assert pipe.text_buffer == "A"


def test_no_majority(pipe):
    assert pipe._try_emit("A") is None
    assert pipe._try_emit("B") is None
    assert pipe._try_emit("C") is None


def test_space_appends_space(pipe):
    feed(pipe, "space")
    assert pipe.text_buffer == " "


def test_del_removes_last(pipe):
    pipe.text_buffer = "HELL"
    feed(pipe, "del")
    assert pipe.text_buffer == "HEL"


def test_nothing_skipped(pipe):
    feed(pipe, "nothing")
    assert pipe.text_buffer == ""


def test_neutral_gate_prevents_held_sign_repeats(pipe):
    feed(pipe, "A")
    assert pipe.text_buffer == "A"
    # Even after cooldown, holding the same sign should not spam letters.
    time.sleep(pipe.cooldown_ms / 1000 + 0.05)
    feed(pipe, "A")
    assert pipe.text_buffer == "A"


def test_neutral_rearms_for_repeated_letters(pipe):
    feed(pipe, "A")
    assert pipe.text_buffer == "A"
    pipe._try_emit("nothing")
    pipe._try_emit("nothing")
    feed(pipe, "A")
    assert pipe.text_buffer == "AA"


def test_uncertain_does_not_rearm(pipe):
    """A few uncertain frames (low conf / low margin) must not re-arm."""
    feed(pipe, "A")
    assert pipe.text_buffer == "A"
    # Simulate uncertain frames: classifier returned a prediction but it was
    # gated out at the conf/margin check, so process() calls _observe_uncertain.
    pipe._observe_uncertain()
    pipe._observe_uncertain()
    pipe._observe_uncertain()
    # Holding the same gesture must still not emit again.
    feed(pipe, "A")
    assert pipe.text_buffer == "A"


def test_cooldown_mode_without_neutral_gate(pipe):
    pipe.require_neutral_between_letters = False
    feed(pipe, "A")
    assert pipe.text_buffer == "A"
    feed(pipe, "A")
    assert pipe.text_buffer == "A"
    time.sleep(pipe.cooldown_ms / 1000 + 0.05)
    feed(pipe, "A")
    assert pipe.text_buffer == "AA"
