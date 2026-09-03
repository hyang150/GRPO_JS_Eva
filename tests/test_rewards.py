"""Extraction tests -- every miss here is reward noise in the experiment."""

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import pytest

from grpo_eva.rewards import (correctness_reward, extract_answer, extract_gold,
                              format_reward)


@pytest.mark.parametrize("text,want", [
    (r"so the total is \boxed{72}", 72.0),
    (r"\boxed{-15}", -15.0),
    (r"\boxed{1,234}", 1234.0),
    (r"\boxed{3.5}", 3.5),
    ("<answer>42</answer>", 42.0),
    ("blah\n#### 18", 18.0),
    ("The answer is 91", 91.0),
    ("the result: $250", 250.0),
    ("she sold 48 clips then 24 more, so 72", 72.0),   # bare last number
    ("no digits at all here", None),
    ("", None),
])
def test_extract_answer(text, want):
    assert extract_answer(text) == want


def test_boxed_wins_over_trailing_prose():
    """A trailing sentence must not override an explicit boxed answer."""
    t = r"I first tried 50. \boxed{72} Note that 100 would be too many."
    assert extract_answer(t) == 72.0


def test_last_boxed_wins_when_the_model_restates():
    assert extract_answer(r"\boxed{5} ... wait, actually \boxed{7}") == 7.0


@pytest.mark.parametrize("field,want", [
    ("Natalia sold 48/2 = <<48/2=24>>24 clips.\n#### 72", 72.0),
    ("...\n#### 1,234", 1234.0),
    ("no marker", None),
])
def test_extract_gold(field, want):
    assert extract_gold(field) == want


def test_gold_ignores_gsm8k_calculator_annotations():
    """The <<48/2=24>> spans must not be mistaken for the answer."""
    assert extract_gold("a <<48/2=24>> b <<24+48=72>>\n#### 72") == 72.0


def test_correctness_reward_is_binary_and_tolerant():
    comps = [r"\boxed{72}", r"\boxed{72.00001}", r"\boxed{71}", "nothing"]
    assert correctness_reward(comps, [72.0] * 4) == [1.0, 1.0, 0.0, 0.0]


def test_correctness_reward_handles_missing_gold():
    assert correctness_reward([r"\boxed{5}"], [None]) == [0.0]


def test_format_reward_needs_an_explicit_marker():
    assert format_reward([r"\boxed{5}", "<answer>5</answer>",
                          "#### 5", "the answer is 5"]) == [1.0, 1.0, 1.0, 0.0]
