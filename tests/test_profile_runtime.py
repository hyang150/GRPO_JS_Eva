import importlib.util
import json
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location(
    "profile_runtime", Path(__file__).resolve().parents[1] / "experiments/profile_runtime.py"
)
profile_runtime = importlib.util.module_from_spec(spec)
spec.loader.exec_module(profile_runtime)


@pytest.mark.parametrize("intervals, expected", [
    ([], 0),
    ([(0, 5), (1, 2)], 5),
    ([(3, 8), (0, 5)], 8),
    ([(0, 2), (4, 6)], 4),
    ([(0, 0), (4, 3)], 0),
])
def test_union_duration(intervals, expected):
    assert profile_runtime.union_duration(intervals) == expected


def test_trace_coverage_does_not_double_count_overlaps(tmp_path):
    trace = tmp_path / "trace.json"
    trace.write_text(json.dumps({"traceEvents": [
        {"cat": "kernel", "ph": "X", "ts": 0, "dur": 1000, "name": "a"},
        {"cat": "kernel", "ph": "X", "ts": 100, "dur": 100, "name": "b"},
        {"cat": "kernel", "ph": "X", "ts": 2000, "dur": 1000, "name": "a"},
        {"cat": "cpu_op", "ph": "X", "ts": 0, "dur": 9000, "name": "host"},
    ]}))
    result = profile_runtime.summarize_trace(trace)
    assert result["kernel_count"] == 3
    assert result["kernel_union_ms"] == 2
    assert result["kernel_span_ms"] == 3
    assert result["kernel_coverage_of_span"] == pytest.approx(2 / 3)


def test_trace_without_cuda_is_explicit(tmp_path):
    trace = tmp_path / "trace.json"
    trace.write_text(json.dumps({"traceEvents": []}))
    result = profile_runtime.summarize_trace(trace)
    assert result["kernel_count"] == 0
    assert "warning" in result


def test_diagnostic_prefix_left_pads_and_removes_terminal_eos():
    import torch

    seq = torch.tensor([[0, 9, 1, 7, 0], [9, 2, 3, 4, 5]])
    attn = torch.tensor([[0, 1, 1, 1, 0], [1, 1, 1, 1, 1]])
    ids, mask = profile_runtime.diagnostic_prefix(seq, attn, 5, pad_id=0, eos_id=7)
    assert ids.tolist() == [[0, 0, 0, 9, 1], [9, 2, 3, 4, 5]]
    assert mask.tolist() == [[0, 0, 0, 1, 1], [1, 1, 1, 1, 1]]


def test_diagnostic_prefix_rejects_empty_rows():
    import torch

    with pytest.raises(ValueError, match="valid token"):
        profile_runtime.diagnostic_prefix(torch.zeros((1, 2), dtype=torch.long),
                                          torch.zeros((1, 2)), 2, 0, 7)
