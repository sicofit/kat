"""Tests for the tool contract and the execution layer.

These exercise argument rendering and validation at the spec level, and then
drive the real :class:`Executor` against the running interpreter so that output
capture, missing-binary handling, timeouts, and output clipping are tested
through an actual subprocess rather than a mock.
"""

from __future__ import annotations

import pytest

from kat.tools import ArgSpec, Executor, ToolArgError, ToolSpec


# --- argument rendering ----------------------------------------------------


def test_render_option_emits_token_and_value():
    spec = ArgSpec(name="port", kind="option", token="-p")
    assert spec.render("80") == ["-p", "80"]


def test_render_flag_is_present_or_absent():
    spec = ArgSpec(name="verbose", kind="flag", token="-v")
    assert spec.render(True) == ["-v"]
    assert spec.render(False) == []


def test_render_positional_emits_bare_value():
    spec = ArgSpec(name="target", kind="positional")
    assert spec.render("host") == ["host"]


# --- target validation -----------------------------------------------------


@pytest.mark.parametrize("value", ["10.10.10.10", "http://host/path", "example.com", "ftp://h:21/x"])
def test_target_validation_accepts_safe_values(value):
    spec = ArgSpec(name="target", kind="positional", validate="target")
    assert spec.render(value) == [value]


@pytest.mark.parametrize("value", ["a; rm -rf /", "$(x)", "host && id", "a|b", "`id`", "two hosts"])
def test_target_validation_rejects_shell_metacharacters(value):
    spec = ArgSpec(name="target", kind="positional", validate="target")
    with pytest.raises(ToolArgError):
        spec.render(value)


# --- port validation -------------------------------------------------------


@pytest.mark.parametrize("value", ["22,80,443", "1-1000", "80"])
def test_port_validation_accepts_lists_and_ranges(value):
    spec = ArgSpec(name="ports", kind="positional", validate="port")
    assert spec.render(value) == [value]


@pytest.mark.parametrize("value", ["80;22", "80 22", "$(nc)", "80,abc"])
def test_port_validation_rejects_malformed_specs(value):
    spec = ArgSpec(name="ports", kind="positional", validate="port")
    with pytest.raises(ToolArgError):
        spec.render(value)


# --- argv assembly ---------------------------------------------------------


def _demo_spec() -> ToolSpec:
    # No builder, so the generic argument machinery (known/unknown/required)
    # applies. The binary need not exist: argv only assembles the vector.
    return ToolSpec(
        name="demo",
        binary="kat-demo-binary-not-installed",
        summary="demo",
        args=(ArgSpec(name="target", kind="positional", required=True),),
    )


def test_argv_assembles_required_positional():
    spec = _demo_spec()
    assert spec.argv({"target": "10.0.0.1"}) == ["kat-demo-binary-not-installed", "10.0.0.1"]


def test_argv_rejects_unknown_argument():
    spec = _demo_spec()
    with pytest.raises(ToolArgError):
        spec.argv({"target": "10.0.0.1", "mystery": "value"})


def test_argv_rejects_missing_required_argument():
    spec = _demo_spec()
    with pytest.raises(ToolArgError):
        spec.argv({})


# --- execution -------------------------------------------------------------


def test_executor_captures_stdout(pyrun_spec):
    result = Executor().run(pyrun_spec, {"code": "print('hello')"})
    assert result.ok is True
    assert result.returncode == 0
    assert "hello" in result.stdout
    assert result.error is None


def test_executor_reports_missing_binary():
    spec = ToolSpec(
        name="ghost",
        binary="kat-nonexistent-binary-xyz",
        summary="not here",
        builder=lambda a: [],
    )
    result = Executor().run(spec, {})
    assert result.error is not None
    assert "not installed" in result.error
    assert result.ok is False


def test_executor_times_out(pyrun_spec):
    result = Executor(timeout=1).run(pyrun_spec, {"code": "import time; time.sleep(5)"})
    assert result.timed_out is True
    assert result.ok is False


def test_executor_clips_oversized_output(pyrun_spec):
    result = Executor(timeout=30, max_output=300).run(pyrun_spec, {"code": "print('x' * 5000)"})
    assert result.truncated is True
    assert len(result.stdout) < 5000
