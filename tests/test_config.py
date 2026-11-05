"""Tests for configuration loading and precedence.

Covered: the dataclass defaults, validation of the approval policy, coercion of
``KAT_*`` environment variables to their declared types, and the precedence
rule that explicit overrides win over the environment.
"""

from __future__ import annotations

import pytest

from kat.config import Config


def test_defaults_are_sane():
    config = Config()
    assert config.base_url == "http://127.0.0.1:9932"
    assert config.model == "local"
    assert config.profile == "professional"
    assert config.approval == "active"
    assert config.max_steps == 30
    assert config.temperature == pytest.approx(0.4)
    assert config.stream is True
    # Derived properties.
    assert config.endpoint == "http://127.0.0.1:9932"
    assert config.sessions_path.is_absolute()


def test_endpoint_strips_trailing_slash():
    config = Config(base_url="http://host:9932/")
    assert config.endpoint == "http://host:9932"


def test_invalid_approval_raises():
    with pytest.raises(ValueError):
        Config(approval="whenever-it-feels-right")


def test_invalid_approval_via_load_raises(tmp_path):
    missing = tmp_path / "no-such-config.toml"
    with pytest.raises(ValueError):
        Config.load(path=missing, approval="bogus")


def test_env_overrides_are_coerced(monkeypatch, tmp_path):
    # Point at a file that does not exist so no ambient config file is read;
    # a non-existent explicit path disables the default-path search entirely.
    missing = tmp_path / "no-such-config.toml"
    monkeypatch.setenv("KAT_MAX_STEPS", "7")
    monkeypatch.setenv("KAT_BASE_URL", "http://example.test:1234")

    config = Config.load(path=missing)

    # KAT_MAX_STEPS is declared int and must be coerced from the string value.
    assert config.max_steps == 7
    assert isinstance(config.max_steps, int)
    # KAT_BASE_URL stays a string.
    assert config.base_url == "http://example.test:1234"
    assert isinstance(config.base_url, str)
    assert config.endpoint == "http://example.test:1234"


def test_overrides_win_over_env(monkeypatch, tmp_path):
    missing = tmp_path / "no-such-config.toml"
    monkeypatch.setenv("KAT_MAX_STEPS", "7")

    config = Config.load(path=missing, max_steps=99)

    assert config.max_steps == 99


def test_none_overrides_are_ignored(monkeypatch, tmp_path):
    # Overrides left as None (the common CLI default) must not clobber the
    # value supplied through the environment.
    missing = tmp_path / "no-such-config.toml"
    monkeypatch.setenv("KAT_MAX_STEPS", "7")

    config = Config.load(path=missing, max_steps=None)

    assert config.max_steps == 7
