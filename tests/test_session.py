"""Tests for conversation state: appending, windowing, and persistence."""

from __future__ import annotations

import os

from kat.session import Session


def test_add_appends_messages():
    session = Session()
    session.add("user", "first")
    session.add("assistant", "second")
    assert session.messages == [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "second"},
    ]


def test_window_keeps_system_first_and_trims_to_budget():
    session = Session(system="S")
    for marker in "01234":
        session.add("user", marker * 10)

    # Budget leaves room for the one-character system prompt plus two of the
    # ten-character messages; the third oldest should push the total negative.
    window = session.window(max_chars=25)
    contents = [message["content"] for message in window]

    # The system prompt is always first.
    assert window[0] == {"role": "system", "content": "S"}
    assert len(window) == 3

    # The most recent messages survive; the oldest are trimmed away.
    assert "4" * 10 in contents  # newest
    assert "3" * 10 in contents
    assert "0" * 10 not in contents
    assert "1" * 10 not in contents
    assert "2" * 10 not in contents

    # Exactly one system message, and it leads.
    assert [m["role"] for m in window].count("system") == 1


def test_window_without_system_has_no_system_message():
    session = Session()  # system defaults to ""
    session.add("user", "hello")
    window = session.window(max_chars=1000)
    assert all(message["role"] != "system" for message in window)
    assert window == [{"role": "user", "content": "hello"}]


def test_window_always_keeps_most_recent_even_when_oversized():
    session = Session(system="prompt")
    session.add("user", "x" * 5000)
    # Budget smaller than the single message: it must still be returned so the
    # agent is never handed an empty window.
    window = session.window(max_chars=10)
    assert window[-1]["content"] == "x" * 5000


def test_reset_clears_messages():
    session = Session()
    session.add("user", "something")
    session.reset()
    assert session.messages == []


def test_save_and_load_round_trip(tmp_path):
    session = Session(id="roundtrip", system="profile text")
    session.add("user", "enumerate the host")
    session.add("assistant", "OBSERVATION [pyrun]\nexit code: 0")

    path = session.save(tmp_path)
    assert path == tmp_path / "roundtrip.json"
    assert path.is_file()

    restored = Session.load(path)
    assert restored.id == session.id
    assert restored.system == session.system
    assert restored.messages == session.messages


def test_latest_picks_the_newest_file(tmp_path):
    older = Session(id="older")
    older.save(tmp_path)
    newer = Session(id="newer")
    newer.save(tmp_path)

    # Pin modification times so ordering does not depend on filesystem clock
    # resolution.
    os.utime(tmp_path / "older.json", (1_000_000, 1_000_000))
    os.utime(tmp_path / "newer.json", (2_000_000, 2_000_000))

    latest = Session.latest(tmp_path)
    assert latest is not None
    assert latest.id == "newer"


def test_latest_returns_none_for_missing_directory(tmp_path):
    assert Session.latest(tmp_path / "does-not-exist") is None
