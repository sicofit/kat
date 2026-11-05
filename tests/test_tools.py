"""Tests for the tool contract and the execution layer.

These exercise argument rendering and validation at the spec level, and then
drive the real :class:`Executor` against the running interpreter so that output
capture, missing-binary handling, timeouts, and output clipping are tested
through an actual subprocess rather than a mock.
"""

from __future__ import annotations

import os

import pytest

import kat.tools.registry as registry_mod
from kat.tools import ArgSpec, Executor, ToolArgError, ToolSpec
from kat.tools.registry import build_registry


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


@pytest.mark.parametrize("meta", ["reason", "description", "rationale", "note", "goal"])
def test_argv_ignores_rationale_metadata_in_args(meta):
    # A weak model narrates its intent inside the arg dict ("reason": "Full port
    # scan ..."). That is not a tool option, so it must be ignored, not raised as
    # an unknown argument that kills the step -- and it must not reach argv.
    spec = _demo_spec()
    argv = spec.argv({"target": "10.0.0.1", meta: "because I felt like it"})
    assert argv == ["kat-demo-binary-not-installed", "10.0.0.1"]


def test_argv_still_rejects_a_real_typo_alongside_metadata():
    # Tolerating metadata must not mask a genuine mistake: an undeclared option
    # that is not a known meta key still fails, so "targ" (meant "target") is
    # caught rather than silently dropped.
    spec = _demo_spec()
    with pytest.raises(ToolArgError):
        spec.argv({"target": "10.0.0.1", "reason": "x", "targ": "10.0.0.2"})


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


def test_executor_shows_printable_strings_for_binary_stdout(pyrun_spec):
    # A tool that dumps a binary body (here a pcap-like blob with cleartext FTP
    # credentials embedded between binary noise) must not flood the context with
    # raw bytes: the executor recovers the printable strings, as `strings` does.
    code = (
        "import sys; b=bytes;"
        "sys.stdout.buffer.write(b(range(0,32))*40 + b'USER nathan\\n' + "
        "b(range(128,256))*20 + b'PASS Buck3tH4TF0RM3!' + b(range(0,32))*40)"
    )
    result = Executor(timeout=30).run(pyrun_spec, {"code": code})
    assert result.ok is True
    assert "printable strings" in result.stdout
    assert "USER nathan" in result.stdout
    assert "PASS Buck3tH4TF0RM3!" in result.stdout
    assert "\x00" not in result.stdout          # raw NUL bytes are not reproduced


def test_executor_leaves_text_stdout_untouched(pyrun_spec):
    result = Executor(timeout=30).run(
        pyrun_spec, {"code": "print('<html><body>login here</body></html>')"}
    )
    assert result.ok is True
    assert "printable strings" not in result.stdout
    assert "login here" in result.stdout


# --- curl URL and credential handling --------------------------------------
#
# The command runs without a shell, so a URL carries its metacharacters to the
# binary verbatim. These exercise the builder: real URLs (query strings, FTP
# credentials, injection payloads) must pass, while whitespace and a leading
# dash must still be refused so the value stays one token and cannot pose as an
# option.


def _curl_spec() -> ToolSpec:
    return build_registry().get("curl")


def _bin_stem(path: str) -> str:
    # The resolved binary differs by host (/usr/bin/ssh on Kali, an absolute
    # ...\ssh.EXE on Windows), so compare the lowercased basename without its
    # extension: "ssh" or "sshpass", never the full path.
    return os.path.splitext(os.path.basename(path))[0].lower()


def test_curl_accepts_ftp_url_with_special_characters_in_password():
    spec = _curl_spec()
    url = "ftp://nathan:Buck3tH4TF0RM3!@10.129.142.36/user.txt"
    argv = spec.argv({"url": url})
    assert argv[-1] == url


@pytest.mark.parametrize(
    "url",
    [
        "http://host/items?id=1&page=2",          # query string with &
        "http://host/vuln?q=1';DROP--",           # injection test payload
        "http://host/path;matrix=1",              # matrix parameter
        "http://user:p@ss!word@host/path",        # credentials in the URL
    ],
)
def test_curl_accepts_real_world_urls(url):
    spec = _curl_spec()
    argv = spec.argv({"url": url})
    assert argv[-1] == url


def test_curl_saves_body_to_file_silently():
    spec = _curl_spec()
    argv = spec.argv({"url": "http://host/download/0", "output": "capture.pcap"})
    assert "-s" in argv
    assert argv[argv.index("-o") + 1] == "capture.pcap"
    assert argv[-1] == "http://host/download/0"


@pytest.mark.parametrize("url", ["http://host/ a b", "with\ttab", "line\nbreak"])
def test_curl_still_rejects_whitespace_in_url(url):
    spec = _curl_spec()
    with pytest.raises(ToolArgError):
        spec.argv({"url": url})


def test_curl_still_rejects_option_injection_via_leading_dash():
    spec = _curl_spec()
    with pytest.raises(ToolArgError):
        spec.argv({"url": "-ocaptured"})


def test_sqlmap_accepts_url_with_multiple_query_parameters():
    spec = build_registry().get("sqlmap")
    url = "http://host/view?id=1&cat=2"
    argv = spec.argv({"url": url})
    assert url in argv


# --- curl as a web-exploitation client -------------------------------------


def test_curl_sends_headers_cookie_and_post_body():
    spec = _curl_spec()
    argv = spec.argv({
        "url": "http://host/login",
        "header": ["X-Forwarded-For: 127.0.0.1", "Authorization: Bearer x"],
        "cookie": "session=abc123",
        "data": "user=admin&pass=p@ss!",
    })
    assert argv.count("-H") == 2
    assert "X-Forwarded-For: 127.0.0.1" in argv
    assert "Authorization: Bearer x" in argv
    assert argv[argv.index("-b") + 1] == "session=abc123"
    assert argv[argv.index("--data") + 1] == "user=admin&pass=p@ss!"
    assert argv[-1] == "http://host/login"


def test_curl_accepts_a_single_header_string():
    spec = _curl_spec()
    argv = spec.argv({"url": "http://host/", "header": "X-Api-Key: secret"})
    assert argv[argv.index("-H") + 1] == "X-Api-Key: secret"


# --- output: a word means "show me", a path means "save it" ----------------
#
# A weak model passes output="yes" meaning "save it". Taken as a filename that
# wrote every body to a file called "yes" the agent could never read, so it
# never saw a page body or followed a link. A boolean-ish word must therefore
# mean "no file, print the body"; only a real path saves to disk.


@pytest.mark.parametrize("word", [
    "yes", "true", "on", "YES",
    # Semantic "show me the body" words a weak model reaches for (output="body"
    # was taken as a filename in a real run, so the pcap body was written to a
    # file called "body" and the agent never read the cleartext inside it).
    "body", "response", "stdout", "print", "show", "content", "console",
    "inline", "Body", "STDOUT",
])
def test_curl_output_boolean_word_prints_the_body(word):
    spec = _curl_spec()
    argv = spec.argv({"url": "http://host/data/1", "output": word})
    assert "-o" not in argv          # no file is created
    assert word not in argv          # the word is not taken as a filename
    assert argv[-1] == "http://host/data/1"


def test_curl_output_real_filename_is_saved_quietly():
    spec = _curl_spec()
    argv = spec.argv({"url": "http://host/download/0", "output": "capture.pcap"})
    assert "-s" in argv
    assert argv[argv.index("-o") + 1] == "capture.pcap"


def test_curl_saving_still_reports_size_and_content_type():
    # Saving to a file leaves curl's stdout empty, which blinds the agent on the
    # step that held the answer (a page saved by mistake, then strings-filtered,
    # hides a short link like /download/0). -w makes the saved step self-describing
    # so a text/* content-type tells the agent it saved a page, not an artefact.
    spec = _curl_spec()
    argv = spec.argv({"url": "http://host/data/0", "output": "page.html"})
    writeout = argv[argv.index("-w") + 1]
    assert "%{content_type}" in writeout
    assert "%{size_download}" in writeout
    assert argv[-1] == "http://host/data/0"  # the URL stays last; -w goes before it


def test_wget_output_boolean_word_streams_to_stdout():
    spec = build_registry().get("wget")
    argv = spec.argv({"url": "http://host/data/1", "output": "yes"})
    assert "yes" not in argv         # not taken as a filename
    assert "-qO" in argv and argv[argv.index("-qO") + 1] == "-"
    assert argv[-1] == "http://host/data/1"


def test_wget_output_real_filename_is_saved():
    spec = build_registry().get("wget")
    argv = spec.argv({"url": "http://host/download/0", "output": "loot.pcap"})
    assert argv[argv.index("-O") + 1] == "loot.pcap"


def test_wget_ignores_a_non_integer_timeout_instead_of_failing():
    spec = build_registry().get("wget")
    argv = spec.argv({"url": "http://host/", "timeout": "yes"})
    assert "--timeout" not in argv
    assert argv[-1] == "http://host/"


# --- remote command execution ----------------------------------------------


def test_ssh_runs_a_single_remote_command_non_interactively():
    spec = build_registry().get("ssh")
    argv = spec.argv({"user": "nathan", "host": "10.129.142.36",
                      "command": "id; sudo -l; cat user.txt"})
    assert "BatchMode=yes" in argv
    assert "nathan@10.129.142.36" in argv
    assert argv[-1] == "id; sudo -l; cat user.txt"


def test_sshpass_wraps_ssh_with_a_password():
    spec = build_registry().get("sshpass")
    argv = spec.argv({"user": "nathan", "host": "10.10.10.10",
                      "password": "Buck3tH4TF0RM3!", "command": "id"})
    assert argv[argv.index("-p") + 1] == "Buck3tH4TF0RM3!"
    assert "ssh" in argv
    assert "nathan@10.10.10.10" in argv
    assert argv[-1] == "id"


def test_ssh_with_a_password_routes_through_sshpass():
    # A weak model reaches for the ssh tool and passes password=... . ssh alone
    # runs BatchMode=yes, which refuses password auth non-interactively, so the
    # login always failed with exit 255. The ssh tool must redirect to sshpass
    # when a password (and no key) is given, and must NOT set BatchMode.
    spec = build_registry().get("ssh")
    argv = spec.argv({"user": "nathan", "host": "10.129.142.36",
                      "password": "Buck3tH4TF0RM3!", "command": "id; sudo -l"})
    assert _bin_stem(argv[0]) == "sshpass"      # routed through sshpass
    assert argv[argv.index("-p") + 1] == "Buck3tH4TF0RM3!"
    assert "BatchMode=yes" not in argv          # would block password auth
    assert "ssh" in argv
    assert "nathan@10.129.142.36" in argv
    assert argv[-1] == "id; sudo -l"


def test_ssh_with_a_key_still_runs_ssh_directly_in_batch_mode():
    # With a key (no password) ssh authenticates non-interactively, so it must
    # stay on ssh+BatchMode and not be rerouted through sshpass.
    spec = build_registry().get("ssh")
    argv = spec.argv({"user": "nathan", "host": "10.129.142.36",
                      "key": "/tmp/id_rsa", "command": "id"})
    assert _bin_stem(argv[0]) == "ssh"          # ssh directly, not sshpass
    assert "BatchMode=yes" in argv
    assert argv[argv.index("-i") + 1] == "/tmp/id_rsa"
    assert argv[-1] == "id"


# --- exploit research and AD attacks ---------------------------------------


def test_searchsploit_splits_terms_into_separate_tokens():
    spec = build_registry().get("searchsploit")
    argv = spec.argv({"terms": "apache 2.4.49"})
    assert "apache" in argv
    assert "2.4.49" in argv


def test_searchsploit_rejects_a_term_posing_as_an_option():
    spec = build_registry().get("searchsploit")
    with pytest.raises(ToolArgError):
        spec.argv({"terms": "apache --www"})


def test_getuserspns_accepts_credentials_with_special_characters():
    spec = build_registry().get("impacket-GetUserSPNs")
    target = "EXAMPLE.LOCAL/svc:P@ss!word"
    argv = spec.argv({"target": target, "dc_ip": "10.10.10.10"})
    assert target in argv
    assert "-request" in argv
    assert argv[argv.index("-dc-ip") + 1] == "10.10.10.10"


def test_ldapsearch_passes_a_filter_with_parentheses():
    spec = build_registry().get("ldapsearch")
    argv = spec.argv({
        "simple_auth": True,
        "uri": "ldap://10.10.10.10",
        "base": "dc=example,dc=com",
        "filter": "(objectClass=user)",
    })
    assert "ldap://10.10.10.10" in argv
    assert "dc=example,dc=com" in argv
    assert "(objectClass=user)" in argv


# --- nc cannot hang a banner grab ------------------------------------------
#
# ``nc -v host 21`` with no timeout blocks on stdin until the executor kills it
# two minutes later, wasting a step. The builder always binds a -w timeout.


def test_nc_injects_a_connection_timeout_by_default():
    spec = build_registry().get("nc")
    argv = spec.argv({"host": "10.129.142.36", "port": "21", "verbose": True})
    assert "-w" in argv
    secs = argv[argv.index("-w") + 1]
    assert secs.isdigit() and int(secs) >= 1
    assert argv[-2:] == ["10.129.142.36", "21"]


def test_nc_keeps_a_caller_supplied_timeout():
    spec = build_registry().get("nc")
    argv = spec.argv({"host": "10.10.10.10", "port": "9999", "wait": 3})
    assert argv[argv.index("-w") + 1] == "3"


# --- whatweb aggression: a word means "be aggressive" ----------------------
#
# whatweb accepts only -a 1|3|4. A model often passes a boolean-ish word or an
# out-of-range number; map anything that is not a valid level to 3 rather than
# failing the step, so fingerprinting still runs.


@pytest.mark.parametrize("word", ["yes", "true", "aggressive", "7"])
def test_whatweb_maps_a_non_choice_aggression_to_3(word):
    spec = build_registry().get("whatweb")
    argv = spec.argv({"target": "http://10.129.142.36", "aggression": word, "verbose": True})
    assert argv[argv.index("-a") + 1] == "3"
    assert "-v" in argv
    assert argv[-1] == "http://10.129.142.36"


@pytest.mark.parametrize("level", ["1", "3", "4"])
def test_whatweb_keeps_a_valid_aggression_level(level):
    spec = build_registry().get("whatweb")
    argv = spec.argv({"target": "http://host", "aggression": level})
    assert argv[argv.index("-a") + 1] == level


# --- wordlist fallback: a guessed path must not kill the step --------------
#
# A weak model guesses a wordlist path that does not exist (common/wordlist.txt)
# and gobuster/ffuf exit 1. The builder falls back to an installed default when
# the given path is missing, so enumeration still runs.


def test_gobuster_uses_a_given_wordlist_that_exists(monkeypatch):
    monkeypatch.setattr(registry_mod.os.path, "exists", lambda p: p == "/tmp/words.txt")
    spec = build_registry().get("gobuster")
    argv = spec.argv({"url": "http://host", "wordlist": "/tmp/words.txt"})
    assert argv[argv.index("-w") + 1] == "/tmp/words.txt"


def test_gobuster_falls_back_when_the_given_wordlist_is_missing(monkeypatch):
    fallback = registry_mod._DEFAULT_WORDLISTS[0]
    monkeypatch.setattr(registry_mod.os.path, "exists", lambda p: p == fallback)
    spec = build_registry().get("gobuster")
    argv = spec.argv({"url": "http://host", "wordlist": "/does/not/exist.txt"})
    assert argv[argv.index("-w") + 1] == fallback


def test_ffuf_falls_back_to_an_installed_wordlist(monkeypatch):
    fallback = registry_mod._DEFAULT_WORDLISTS[0]
    monkeypatch.setattr(registry_mod.os.path, "exists", lambda p: p == fallback)
    spec = build_registry().get("ffuf")
    argv = spec.argv({"url": "http://host/FUZZ"})
    assert argv[argv.index("-w") + 1] == fallback


# A saved capture only gives up cleartext credentials when its payload is shown.
# tcpdump must therefore accept the payload flags (-A/-X) and verbosity (-v)
# alongside reading a file, so a model can dump an FTP/HTTP packet's bytes.


def test_tcpdump_accepts_payload_and_verbose_flags():
    spec = build_registry().get("tcpdump")
    argv = spec.argv(
        {"read": "0.pcap", "no_resolve": True, "ascii": True, "hex": True, "verbose": True}
    )
    assert "-A" in argv
    assert "-X" in argv
    assert "-v" in argv
    assert argv[argv.index("-r") + 1] == "0.pcap"
