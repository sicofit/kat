"""Default catalog of command-line security tools.

Each entry is a :class:`~kat.tools.base.ToolSpec` that describes a real tool and
the arguments the agent may supply. Availability is resolved at run time with
``shutil.which`` by :class:`ToolSpec` and :class:`Registry`, so the catalog may
list tools that are not installed on a given host; those simply report as
unavailable rather than being omitted.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

from .base import ArgSpec, Registry, ToolArgError, ToolSpec, _check_ports, _check_target

# ---------------------------------------------------------------------------
# Builder helpers
#
# A builder receives the argument dictionary and returns the argument list that
# follows the binary. ArgSpec validation is bypassed for builders, so every
# value a builder places on the command line is sanitized here. Commands are run
# without a shell, but rejecting shell metacharacters and whitespace keeps each
# value a single, well-formed token and guards against option injection.
# ---------------------------------------------------------------------------

# Characters rejected in path and list values (wordlists, extensions, file
# names) assembled by a builder. Query and fragment characters (``? = % + #``)
# are permitted, because they are required to express real values and the
# command never reaches a shell; shell metacharacters, quotes, and whitespace
# are not. URL and credential values use :func:`_safe_url`, which is looser
# because a real URL legitimately carries those metacharacters.
_FORBIDDEN_CHARS = frozenset(" \t\n\r\x0b\x0c\"'`\\;|&$<>(){}[]*!")


def _safe(name: str, value: Any) -> str:
    """Return ``value`` as a single safe token or raise :class:`ToolArgError`."""
    if value in (None, "", False):
        raise ToolArgError(f"{name} is required")
    text = str(value).strip()
    if not text:
        raise ToolArgError(f"{name} is required")
    found = sorted(set(text) & _FORBIDDEN_CHARS)
    if found:
        shown = " ".join(repr(ch) for ch in found)
        raise ToolArgError(f"{name}: unsupported character(s): {shown}")
    if text.startswith("-"):
        raise ToolArgError(f"{name}: value must not begin with '-'")
    return text


# Whitespace and control characters are all that a URL or credential token may
# not contain: they would split the single argument or are invalid in a URL.
_URL_FORBIDDEN = frozenset(" \t\n\r\x0b\x0c")


def _safe_url(name: str, value: Any) -> str:
    """Return a URL or credential value as one token, or raise.

    The command is executed without a shell (``shell=False``), so the value is
    passed verbatim as a single argument and never reaches a shell. Metacharacters
    such as ``& ; | ! $`` that are meaningful in a shell are therefore inert here
    but essential in real URLs (query strings, FTP credentials, injection test
    payloads). Only whitespace and control characters are rejected, so the value
    stays a single well-formed token, and a leading ``-`` is refused so the value
    cannot be read by the binary as an option.
    """
    if value in (None, "", False):
        raise ToolArgError(f"{name} is required")
    text = str(value).strip()
    if not text:
        raise ToolArgError(f"{name} is required")
    found = sorted(set(text) & _URL_FORBIDDEN)
    if found:
        shown = " ".join(repr(ch) for ch in found)
        raise ToolArgError(f"{name}: unsupported character(s): {shown}")
    if text.startswith("-"):
        raise ToolArgError(f"{name}: value must not begin with '-'")
    return text


def _safe_int(name: str, value: Any) -> str:
    try:
        return str(int(value))
    except (TypeError, ValueError):
        raise ToolArgError(f"{name}: expected an integer") from None


# Only characters that would corrupt the request line itself (newlines, carriage
# returns, null) are rejected. Header values, POST bodies, and remote commands
# are single argv tokens under shell=False, so spaces and shell metacharacters
# are inert locally; a remote command is interpreted by the remote shell, which
# is the point. This keeps rich payloads expressible while the value stays well
# formed.
_TEXT_FORBIDDEN = frozenset("\n\r\x00")


def _safe_text(name: str, value: Any) -> str:
    """Return a free-form value (header, body, remote command) as one token."""
    if value in (None, "", False):
        raise ToolArgError(f"{name} is required")
    text = str(value)
    found = sorted(set(text) & _TEXT_FORBIDDEN)
    if found:
        shown = " ".join(repr(ch) for ch in found)
        raise ToolArgError(f"{name}: unsupported character(s): {shown}")
    return text


def _as_list(value: Any) -> list[Any]:
    """Normalise a value that may be a single item or a list into a list."""
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


# Words a model reaches for when it means "yes, save it" rather than naming a
# file. They are not filenames.
# Values that are not filenames but a request to SEE the body in the observation:
# boolean-ish confirmations ("yes" = "yes, save it") and display words ("body",
# "stdout"). A weak model reaches for all of these meaning "show me", and taken as
# filenames they wrote every response to one junk file the model could never read,
# so it never saw a page body and could not follow its links.
_SHOW_WORDS = frozenset({
    "yes", "y", "true", "on", "no", "n", "false", "off",
    "body", "response", "stdout", "print", "show", "content", "console", "inline",
})


def _resolve_output(value: Any) -> str | None:
    """Interpret an ``output`` argument as a real file path, or as "no file".

    ``output`` names a file to save the response body to. When the value is one
    of :data:`_SHOW_WORDS` (or empty/boolean) it is not a filename but "show me
    the body", so this returns ``None`` -- no file -- and the body is printed for
    the agent to read in the observation. A genuine path is returned sanitised,
    for saving a binary such as a packet capture that another tool will open.
    """
    if value in (None, "", True, False):
        return None
    text = str(value).strip()
    if not text or text.lower() in _SHOW_WORDS:
        return None
    return _safe("output", text)


# Wordlists that ship with Kali / common pentest images, tried in order when the
# caller's wordlist does not exist on this host.
_DEFAULT_WORDLISTS = (
    "/usr/share/seclists/Discovery/Web-Content/common.txt",
    "/usr/share/seclists/Discovery/Web-Content/raft-small-words.txt",
    "/usr/share/wordlists/dirb/common.txt",
)


def _resolve_wordlist(value: Any) -> str:
    """Return a usable wordlist path.

    A weak model often names a wordlist that is not installed (for example
    ``common/wordlist.txt``), which makes the whole scan exit immediately with
    no findings. If the caller's path exists it is used; otherwise the first
    installed default is substituted so the scan still runs. When nothing is
    found the caller's value is returned unchanged, so the failure is honest.
    """
    given = _safe("wordlist", value) if value not in (None, "", False) else None
    if given and os.path.exists(given):
        return given
    for path in _DEFAULT_WORDLISTS:
        if os.path.exists(path):
            return path
    if given:
        return given
    raise ToolArgError("wordlist is required")


def _build_curl(args: Mapping[str, Any]) -> list[str]:
    tail: list[str] = []
    if args.get("head"):
        tail.append("-I")
    if args.get("include"):
        tail.append("-i")
    if args.get("follow"):
        tail.append("-L")
    if args.get("insecure"):
        tail.append("-k")
    method = args.get("method")
    if method:
        verb = str(method).strip().upper()
        if verb not in {"GET", "HEAD", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"}:
            raise ToolArgError(f"method: {method!r} is not a supported HTTP method")
        tail += ["-X", verb]
    for header in _as_list(args.get("header") or []):
        tail += ["-H", _safe_text("header", header)]
    if args.get("cookie"):
        tail += ["-b", _safe_text("cookie", args["cookie"])]
    if args.get("data"):
        # A request body; curl issues a POST unless a method was set explicitly.
        tail += ["--data", _safe_text("data", args["data"])]
    output = _resolve_output(args.get("output"))
    if output:
        # A real filename: save the body to it (for example a packet capture)
        # so another tool can read it, staying silent so the progress meter does
        # not drown the log. With no real filename the body prints normally and
        # the agent reads it in the observation. Even when saving, -w prints the
        # size and content-type to stdout so the step is not blind: a page saved
        # to a file by mistake announces itself as text/* (re-fetch and read it),
        # while a real artefact announces its binary type (open it with a reader).
        tail += ["-s", "-o", output, "-w",
                 "[saved %{size_download} bytes, content-type %{content_type}]\\n"]
    tail.append(_safe_url("url", args.get("url")))
    return tail


def _build_wget(args: Mapping[str, Any]) -> list[str]:
    tail: list[str] = []
    if args.get("recursive"):
        tail.append("-r")
    if args.get("no_check_certificate"):
        tail.append("--no-check-certificate")
    timeout = args.get("timeout")
    if timeout not in (None, "", True, False) and str(timeout).strip().lower() not in _SHOW_WORDS:
        try:
            tail += ["--timeout", str(int(str(timeout).strip()))]
        except ValueError:
            pass  # ignore a non-integer timeout rather than failing the fetch
    output = _resolve_output(args.get("output"))
    if output:
        tail += ["-O", output]
    else:
        # No real filename: write the body to stdout (quietly) so the agent can
        # read it, mirroring curl, instead of inventing a file it cannot inspect.
        tail += ["-qO", "-"]
    tail.append(_safe_url("url", args.get("url")))
    return tail


def _build_gobuster(args: Mapping[str, Any]) -> list[str]:
    # The directory module is a positional subcommand that precedes its options.
    tail = [
        "dir",
        "-u", _safe_url("url", args.get("url")),
        "-w", _resolve_wordlist(args.get("wordlist")),
    ]
    if args.get("extensions"):
        tail += ["-x", _safe("extensions", args["extensions"])]
    if args.get("threads"):
        tail += ["-t", _safe_int("threads", args["threads"])]
    if args.get("insecure"):
        tail.append("-k")
    return tail


def _build_nc(args: Mapping[str, Any]) -> list[str]:
    # A bare banner grab (nc -v host 21) blocks on stdin until the executor's
    # timeout, wasting a step and two minutes. Always bind a short connection
    # timeout with -w so the call returns the banner and ends: use the caller's
    # value when it is a positive integer, otherwise a small default.
    tail: list[str] = []
    if args.get("verbose"):
        tail.append("-v")
    if args.get("zero_io"):
        tail.append("-z")
    wait = args.get("wait")
    try:
        secs = int(str(wait).strip())
        if secs < 1:
            secs = 5
    except (TypeError, ValueError):
        secs = 5
    tail += ["-w", str(secs)]
    tail.append(_check_target("host", args.get("host")))
    tail.append(_check_ports("port", args.get("port")))
    return tail


def _build_whatweb(args: Mapping[str, Any]) -> list[str]:
    # whatweb accepts only -a 1|3|4. A model often passes a boolean-ish word
    # meaning "be aggressive"; map that to 3 rather than failing the step.
    tail: list[str] = []
    aggression = args.get("aggression")
    if aggression not in (None, "", True, False):
        level = str(aggression).strip()
        if level not in {"1", "3", "4"}:
            level = "3"
        tail += ["-a", level]
    if args.get("verbose"):
        tail.append("-v")
    tail.append(_safe_url("target", args.get("target")))
    return tail


def _build_ffuf(args: Mapping[str, Any]) -> list[str]:
    # The wordlist is substituted wherever the FUZZ keyword appears; append it to
    # the path when the caller did not place it explicitly.
    url = _safe_url("url", args.get("url"))
    if "FUZZ" not in url:
        url = url.rstrip("/") + "/FUZZ"
    tail = ["-u", url, "-w", _resolve_wordlist(args.get("wordlist"))]
    if args.get("extensions"):
        tail += ["-e", _safe("extensions", args["extensions"])]
    if args.get("match_codes"):
        tail += ["-mc", _safe("match_codes", args["match_codes"])]
    if args.get("threads"):
        tail += ["-t", _safe_int("threads", args["threads"])]
    return tail


def _build_sqlmap(args: Mapping[str, Any]) -> list[str]:
    # ``-u`` carries the full URL, including the parameter under test; ``--batch``
    # keeps the run non-interactive.
    tail = ["-u", _safe_url("url", args.get("url")), "--batch"]
    if args.get("level"):
        tail += ["--level", _safe_int("level", args["level"])]
    if args.get("risk"):
        tail += ["--risk", _safe_int("risk", args["risk"])]
    if args.get("dbs"):
        tail.append("--dbs")
    if args.get("tables"):
        tail.append("--tables")
    if args.get("dump"):
        tail.append("--dump")
    return tail


def _build_john(args: Mapping[str, Any]) -> list[str]:
    # John expects its long options in ``--name=value`` form, which a declarative
    # option (rendered as two tokens) cannot produce.
    tail: list[str] = []
    if args.get("wordlist"):
        tail.append("--wordlist=" + _safe("wordlist", args["wordlist"]))
    if args.get("format"):
        tail.append("--format=" + _safe("format", args["format"]))
    if args.get("rules"):
        tail.append("--rules")
    if args.get("show"):
        tail.append("--show")
    tail.append(_safe("hashfile", args.get("hashfile")))
    return tail


_SSH_HARDENING = [
    "-o", "StrictHostKeyChecking=no",
    "-o", "UserKnownHostsFile=/dev/null",
]


def _build_ssh(args: Mapping[str, Any]):
    # One non-interactive command over SSH. With a password (and no key) ssh
    # cannot authenticate non-interactively -- BatchMode blocks the prompt and the
    # login always fails -- so redirect through sshpass, which feeds the password
    # in. A weak model reaches for the ssh tool with a password; make that work
    # instead of silently dropping it. With a key or the agent, run ssh directly.
    if args.get("password") and not args.get("key"):
        return ("sshpass", _build_sshpass(args))
    tail = [*_SSH_HARDENING, "-o", "BatchMode=yes"]
    if args.get("port"):
        tail += ["-p", _safe_int("port", args["port"])]
    if args.get("key"):
        tail += ["-i", _safe("key", args["key"])]
    user = _safe("user", args.get("user"))
    host = _safe("host", args.get("host"))
    tail.append(f"{user}@{host}")
    tail.append(_safe_text("command", args.get("command")))
    return tail


def _build_sshpass(args: Mapping[str, Any]) -> list[str]:
    # Password authentication wrapper around ssh for one remote command. sshpass
    # supplies the password so the ssh prompt never blocks.
    tail = ["-p", _safe_text("password", args.get("password")), "ssh", *_SSH_HARDENING]
    if args.get("port"):
        tail += ["-p", _safe_int("port", args["port"])]
    user = _safe("user", args.get("user"))
    host = _safe("host", args.get("host"))
    tail.append(f"{user}@{host}")
    tail.append(_safe_text("command", args.get("command")))
    return tail


def _build_searchsploit(args: Mapping[str, Any]) -> list[str]:
    # Each whitespace-separated term is its own token (an AND search). Reject a
    # term that would pose as an option.
    terms = _safe_text("terms", args.get("terms")).split()
    if not terms:
        raise ToolArgError("terms is required")
    for term in terms:
        if term.startswith("-"):
            raise ToolArgError(f"terms: a term must not begin with '-': {term!r}")
    tail: list[str] = []
    if args.get("cve"):
        tail.append("--cve")
    return tail + terms


def _impacket_target(args: Mapping[str, Any]) -> str:
    # Connection string such as DOMAIN/user:password@host; the password may carry
    # characters a shell would treat as special, which is safe under shell=False.
    return _safe_url("target", args.get("target"))


def _build_getnpusers(args: Mapping[str, Any]) -> list[str]:
    # Hunt for AS-REP-roastable accounts (no pre-auth) and request their hashes.
    tail = [_impacket_target(args), "-format", "hashcat"]
    if args.get("usersfile"):
        tail += ["-usersfile", _safe("usersfile", args["usersfile"])]
    if args.get("no_pass"):
        tail.append("-no-pass")
    if args.get("request"):
        tail.append("-request")
    if args.get("dc_ip"):
        tail += ["-dc-ip", _safe("dc_ip", args["dc_ip"])]
    return tail


def _build_getuserspns(args: Mapping[str, Any]) -> list[str]:
    # Kerberoast: request service-ticket hashes for accounts with an SPN.
    tail = [_impacket_target(args), "-request"]
    if args.get("dc_ip"):
        tail += ["-dc-ip", _safe("dc_ip", args["dc_ip"])]
    return tail


def _build_secretsdump(args: Mapping[str, Any]) -> list[str]:
    # Dump secrets (SAM, LSA, or domain hashes over DRSUAPI with -just-dc).
    tail = [_impacket_target(args)]
    if args.get("just_dc"):
        tail.append("-just-dc")
    if args.get("dc_ip"):
        tail += ["-dc-ip", _safe("dc_ip", args["dc_ip"])]
    return tail


def build_registry() -> Registry:
    """Build the registry from the bundled catalog of security tools."""
    specs: list[ToolSpec] = [
        # -- Reconnaissance ------------------------------------------------
        ToolSpec(
            name="nmap",
            binary="nmap",
            summary="Network port scanner with service, version, and script detection.",
            category="recon",
            args=(
                ArgSpec("ports", token="-p", validate="port",
                        help="Ports or ranges to scan, for example 22,80,443 or 1-1024."),
                ArgSpec("min_rate", token="--min-rate", validate="int",
                        help="Minimum number of packets to send per second."),
                ArgSpec("service_scan", kind="flag", token="-sV",
                        help="Identify the service and version behind open ports."),
                ArgSpec("default_scripts", kind="flag", token="-sC",
                        help="Run the default script set."),
                ArgSpec("aggressive", kind="flag", token="-A",
                        help="Enable OS and version detection, scripts, and traceroute."),
                ArgSpec("no_ping", kind="flag", token="-Pn",
                        help="Skip host discovery and treat the host as online."),
                ArgSpec("udp_scan", kind="flag", token="-sU",
                        help="Scan UDP ports instead of TCP."),
                ArgSpec("target", kind="positional", required=True, validate="target",
                        help="Host, address, or CIDR range to scan."),
            ),
        ),
        ToolSpec(
            name="masscan",
            binary="masscan",
            summary="High-rate scanner for sweeping large address and port ranges.",
            category="recon",
            args=(
                ArgSpec("ports", token="-p", validate="port",
                        help="Ports or ranges to scan."),
                ArgSpec("top_ports", token="--top-ports", validate="int",
                        help="Scan the given number of most common ports."),
                ArgSpec("rate", token="--rate", validate="int",
                        help="Transmit rate in packets per second."),
                ArgSpec("target", kind="positional", required=True, validate="target",
                        help="Address or CIDR range to scan."),
            ),
        ),
        ToolSpec(
            name="arp-scan",
            binary="arp-scan",
            summary="Discover hosts on the local network using ARP requests.",
            category="recon",
            args=(
                ArgSpec("localnet", kind="flag", token="--localnet",
                        help="Scan the whole network attached to the interface."),
                ArgSpec("interface", token="-I", validate="target",
                        help="Network interface to send from, for example eth0."),
                ArgSpec("target", kind="positional", validate="target",
                        help="Address or CIDR range to scan when not using --localnet."),
            ),
        ),
        # -- DNS -----------------------------------------------------------
        ToolSpec(
            name="whois",
            binary="whois",
            summary="Look up registration records for a domain or address.",
            category="dns",
            args=(
                ArgSpec("server", token="-h", validate="target",
                        help="Specific whois server to query."),
                ArgSpec("target", kind="positional", required=True, validate="target",
                        help="Domain name or IP address to look up."),
            ),
        ),
        ToolSpec(
            name="dig",
            binary="dig",
            summary="Resolve DNS records for a name and record type.",
            category="dns",
            args=(
                ArgSpec("name", kind="positional", required=True, validate="target",
                        help="Name to resolve."),
                ArgSpec("type", kind="positional",
                        choices=("A", "AAAA", "MX", "NS", "TXT", "CNAME",
                                 "SOA", "PTR", "SRV", "ANY"),
                        help="Record type to request."),
                ArgSpec("short", kind="flag", token="+short",
                        help="Print only the concise answer."),
                ArgSpec("trace", kind="flag", token="+trace",
                        help="Trace the delegation path from the root servers."),
            ),
        ),
        ToolSpec(
            name="dnsrecon",
            binary="dnsrecon",
            summary="Enumerate DNS records, zones, and subdomains for a domain.",
            category="dns",
            args=(
                ArgSpec("domain", token="-d", required=True, validate="target",
                        help="Target domain."),
                ArgSpec("type", token="-t",
                        choices=("std", "rvl", "brt", "srv", "axfr", "zonewalk"),
                        help="Enumeration type to perform."),
                ArgSpec("nameserver", token="-n", validate="target",
                        help="Name server to query."),
                ArgSpec("wordlist", token="-D", validate="target",
                        help="Path to a subdomain wordlist for brute enumeration."),
            ),
        ),
        # -- Web -----------------------------------------------------------
        ToolSpec(
            name="curl",
            binary="curl",
            summary="HTTP client for issuing requests to a URL and reading the response.",
            category="web",
            builder=_build_curl,
            args=(
                ArgSpec("url", kind="positional", required=True,
                        help="Target URL, including any query string, credentials, or payload."),
                ArgSpec("method",
                        choices=("GET", "HEAD", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"),
                        help="HTTP method; GET is used when omitted."),
                ArgSpec("output",
                        help="Optional file path to save a binary body to (for example a "
                             "packet capture); omit it to read the body in the response."),
                ArgSpec("header",
                        help="Request header 'Name: value'; may be a list for several."),
                ArgSpec("cookie",
                        help="Cookie string to send, for example 'session=abc123'."),
                ArgSpec("data",
                        help="Request body to send; issues a POST unless a method is set."),
                ArgSpec("head", kind="flag", help="Request response headers only."),
                ArgSpec("include", kind="flag", help="Include response headers in the output."),
                ArgSpec("follow", kind="flag", help="Follow HTTP redirects."),
                ArgSpec("insecure", kind="flag", help="Do not validate the server certificate."),
            ),
        ),
        ToolSpec(
            name="wget",
            binary="wget",
            summary="Retrieve files and pages from a URL over HTTP or FTP.",
            category="web",
            builder=_build_wget,
            args=(
                ArgSpec("recursive", kind="flag", token="-r",
                        help="Download recursively."),
                ArgSpec("output", token="-O",
                        help="Optional file path to save a binary download to; omit it to "
                             "read the body in the response."),
                ArgSpec("timeout", token="--timeout",
                        help="Network timeout in seconds; ignored if not an integer."),
                ArgSpec("no_check_certificate", kind="flag", token="--no-check-certificate",
                        help="Do not validate the server certificate."),
                ArgSpec("url", kind="positional", required=True, validate="target",
                        help="URL to retrieve."),
            ),
        ),
        ToolSpec(
            name="whatweb",
            binary="whatweb",
            summary="Identify web technologies, frameworks, and server software.",
            category="web",
            builder=_build_whatweb,
            args=(
                ArgSpec("aggression", token="-a", choices=("1", "3", "4"),
                        help="Aggression level: 1 passive, 3 aggressive, 4 heavy "
                             "(a word like 'yes' maps to 3)."),
                ArgSpec("verbose", kind="flag", token="-v",
                        help="Report detailed plugin output."),
                ArgSpec("target", kind="positional", required=True, validate="target",
                        help="URL or host to fingerprint."),
            ),
        ),
        ToolSpec(
            name="nikto",
            binary="nikto",
            summary="Scan a web server for known issues and misconfigurations.",
            category="web",
            args=(
                ArgSpec("host", token="-h", required=True, validate="target",
                        help="Target host or URL."),
                ArgSpec("port", token="-p", validate="port",
                        help="Port to scan."),
                ArgSpec("ssl", kind="flag", token="-ssl",
                        help="Force a TLS connection."),
                ArgSpec("tuning", token="-Tuning",
                        help="Restrict the scan to selected test categories."),
            ),
        ),
        ToolSpec(
            name="gobuster",
            binary="gobuster",
            summary="Discover web content by requesting paths from a wordlist.",
            category="web",
            builder=_build_gobuster,
            args=(
                ArgSpec("url", required=True, help="Base URL to scan."),
                ArgSpec("wordlist", required=True, help="Path to the wordlist."),
                ArgSpec("extensions", help="Comma-separated file extensions to append."),
                ArgSpec("threads", validate="int", help="Number of concurrent requests."),
                ArgSpec("insecure", kind="flag", help="Do not validate the server certificate."),
            ),
        ),
        ToolSpec(
            name="ffuf",
            binary="ffuf",
            summary="Discover web content and parameters by fuzzing with a wordlist.",
            category="web",
            builder=_build_ffuf,
            args=(
                ArgSpec("url", required=True,
                        help="Target URL; the FUZZ keyword is appended if absent."),
                ArgSpec("wordlist", required=True, help="Path to the wordlist."),
                ArgSpec("extensions", help="Comma-separated extensions to append to each word."),
                ArgSpec("match_codes", help="Comma-separated HTTP status codes to treat as matches."),
                ArgSpec("threads", validate="int", help="Number of concurrent requests."),
            ),
        ),
        ToolSpec(
            name="feroxbuster",
            binary="feroxbuster",
            summary="Recursively discover web content using a wordlist.",
            category="web",
            args=(
                ArgSpec("url", token="-u", required=True, validate="target",
                        help="Base URL to scan."),
                ArgSpec("wordlist", token="-w", required=True, validate="target",
                        help="Path to the wordlist."),
                ArgSpec("extensions", token="-x",
                        help="Comma-separated file extensions to request."),
                ArgSpec("threads", token="-t", validate="int",
                        help="Number of concurrent requests."),
                ArgSpec("depth", token="-d", validate="int",
                        help="Maximum recursion depth."),
                ArgSpec("insecure", kind="flag", token="-k",
                        help="Do not validate the server certificate."),
            ),
        ),
        ToolSpec(
            name="sqlmap",
            binary="sqlmap",
            summary="Test a URL parameter for SQL injection and enumerate the database.",
            category="web",
            builder=_build_sqlmap,
            args=(
                ArgSpec("url", required=True,
                        help="Target URL, including the parameter to test."),
                ArgSpec("level", validate="int", help="Test level from 1 to 5."),
                ArgSpec("risk", validate="int", help="Risk level from 1 to 3."),
                ArgSpec("dbs", kind="flag", help="Enumerate the available databases."),
                ArgSpec("tables", kind="flag", help="Enumerate tables in the database."),
                ArgSpec("dump", kind="flag", help="Retrieve the contents of the selected table."),
            ),
        ),
        # -- SMB and network services -------------------------------------
        ToolSpec(
            name="smbclient",
            binary="smbclient",
            summary="Access SMB and CIFS shares to list and read files.",
            category="smb",
            args=(
                ArgSpec("list", token="-L", validate="target",
                        help="List the shares offered by this host."),
                ArgSpec("no_pass", kind="flag", token="-N",
                        help="Connect without prompting for a password."),
                ArgSpec("user", token="-U",
                        help="Credentials as user or user%password."),
                ArgSpec("command", token="-c",
                        help="Run a client command non-interactively, for example 'ls'."),
                ArgSpec("service", kind="positional", validate="target",
                        help="Share path such as //host/share."),
            ),
        ),
        ToolSpec(
            name="rpcclient",
            binary="rpcclient",
            summary="Issue MS-RPC queries to enumerate Windows hosts.",
            category="smb",
            args=(
                ArgSpec("user", token="-U",
                        help="Credentials as user or user%password."),
                ArgSpec("no_pass", kind="flag", token="-N",
                        help="Connect without prompting for a password."),
                ArgSpec("command", token="-c",
                        help="RPC command to run, for example 'enumdomusers'."),
                ArgSpec("host", kind="positional", required=True, validate="target",
                        help="Target host or address."),
            ),
        ),
        ToolSpec(
            name="enum4linux-ng",
            binary="enum4linux-ng",
            summary="Enumerate information exposed by SMB and related services.",
            category="smb",
            args=(
                ArgSpec("all", kind="flag", token="-A",
                        help="Run the full set of simple enumeration checks."),
                ArgSpec("user", token="-u", help="Username for authenticated checks."),
                ArgSpec("password", token="-p", help="Password for authenticated checks."),
                ArgSpec("host", kind="positional", required=True, validate="target",
                        help="Target host or address."),
            ),
        ),
        ToolSpec(
            name="netexec",
            binary="netexec",
            summary="Enumerate and test authentication across network services.",
            category="smb",
            args=(
                ArgSpec("protocol", kind="positional", required=True,
                        choices=("smb", "ssh", "winrm", "ldap", "mssql",
                                 "rdp", "ftp", "wmi", "nfs"),
                        help="Protocol module to use."),
                ArgSpec("target", kind="positional", required=True, validate="target",
                        help="Target host, address, or range."),
                ArgSpec("user", token="-u", help="Username or path to a username list."),
                ArgSpec("password", token="-p", help="Password or path to a password list."),
                ArgSpec("hash", token="-H", help="NT hash to authenticate with instead of a password."),
                ArgSpec("shares", kind="flag", token="--shares",
                        help="Enumerate shares over SMB."),
                ArgSpec("command", token="-x",
                        help="Command to run on the target over the chosen protocol."),
                ArgSpec("local_auth", kind="flag", token="--local-auth",
                        help="Authenticate against the local account database, not the domain."),
            ),
        ),
        ToolSpec(
            name="nc",
            binary="nc",
            summary="Open and inspect raw TCP or UDP connections.",
            category="net",
            builder=_build_nc,
            args=(
                ArgSpec("verbose", kind="flag", token="-v",
                        help="Report connection progress."),
                ArgSpec("zero_io", kind="flag", token="-z",
                        help="Report whether a port is open without sending data."),
                ArgSpec("wait", token="-w", validate="int",
                        help="Connection timeout in seconds (a short default is "
                             "applied so the connection cannot hang)."),
                ArgSpec("host", kind="positional", required=True, validate="target",
                        help="Target host or address."),
                ArgSpec("port", kind="positional", required=True, validate="port",
                        help="Target port."),
            ),
        ),
        # -- Credentials ---------------------------------------------------
        ToolSpec(
            name="hydra",
            binary="hydra",
            summary="Test network login services using supplied credentials.",
            category="creds",
            args=(
                ArgSpec("login", token="-l", help="Single username to try."),
                ArgSpec("login_list", token="-L", validate="target",
                        help="Path to a list of usernames."),
                ArgSpec("password", token="-p", help="Single password to try."),
                ArgSpec("password_list", token="-P", validate="target",
                        help="Path to a list of passwords."),
                ArgSpec("target", kind="positional", required=True, validate="target",
                        help="Target host or address."),
                ArgSpec("service", kind="positional", required=True,
                        choices=("ssh", "ftp", "smb", "rdp", "mysql", "postgres",
                                 "vnc", "telnet", "pop3", "imap", "smtp"),
                        help="Service module to test."),
            ),
        ),
        ToolSpec(
            name="john",
            binary="john",
            summary="Recover passwords from hashes using dictionary and rule-based attacks.",
            category="creds",
            active=False,
            builder=_build_john,
            args=(
                ArgSpec("hashfile", kind="positional", required=True,
                        help="Path to the file containing the hashes."),
                ArgSpec("wordlist", help="Path to a wordlist for dictionary mode."),
                ArgSpec("format", help="Hash format name, for example raw-md5 or nt."),
                ArgSpec("rules", kind="flag", help="Apply the default mangling rules."),
                ArgSpec("show", kind="flag", help="Show passwords already recovered."),
            ),
        ),
        ToolSpec(
            name="hashcat",
            binary="hashcat",
            summary="Recover passwords from hashes using accelerated attacks.",
            category="creds",
            active=False,
            args=(
                ArgSpec("hash_mode", token="-m", validate="int",
                        help="Hash type code, for example 0 for MD5 or 1000 for NTLM."),
                ArgSpec("attack_mode", token="-a", validate="int",
                        help="Attack mode, for example 0 for a dictionary attack."),
                ArgSpec("hashfile", kind="positional", required=True, validate="target",
                        help="Path to the file containing the hashes."),
                ArgSpec("wordlist", kind="positional", validate="target",
                        help="Path to the wordlist for dictionary attacks."),
            ),
        ),
        # -- Forensics and file analysis ----------------------------------
        ToolSpec(
            name="strings",
            binary="strings",
            summary="Extract printable character sequences from a file.",
            category="forensics",
            active=False,
            args=(
                ArgSpec("min_len", token="-n", validate="int",
                        help="Minimum sequence length to report."),
                ArgSpec("file", kind="positional", required=True, validate="target",
                        help="Path to the file to examine."),
            ),
        ),
        ToolSpec(
            name="file",
            binary="file",
            summary="Identify a file's type from its contents.",
            category="forensics",
            active=False,
            args=(
                ArgSpec("mime", kind="flag", token="-i",
                        help="Report the MIME type instead of a description."),
                ArgSpec("path", kind="positional", required=True, validate="target",
                        help="Path to the file to identify."),
            ),
        ),
        ToolSpec(
            name="binwalk",
            binary="binwalk",
            summary="Analyze binary files for embedded data and file systems.",
            category="forensics",
            active=False,
            args=(
                ArgSpec("extract", kind="flag", token="-e",
                        help="Extract recognized content to disk."),
                ArgSpec("file", kind="positional", required=True, validate="target",
                        help="Path to the file to analyze."),
            ),
        ),
        ToolSpec(
            name="exiftool",
            binary="exiftool",
            summary="Read embedded metadata from media and document files.",
            category="forensics",
            active=False,
            args=(
                ArgSpec("file", kind="positional", required=True, validate="target",
                        help="Path to the file whose metadata should be read."),
            ),
        ),
        ToolSpec(
            name="tcpdump",
            binary="tcpdump",
            summary="Capture and inspect network packets from an interface or file.",
            category="forensics",
            active=False,
            args=(
                ArgSpec("interface", token="-i", validate="target",
                        help="Interface to capture on."),
                ArgSpec("count", token="-c", validate="int",
                        help="Stop after capturing this many packets."),
                ArgSpec("no_resolve", kind="flag", token="-n",
                        help="Do not resolve addresses or ports to names."),
                ArgSpec("ascii", kind="flag", token="-A",
                        help="Print each packet's payload as ASCII - reveals cleartext "
                             "protocol data such as FTP/HTTP credentials."),
                ArgSpec("hex", kind="flag", token="-X",
                        help="Print each packet's payload in hex and ASCII."),
                ArgSpec("verbose", kind="flag", token="-v",
                        help="Show more protocol detail per packet."),
                ArgSpec("read", token="-r", validate="target",
                        help="Read packets from a saved capture file."),
                ArgSpec("write", token="-w", validate="target",
                        help="Write captured packets to this file path."),
                ArgSpec("filter", kind="positional",
                        help="Packet filter expression, for example 'tcp port 80'."),
            ),
        ),
        ToolSpec(
            name="tshark",
            binary="tshark",
            summary="Capture and analyze network traffic with display filters.",
            category="forensics",
            active=False,
            args=(
                ArgSpec("interface", token="-i", validate="target",
                        help="Interface to capture on."),
                ArgSpec("count", token="-c", validate="int",
                        help="Stop after capturing this many packets."),
                ArgSpec("read", token="-r", validate="target",
                        help="Read packets from a saved capture file."),
                ArgSpec("display_filter", token="-Y",
                        help="Display filter expression, for example 'http.request'."),
            ),
        ),
        # -- Remote command execution -------------------------------------
        ToolSpec(
            name="ssh",
            binary="ssh",
            summary="Run a single command on a host over SSH using a key, the agent, or a password.",
            category="net",
            builder=_build_ssh,
            args=(
                ArgSpec("user", required=True, help="Username to log in as."),
                ArgSpec("host", required=True, help="Target host or address."),
                ArgSpec("command", required=True,
                        help="Command to run on the remote host, for example 'id; sudo -l'."),
                ArgSpec("password", help="Password for the account (authenticates via sshpass)."),
                ArgSpec("key", help="Path to a private key file for authentication."),
                ArgSpec("port", validate="int", help="SSH port if not 22."),
            ),
        ),
        ToolSpec(
            name="sshpass",
            binary="sshpass",
            summary="Run a single command over SSH using a password (non-interactive).",
            category="net",
            builder=_build_sshpass,
            args=(
                ArgSpec("user", required=True, help="Username to log in as."),
                ArgSpec("host", required=True, help="Target host or address."),
                ArgSpec("password", required=True, help="Password for the account."),
                ArgSpec("command", required=True,
                        help="Command to run on the remote host, for example 'id; sudo -l'."),
                ArgSpec("port", validate="int", help="SSH port if not 22."),
            ),
        ),
        ToolSpec(
            name="ldapsearch",
            binary="ldapsearch",
            summary="Query an LDAP or Active Directory server for directory objects.",
            category="smb",
            args=(
                ArgSpec("simple_auth", kind="flag", token="-x",
                        help="Use simple authentication (or anonymous bind)."),
                ArgSpec("uri", token="-H", validate="target",
                        help="Server URI such as ldap://10.10.10.10."),
                ArgSpec("base", token="-b",
                        help="Search base DN, for example dc=example,dc=com."),
                ArgSpec("bind_dn", token="-D",
                        help="Bind DN to authenticate as."),
                ArgSpec("password", token="-w",
                        help="Password for the bind DN."),
                ArgSpec("filter", kind="positional",
                        help="LDAP filter, for example (objectClass=user)."),
            ),
        ),
        # -- Exploit research and scanning --------------------------------
        ToolSpec(
            name="searchsploit",
            binary="searchsploit",
            summary="Search the Exploit-DB archive for known exploits by keyword.",
            category="recon",
            active=False,
            builder=_build_searchsploit,
            args=(
                ArgSpec("terms", required=True,
                        help="Search terms, for example 'apache 2.4.49'."),
                ArgSpec("cve", kind="flag", help="Treat the terms as a CVE identifier."),
            ),
        ),
        ToolSpec(
            name="nuclei",
            binary="nuclei",
            summary="Scan a target with community templates for known vulnerabilities.",
            category="web",
            args=(
                ArgSpec("url", token="-u", required=True, validate="target",
                        help="Target URL to scan."),
                ArgSpec("severity", token="-severity",
                        help="Restrict to severities, for example critical,high."),
                ArgSpec("tags", token="-tags",
                        help="Restrict to template tags, for example cve,rce."),
            ),
        ),
        ToolSpec(
            name="wpscan",
            binary="wpscan",
            summary="Scan a WordPress site for users, plugins, and known issues.",
            category="web",
            args=(
                ArgSpec("url", token="--url", required=True, validate="target",
                        help="WordPress site URL."),
                ArgSpec("enumerate", token="-e",
                        help="What to enumerate, for example vp,vt,u."),
                ArgSpec("api_token", token="--api-token",
                        help="WPScan API token for vulnerability data."),
            ),
        ),
        # -- Active Directory credential attacks --------------------------
        ToolSpec(
            name="impacket-GetNPUsers",
            binary="impacket-GetNPUsers",
            summary="Find AS-REP-roastable accounts and request their hashes.",
            category="creds",
            builder=_build_getnpusers,
            args=(
                ArgSpec("target", required=True,
                        help="Connection string, for example EXAMPLE.LOCAL/ or EXAMPLE.LOCAL/user."),
                ArgSpec("usersfile", help="Path to a list of usernames to test."),
                ArgSpec("no_pass", kind="flag", help="Do not send a password (roasting)."),
                ArgSpec("request", kind="flag", help="Request the AS-REP hash for each account."),
                ArgSpec("dc_ip", help="Domain controller address."),
            ),
        ),
        ToolSpec(
            name="impacket-GetUserSPNs",
            binary="impacket-GetUserSPNs",
            summary="Kerberoast: request service-ticket hashes for accounts with an SPN.",
            category="creds",
            builder=_build_getuserspns,
            args=(
                ArgSpec("target", required=True,
                        help="Connection string, for example EXAMPLE.LOCAL/user:password."),
                ArgSpec("dc_ip", help="Domain controller address."),
            ),
        ),
        ToolSpec(
            name="impacket-secretsdump",
            binary="impacket-secretsdump",
            summary="Dump SAM, LSA, or domain credential hashes from a host.",
            category="creds",
            builder=_build_secretsdump,
            args=(
                ArgSpec("target", required=True,
                        help="Connection string, for example EXAMPLE.LOCAL/user:password@host."),
                ArgSpec("just_dc", kind="flag",
                        help="Dump domain hashes over DRSUAPI (NTDS) only."),
                ArgSpec("dc_ip", help="Domain controller address."),
            ),
        ),
    ]
    return Registry(specs)
