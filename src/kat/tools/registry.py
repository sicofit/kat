"""Default catalog of command-line security tools.

Each entry is a :class:`~kat.tools.base.ToolSpec` that describes a real tool and
the arguments the agent may supply. Availability is resolved at run time with
``shutil.which`` by :class:`ToolSpec` and :class:`Registry`, so the catalog may
list tools that are not installed on a given host; those simply report as
unavailable rather than being omitted.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .base import ArgSpec, Registry, ToolArgError, ToolSpec

# ---------------------------------------------------------------------------
# Builder helpers
#
# A builder receives the argument dictionary and returns the argument list that
# follows the binary. ArgSpec validation is bypassed for builders, so every
# value a builder places on the command line is sanitized here. Commands are run
# without a shell, but rejecting shell metacharacters and whitespace keeps each
# value a single, well-formed token and guards against option injection.
# ---------------------------------------------------------------------------

# Characters rejected in host, URL, path, and list values assembled by a
# builder. URL query and fragment characters (``? = % + #``) are permitted,
# because they are required to express real targets and the command never
# reaches a shell; shell metacharacters, quotes, and whitespace are not.
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


def _safe_int(name: str, value: Any) -> str:
    try:
        return str(int(value))
    except (TypeError, ValueError):
        raise ToolArgError(f"{name}: expected an integer") from None


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
    tail.append(_safe("url", args.get("url")))
    return tail


def _build_gobuster(args: Mapping[str, Any]) -> list[str]:
    # The directory module is a positional subcommand that precedes its options.
    tail = [
        "dir",
        "-u", _safe("url", args.get("url")),
        "-w", _safe("wordlist", args.get("wordlist")),
    ]
    if args.get("extensions"):
        tail += ["-x", _safe("extensions", args["extensions"])]
    if args.get("threads"):
        tail += ["-t", _safe_int("threads", args["threads"])]
    if args.get("insecure"):
        tail.append("-k")
    return tail


def _build_ffuf(args: Mapping[str, Any]) -> list[str]:
    # The wordlist is substituted wherever the FUZZ keyword appears; append it to
    # the path when the caller did not place it explicitly.
    url = _safe("url", args.get("url"))
    if "FUZZ" not in url:
        url = url.rstrip("/") + "/FUZZ"
    tail = ["-u", url, "-w", _safe("wordlist", args.get("wordlist"))]
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
    tail = ["-u", _safe("url", args.get("url")), "--batch"]
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
                        help="Target URL, including any query string."),
                ArgSpec("method",
                        choices=("GET", "HEAD", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"),
                        help="HTTP method; GET is used when omitted."),
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
            args=(
                ArgSpec("recursive", kind="flag", token="-r",
                        help="Download recursively."),
                ArgSpec("output", token="-O", validate="target",
                        help="Write the download to this file path."),
                ArgSpec("timeout", token="--timeout", validate="int",
                        help="Network timeout in seconds."),
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
            args=(
                ArgSpec("aggression", token="-a", choices=("1", "3", "4"),
                        help="Aggression level: 1 passive, 3 aggressive, 4 heavy."),
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
                ArgSpec("shares", kind="flag", token="--shares",
                        help="Enumerate shares over SMB."),
            ),
        ),
        ToolSpec(
            name="nc",
            binary="nc",
            summary="Open and inspect raw TCP or UDP connections.",
            category="net",
            args=(
                ArgSpec("verbose", kind="flag", token="-v",
                        help="Report connection progress."),
                ArgSpec("zero_io", kind="flag", token="-z",
                        help="Report whether a port is open without sending data."),
                ArgSpec("wait", token="-w", validate="int",
                        help="Connection timeout in seconds."),
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
    ]
    return Registry(specs)
