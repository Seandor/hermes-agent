"""Hash-bound transport fallback for registry artifacts, never a second resolver.

Only uv's failed artifact is fetched. Installing its exact pin from a private
wheelhouse via uv's supported interface lets unchanged sync finish; populating
an alternate index's cache alone cannot satisfy the original locked URL.
"""
from __future__ import annotations

from collections.abc import Mapping
import configparser
from dataclasses import dataclass, replace
from functools import partial
import hashlib
from http.client import HTTPConnection, HTTPException, HTTPResponse, HTTPSConnection
import ipaddress
from pathlib import Path
import re
import ssl
import tempfile
import time
from typing import TYPE_CHECKING
from urllib.error import HTTPError, URLError
from urllib.parse import unquote, urlsplit
from urllib.request import HTTPHandler, HTTPRedirectHandler, build_opener

from pm.environment import classify_uv_failure  # noqa: TID251 -- private engine collaborator
from pm.index_config import pip_config_candidates
from pm.package import InstallError

if TYPE_CHECKING:
    import subprocess

    from pm.environment import PythonEnvironment  # noqa: TID251 -- private engine collaborator

CONFIG_KEY = "[hermes] python-artifact-mirror"
# A universal lock includes many platform wheels per package. Cover the supported
# full graph without letting an unusually large lock remove the absolute bound.
MAX_ARTIFACTS = 4096
_MAX_SIZE = 2 * 1024**3


def _error(package: str, reason: str) -> InstallError:
    return InstallError(package, f"Python artifact mirror: {reason}",
                        f"check {CONFIG_KEY} in pip.conf/pip.ini; the original uv.lock is unchanged")


def _url(value: str) -> bool:
    try:
        parsed = urlsplit(value)
        loopback = parsed.hostname == "localhost"
        if parsed.hostname and not loopback:
            try:
                loopback = ipaddress.ip_address(parsed.hostname).is_loopback
            except ValueError:
                pass
        return bool(parsed.hostname and parsed.port != 0 and not parsed.username and not parsed.password
                    and not parsed.query and not parsed.fragment and "%" not in parsed.netloc
                    and not any(part in (".", "..") for part in unquote(parsed.path).split("/"))
                    and not re.search(r"[\s\\\x00-\x1f\x7f]", value)
                    and (parsed.scheme == "https" or (parsed.scheme == "http" and loopback)))
    except ValueError:
        return False


def configured_mirror(env: Mapping[str, str]) -> str | None:
    """Explicit transport consent, separate from pip/uv's resolution indexes."""
    parser = configparser.RawConfigParser()
    try:
        parser.read(str(path) for path in pip_config_candidates(env))
        value = parser.get("hermes", "python-artifact-mirror", fallback="").strip().rstrip("/")
    except (OSError, configparser.Error):
        if parser.has_option("hermes", "python-artifact-mirror"):
            raise _error("venv", f"cannot read {CONFIG_KEY}") from None
        return None
    if not value:
        return None
    if not _url(value):
        raise _error("venv", f"{CONFIG_KEY} requires HTTPS without credentials, query or fragment (HTTP only on loopback)")
    return value


@dataclass(frozen=True)
class LockedArtifact:
    package: str
    version: str
    url: str
    sha256: str
    size: int
    filename: str


def failed_artifact(lock: bytes, diagnostic: str) -> LockedArtifact | None:
    """Let uv select markers, extras, forks and platform tags; match its exact failed URL."""
    # Older bootstrap interpreters can still use the default path without tomllib.
    import tomllib

    data = tomllib.loads(lock.decode("utf-8"))
    for package in data.get("package", []):
        if package.get("source", {}).get("registry", "").rstrip("/") != "https://pypi.org/simple":
            continue
        for entry in [*package.get("wheels", []), package.get("sdist", {})]:
            url = entry.get("url", "")
            if not url or url not in diagnostic:
                continue
            name, version = package["name"], package["version"]
            parsed = urlsplit(url)
            if not _url(url) or not parsed.path.startswith("/packages/"):
                continue
            # Only PyPI's artifact CDN is substituted; private registry and URL pins are not.
            if parsed.hostname != "files.pythonhosted.org" and not (
                parsed.scheme == "http" and _url(url)
            ):
                continue
            filename = unquote(parsed.path.rsplit("/", 1)[-1])
            digest, size = entry.get("hash", ""), entry.get("size")
            if (not re.fullmatch(r"[a-z0-9]+(?:[-_.][a-z0-9]+)*", name)
                    or not re.fullmatch(r"[A-Za-z0-9_.+!]+", version)
                    or not re.fullmatch(r"[A-Za-z0-9_.+-]+\.(?:whl|tar\.gz|zip)", filename)
                    or any(part in ("", ".", "..") for part in unquote(parsed.path).split("/")[1:])
                    or not re.fullmatch(r"sha256:[a-f0-9]{64}", digest)
                    or type(size) is not int or not 0 < size <= _MAX_SIZE):
                raise _error(name, "failed artifact lacks a valid exact SHA-256, size or filename")
            return LockedArtifact(name, version, url, digest[7:], size, filename)
    return None


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class _DeadlineResponse(HTTPResponse):
    def __init__(self, sock, *args, deadline: float, **kwargs):
        self._socket = sock
        self._deadline = deadline
        super().__init__(sock, *args, **kwargs)

    def read1(self, n=-1):
        if self.isclosed():
            return b""
        remaining = self._deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("artifact deadline exceeded")
        self._socket.settimeout(min(30, remaining))
        return super().read1(n)


def _connection(kind, host, *, deadline: float, **kwargs):
    connection = kind(host, **kwargs)
    connection.response_class = partial(_DeadlineResponse, deadline=deadline)
    return connection


class _DeadlineHandler(HTTPHandler):
    # Run before urllib's default HTTPS handler, retaining its proxy/tunnel support.
    handler_order = 400

    def __init__(self, deadline: float):
        super().__init__()
        self.deadline = deadline

    def http_open(self, req):
        return self.do_open(partial(_connection, HTTPConnection, deadline=self.deadline), req)

    def https_open(self, request):
        return self.do_open(partial(_connection, HTTPSConnection, deadline=self.deadline), request)


def fetch_artifact(artifact: LockedArtifact, mirror: str, directory: Path, *, deadline: float) -> Path:
    """Never pass an unverified byte or a remote URL to uv; never include URLs in errors."""
    url = mirror + urlsplit(artifact.url).path
    destination = directory / artifact.filename
    opener = build_opener(_NoRedirect(), _DeadlineHandler(deadline))
    for attempt in range(2):
        try:
            digest = hashlib.sha256()
            size = 0
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise _error(artifact.package, "download deadline exceeded")
            with opener.open(url, timeout=min(30, remaining)) as response, destination.open("wb") as stream:
                while block := response.read1(min(1024 * 1024, artifact.size - size + 1)):
                    size += len(block)
                    if size > artifact.size or time.monotonic() >= deadline:
                        raise _error(artifact.package, "artifact exceeds locked size or download deadline")
                    digest.update(block)
                    stream.write(block)
            if size != artifact.size or digest.hexdigest() != artifact.sha256:
                raise _error(artifact.package, "artifact SHA-256 or size differs from uv.lock")
            return destination
        except HTTPError as exc:
            retry = exc.code in (408, 429, 500, 502, 503, 504)
        except URLError as exc:
            retry = not isinstance(exc.reason, ssl.SSLError)
        except (TimeoutError, ConnectionError, HTTPException):
            retry = True
        except OSError:
            raise _error(artifact.package, "cannot read or write artifact") from None
        if not retry or attempt == 1:
            raise _error(artifact.package, "artifact unavailable (at most two download attempts)") from None
    raise AssertionError("unreachable")


def _failure(package: str, stage: str, result) -> InstallError:
    """Keep build/conflict verdicts used by carried extras, without retaining package URLs."""
    failure = classify_uv_failure(stage, result.returncode, (result.stderr or "") + (result.stdout or ""))
    cause = f"uv {stage} exited {result.returncode}; diagnostic withheld to protect package URLs"
    remedy = re.sub(r"https?://[^'\"`<>]+", "<package URL>", failure.remedy)
    return type(failure)(package, cause, f"{remedy}; check {CONFIG_KEY} in pip.conf/pip.ini")


def _artifact_budget(lock: bytes) -> int:
    """Count distinct registry URLs, not package rows or top-level invocations.

    This only bounds work; failed_artifact still validates every selected entry.
    uv remains responsible for selecting the platform/extra-specific graph.
    """
    import tomllib

    data = tomllib.loads(lock.decode("utf-8"))
    urls = {entry["url"] for package in data.get("package", [])
            if package.get("source", {}).get("registry", "").rstrip("/") == "https://pypi.org/simple"
            for entry in [*package.get("wheels", []), package.get("sdist", {})] if entry.get("url")}
    return min(len(urls), MAX_ARTIFACTS)


def sync_with_mirror(environment: PythonEnvironment, command: list[str], *, source: Path,
                     timeout: int, mirror: str) -> subprocess.CompletedProcess:
    """Retry only new, exact failed artifacts, ending with the original sync command."""
    deadline = time.monotonic() + timeout
    lock_path = source / "uv.lock"
    lock = lock_path.read_bytes()
    budget = _artifact_budget(lock)
    # Capture instead of streaming: neither hostile HTTP errors nor URL credentials may reach logs.
    quiet = replace(environment, output=None, deadline=deadline)
    seen: set[str] = set()
    with tempfile.TemporaryDirectory(prefix="pm-python-artifacts-") as temp:
        root = Path(temp)
        while True:
            if lock_path.read_bytes() != lock:
                raise _error("venv", "uv.lock changed during fallback")
            remaining = int(deadline - time.monotonic())
            if remaining <= 0:
                raise _error("venv", "sync deadline exceeded")
            try:
                result = quiet._with_cache_recovery(
                    "sync", lambda: quiet._sync_attempt(command, cwd=source, timeout=remaining))
            except InstallError:
                if time.monotonic() >= deadline:
                    raise _error("venv", "sync deadline exceeded") from None
                raise _error("venv", "uv sync or cache recovery failed; run hermes pm doctor") from None
            if time.monotonic() >= deadline:
                raise _error("venv", "sync deadline exceeded")
            if result.returncode == 0 and lock_path.read_bytes() == lock:
                return result
            artifact = failed_artifact(lock, (result.stderr or "") + (result.stdout or ""))
            if artifact is None:
                raise _failure("venv", "sync", result) from None
            if artifact.url in seen or len(seen) >= budget:
                raise _error(artifact.package, f"fallback exhausted (at most {budget} distinct artifacts)")
            seen.add(artifact.url)
            directory = root / str(len(seen))
            directory.mkdir()
            fetch_artifact(artifact, mirror, directory, deadline=deadline)
            if lock_path.read_bytes() != lock:
                raise _error(artifact.package, "uv.lock changed during fallback")
            requirements = directory / "requirements.txt"
            requirements.write_text(f"{artifact.package}=={artifact.version} --hash=sha256:{artifact.sha256}\n", encoding="utf-8")
            installed = replace(quiet, no_config=True, offline=True)._run([
                "pip", "install", "--python", str(environment.executable), "--no-deps",
                "--no-index", "--find-links", str(directory), "--require-hashes",
                "-r", str(requirements),
            ], cwd=source, timeout=max(1, int(deadline - time.monotonic())))
            if installed.returncode:
                raise _failure(artifact.package, "pip", installed) from None
