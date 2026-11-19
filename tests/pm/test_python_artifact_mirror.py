"""Exact locked artifacts may use an explicitly opted-in transport mirror."""
from __future__ import annotations

from dataclasses import replace
from functools import partial
import hashlib
from http.server import SimpleHTTPRequestHandler
import os
from pathlib import Path
import shutil
import sys
import time

import pytest

from pm.environment import PythonEnvironment
from pm.package import InstallError
from tests.pm._fixtures import _wheel, threaded_server


@pytest.fixture
def mirror_project(tmp_path):
    web = tmp_path / "web"
    packages = web / "packages" / "aa"
    packages.mkdir(parents=True)
    wheel = _wheel(packages, "mirror_dep")
    payload = wheel.read_bytes()
    requests = []

    class Handler(SimpleHTTPRequestHandler):
        def do_GET(self):
            requests.append(self.path)
            if self.server is origin:
                self.send_error(404)
            else:
                super().do_GET()

        def log_message(self, *_args):
            pass

    with threaded_server(partial(Handler, directory=str(web))) as origin, threaded_server(
        partial(Handler, directory=str(web))
    ) as server:
        base = f"http://127.0.0.1:{server.server_port}"
        original = f"http://127.0.0.1:{origin.server_port}"
        source = tmp_path / "source with spaces"
        source.mkdir()
        (source / "pyproject.toml").write_text(
            '[project]\nname="mirror-root"\nversion="1"\nrequires-python=">=3.11"\n'
            'dependencies=["mirror-dep==1.0"]\n[tool.uv]\npackage=false\n', encoding="utf-8")
        digest = hashlib.sha256(payload).hexdigest()
        (source / "uv.lock").write_text(
            'version=1\nrevision=3\nrequires-python=">=3.11"\n'
            '[[package]]\nname="mirror-dep"\nversion="1.0"\n'
            'source={registry="https://pypi.org/simple"}\n'
            f'wheels=[{{url="{original}/packages/aa/{wheel.name}",hash="sha256:{digest}",size={len(payload)}}}]\n'
            '[[package]]\nname="mirror-root"\nversion="1"\nsource={virtual="."}\n'
            'dependencies=[{name="mirror-dep"}]\n[package.metadata]\n'
            'requires-dist=[{name="mirror-dep",specifier="==1.0"}]\n', encoding="utf-8")
        config = tmp_path / "pip.ini"
        config.write_text(f"[hermes]\npython-artifact-mirror = {base}\n", encoding="utf-8")
        env = {key: value for key, value in os.environ.items()
               if not key.startswith(("UV_", "PIP_", "PYTHON", "HERMES_")) and key != "VIRTUAL_ENV"}
        env.update(HOME=str(tmp_path), USERPROFILE=str(tmp_path), PIP_CONFIG_FILE=str(config))
        uv = shutil.which("uv")
        assert uv, "artifact-mirror integration requires real uv"
        environment = PythonEnvironment(uv=Path(uv), python=Path(sys.executable),
                                        destination=tmp_path / "venv", cache=tmp_path / "cache", env=env)
        environment.create()
        yield source, environment, wheel, requests, config


@pytest.mark.parametrize("isolated", [False, True])
def test_unchanged_lock_succeeds_after_exact_artifact_fallback(mirror_project, isolated):
    source, environment, wheel, requests, _config = mirror_project
    before = {name: (source / name).read_bytes() for name in ("pyproject.toml", "uv.lock")}
    environment = replace(environment, no_config=isolated)
    environment.sync(source, locked=isolated)
    assert f"/packages/aa/{wheel.name}" in requests
    assert all((source / name).read_bytes() == value for name, value in before.items())
    from pm.environments import site_packages

    assert (site_packages(environment.destination) / "mirror_dep" / "__init__.py").is_file()
    # The final sync audits the result without an origin or mirror download.
    requests.clear()
    replace(environment, offline=True).sync(source, locked=isolated)
    assert requests == []


def test_default_sync_never_uses_artifact_mirror(mirror_project):
    source, environment, _wheel_path, requests, config = mirror_project
    config.write_text("[global]\nindex-url=https://pypi.org/simple\n", encoding="utf-8")
    with pytest.raises(InstallError):
        environment.sync(source)
    assert len(requests) == 1


@pytest.mark.parametrize("damage", ["hash", "short", "long", "missing"])
def test_wrong_or_missing_artifact_never_installs(mirror_project, damage):
    source, environment, wheel, _requests, _config = mirror_project
    before = (source / "uv.lock").read_bytes()
    payload = wheel.read_bytes()
    if damage == "missing":
        wheel.unlink()
    else:
        wheel.write_bytes({"hash": bytes([payload[0] ^ 1]) + payload[1:],
                           "short": payload[:-1], "long": payload + b"x"}[damage])
    with pytest.raises(InstallError, match="mirror-dep.*Python artifact mirror"):
        environment.sync(source)
    from pm.environments import site_packages

    assert not (site_packages(environment.destination) / "mirror_dep").exists()
    assert (source / "uv.lock").read_bytes() == before


@pytest.mark.parametrize("url", [
    "http://mirror.example", "https://reader:password@mirror.example", "https://mirror.example/?token=private",
    "https://mirror.example/#private", "https://mirror.example:bad", "file:///wheelhouse",
    "https://mirror.example/\nprivate", "https://mirror.example/../private",
])
def test_invalid_config_is_private_and_rejected_before_uv(mirror_project, monkeypatch, url):
    source, environment, _wheel_path, requests, config = mirror_project
    config.write_text(f"[hermes]\npython-artifact-mirror={url}\n", encoding="utf-8")
    monkeypatch.setattr(PythonEnvironment, "_run", lambda *a, **k: pytest.fail("must reject before uv"))
    with pytest.raises(InstallError) as error:
        environment.sync(source)
    assert url not in str(error.value)
    assert "python-artifact-mirror" in str(error.value)
    assert requests == []


def test_index_configuration_is_not_artifact_consent(tmp_path, monkeypatch):
    from pm.python_artifact_mirror import configured_mirror

    config = tmp_path / "pip.ini"
    config.write_text("[global]\nindex-url=https://mirror.example/simple\n", encoding="utf-8")
    monkeypatch.setenv("UV_INDEX_URL", "https://another.example/simple")
    assert configured_mirror({"PIP_CONFIG_FILE": str(config)}) is None
    config.write_text("[hermes]\npython-artifact-mirror=https://mirror.example/pypi/\n", encoding="utf-8")
    assert configured_mirror({"PIP_CONFIG_FILE": str(config)}) == "https://mirror.example/pypi"
    assert configured_mirror({"PIP_CONFIG_FILE": os.devnull}) is None


@pytest.mark.parametrize("source", ['{url="https://files.pythonhosted.org/packages/x.whl"}',
                                    '{git="https://example.test/repo",rev="abc"}', '{directory="."}'])
def test_non_registry_dependencies_are_not_substituted(source):
    from pm.python_artifact_mirror import failed_artifact

    url = "https://files.pythonhosted.org/packages/aa/demo-1-py3-none-any.whl"
    lock = (f'[[package]]\nname="demo"\nversion="1"\nsource={source}\n'
            f'wheels=[{{url="{url}",hash="sha256:{"a" * 64}",size=5}}]\n').encode()
    assert failed_artifact(lock, url) is None


@pytest.mark.parametrize("url", ["https://reader:password@files.pythonhosted.org/packages/a.whl",
                                "https://files.pythonhosted.org/packages/a.whl?private=1",
                                "https://other.example/packages/a.whl"])
def test_credential_or_non_pypi_artifact_is_not_substituted(url):
    from pm.python_artifact_mirror import failed_artifact

    lock = (f'[[package]]\nname="demo"\nversion="1"\nsource={{registry="https://pypi.org/simple"}}\n'
            f'wheels=[{{url="{url}",hash="sha256:{"a" * 64}",size=5}}]\n').encode()
    assert failed_artifact(lock, url) is None


@pytest.mark.parametrize("status, attempts, success", [(503, 2, True), (404, 1, False), (302, 1, False), (500, 2, False)])
def test_download_retries_and_redirects_are_bounded(tmp_path, status, attempts, success):
    from http.server import BaseHTTPRequestHandler
    from pm.python_artifact_mirror import LockedArtifact, fetch_artifact

    requests = []
    payload = b"locked"

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            requests.append(self.path)
            code = 200 if success and len(requests) == 2 else status
            self.send_response(code)
            self.send_header("Location", "https://reader:password@unreachable.example/private")
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *_args):
            pass

    with threaded_server(Handler) as server:
        mirror = f"http://127.0.0.1:{server.server_port}"
        artifact = LockedArtifact("demo", "1", "https://files.pythonhosted.org/packages/demo.whl",
                                  hashlib.sha256(payload).hexdigest(), len(payload), "demo.whl")
        if success:
            assert fetch_artifact(artifact, mirror, tmp_path, deadline=time.monotonic() + 10).read_bytes() == payload
        else:
            with pytest.raises(InstallError) as error:
                fetch_artifact(artifact, mirror, tmp_path, deadline=time.monotonic() + 10)
            assert "password" not in str(error.value)
            assert "private" not in str(error.value)
        assert len(requests) == attempts


def test_private_registry_artifact_is_not_rewritten():
    from pm.python_artifact_mirror import failed_artifact

    url = "https://files.pythonhosted.org/packages/aa/demo-1-py3-none-any.whl"
    lock = (f'[[package]]\nname="demo"\nversion="1"\nsource={{registry="https://private.example/simple"}}\n'
            f'wheels=[{{url="{url}",hash="sha256:{"a" * 64}",size=5}}]\n').encode()
    assert failed_artifact(lock, url) is None


@pytest.mark.parametrize("pin", [f'hash="sha256:{"a" * 63}",size=5', f'hash="sha256:{"a" * 64}"',
                                f'hash="sha256:{"a" * 64}",size=-1', f'hash="sha256:{"a" * 64}",size=true'])
def test_missing_or_invalid_integrity_cannot_fallback(pin):
    from pm.python_artifact_mirror import failed_artifact

    url = "https://files.pythonhosted.org/packages/aa/demo-1-py3-none-any.whl"
    lock = (f'[[package]]\nname="demo"\nversion="1"\nsource={{registry="https://pypi.org/simple"}}\n'
            f'wheels=[{{url="{url}",{pin}' + '}]\n').encode()
    with pytest.raises(InstallError, match="demo.*SHA-256"):
        failed_artifact(lock, url)


def test_repeat_failure_and_artifact_budget_stop_without_looping(mirror_project, monkeypatch):
    import subprocess
    import pm.python_artifact_mirror as fallback

    source, environment, _wheel_path, _requests, _config = mirror_project
    real_run = PythonEnvironment._run
    calls = []

    def run(self, args, **kwargs):
        calls.append(args[0])
        if args[0] == "pip":
            return subprocess.CompletedProcess(args, 0, "", "")
        return real_run(self, args, **kwargs)

    monkeypatch.setattr(PythonEnvironment, "_run", run)
    with pytest.raises(InstallError, match="mirror-dep.*fallback exhausted"):
        environment.sync(source)
    assert calls == ["sync", "pip", "sync"]
    calls.clear()
    monkeypatch.setattr(fallback, "MAX_ARTIFACTS", 0)
    with pytest.raises(InstallError, match="fallback exhausted"):
        environment.sync(source)
    assert calls == ["sync"]


def test_uv_ignores_unreachable_unselected_lock_package(mirror_project):
    source, environment, wheel, requests, _config = mirror_project
    lock = source / "uv.lock"
    text = lock.read_text(encoding="utf-8")
    # A second pin is deliberately unavailable, but not part of this platform's selected graph.
    text += '\n[[package]]\nname="not-selected"\nversion="9"\nsource={registry="https://pypi.org/simple"}\n'
    text += f'wheels=[{{url="https://files.pythonhosted.org/packages/absent/not_selected-9-py3-none-any.whl",hash="sha256:{"f" * 64}",size=10}}]\n'
    lock.write_text(text, encoding="utf-8")
    before = lock.read_bytes()
    environment.sync(source)
    assert requests == [f"/packages/aa/{wheel.name}", f"/packages/aa/{wheel.name}"]
    assert lock.read_bytes() == before


def test_source_archive_uses_exact_hash_without_resolving_runtime_dependencies(mirror_project):
    import io
    import tarfile

    source, environment, wheel, requests, _config = mirror_project
    archive_path = wheel.with_name("mirror_dep-1.0.tar.gz")
    contents = {
        "pyproject.toml": (b'[project]\nname="mirror-dep"\nversion="1.0"\n'
                           b'[build-system]\nrequires=[]\nbuild-backend="local_backend"\nbackend-path=["."]\n'),
        "local_backend.py": (f'from pathlib import Path\nimport shutil\n'
                             f'def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):\n'
                             f'    shutil.copyfile("payload.whl", Path(wheel_directory) / {wheel.name!r})\n'
                             f'    return {wheel.name!r}\n').encode(),
        "payload.whl": wheel.read_bytes(),
    }
    with tarfile.open(archive_path, "w:gz") as archive:
        for name, payload in contents.items():
            member = tarfile.TarInfo("mirror_dep-1.0/" + name)
            member.size = len(payload)
            archive.addfile(member, io.BytesIO(payload))
    lock = source / "uv.lock"
    text = lock.read_text(encoding="utf-8")
    wheel_line = next(line for line in text.splitlines() if line.startswith("wheels="))
    original = wheel_line.split('url="')[1].split('"')[0]
    sdist = (f'sdist={{url="{original.rsplit("/", 1)[0]}/{archive_path.name}",'
             f'hash="sha256:{hashlib.sha256(archive_path.read_bytes()).hexdigest()}",size={archive_path.stat().st_size}}}')
    lock.write_text(text.replace(wheel_line, sdist), encoding="utf-8")
    before = lock.read_bytes()
    environment.sync(source)
    assert f"/packages/aa/{archive_path.name}" in requests
    assert lock.read_bytes() == before


@pytest.mark.parametrize("isolated", [False, True])
def test_unrelated_malformed_pip_config_does_not_opt_in(mirror_project, monkeypatch, isolated):
    import subprocess

    source, environment, _wheel_path, _requests, config = mirror_project
    config.write_text("unrelated invalid pip setting\n", encoding="utf-8")
    called = []

    def run(self, args, **kwargs):
        called.append(args)
        return subprocess.CompletedProcess(args, 0, "", "")

    monkeypatch.setattr(PythonEnvironment, "_run", run)
    replace(environment, no_config=isolated).sync(source)
    assert len(called) == 1


@pytest.mark.parametrize("stage", ["sync", "pip"])
def test_build_verdict_is_preserved_for_carried_extra_recovery(mirror_project, monkeypatch, stage):
    import subprocess
    from pm.carried_extras import BUILD_VERDICTS
    from pm.environment import BuildFailure

    source, environment, _wheel_path, _requests, _config = mirror_project
    real_run = PythonEnvironment._run

    def run(self, args, **kwargs):
        if args[0] == stage:
            return subprocess.CompletedProcess(args, 1, "", "The build backend returned an error\nNo such file or directory: 'https://reader:password@example.test/private/clang'")
        return real_run(self, args, **kwargs)

    monkeypatch.setattr(PythonEnvironment, "_run", run)
    with pytest.raises(BuildFailure) as error:
        environment.sync(source)
    assert isinstance(error.value, BUILD_VERDICTS)
    assert "password" not in str(error.value)
    assert "private" not in str(error.value)


def test_slow_progressing_body_still_obeys_deadline(tmp_path):
    from http.server import BaseHTTPRequestHandler
    import threading
    from pm.python_artifact_mirror import LockedArtifact, fetch_artifact

    stop = threading.Event()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Length", "100")
            self.end_headers()
            for _ in range(100):
                try:
                    self.wfile.write(b"x")
                    self.wfile.flush()
                except OSError:
                    break
                if stop.wait(0.03):
                    break

        def log_message(self, *_args):
            pass

    with threaded_server(Handler) as server:
        artifact = LockedArtifact("demo", "1", "https://files.pythonhosted.org/packages/demo.whl", "a" * 64, 100, "demo.whl")
        started = time.monotonic()
        try:
            with pytest.raises(InstallError):
                fetch_artifact(artifact, f"http://127.0.0.1:{server.server_port}", tmp_path, deadline=started + 0.2)
            assert time.monotonic() - started < 2
        finally:
            stop.set()


@pytest.mark.parametrize("recovery", ["compile", "cache"])
def test_recovery_cannot_start_after_shared_deadline(mirror_project, monkeypatch, recovery):
    import subprocess

    source, environment, _wheel_path, _requests, _config = mirror_project
    now = [time.monotonic()]
    calls = []

    def run(args, **kwargs):
        calls.append(kwargs["timeout"])
        now[0] += 6
        diagnostic = ("failed to bytecode-compile" if recovery == "compile" else
                      "error: Failed to install: alpha-1.0-py3-none-any.whl (alpha==1.0)\n"
                      "Caused by: The wheel is invalid: Invalid Wheel-Version")
        return subprocess.CompletedProcess(args, 1 if len(calls) == 1 else 0, "", diagnostic)

    monkeypatch.setattr(time, "monotonic", lambda: now[0])
    monkeypatch.setattr(subprocess, "run", run)
    with pytest.raises(InstallError, match="deadline"):
        environment.sync(source, timeout=5)
    assert len(calls) == 1
