"""Releases: tag ``main``, follow the Release workflow, and check the result.

``prokopton-ml release vVERSION`` is the same for the platform and for each
product; options say what to check:

- ``--version-file pyproject.toml``: the tag must match the package version;
- ``--pypi PACKAGE``: when the workflow waits for approval to publish to
  PyPI, offer to approve it here, then check that PyPI lists the version;
- ``--check-url URL``: check that the live page shows the new version, in
  ``<span data-version>VERSION/COMMIT</span>`` (COMMIT is the first 7
  characters of the commit's hash).

Before tagging, it checks that you are on ``main`` with no uncommitted
changes, that ``main`` matches GitHub, that the version is new, and that
this commit has not been released already, so a release always contains
something new. It stops at the first problem: nothing after it runs.

With ``--dry-run`` it makes those checks and lists the changes, then stops
before tagging. It uses ``git`` and the GitHub command-line tool ``gh``
(signed in).
"""

import re
import subprocess
import time
import tomllib
import urllib.error
import urllib.request
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from prokopton.ml import __version__

__all__ = [
    "Release",
    "ReleaseError",
    "Tools",
    "pypi_lists",
    "release",
]

TAG = re.compile(r"v\d+\.\d+\.\d+((a|b|rc)\d+)?")
"""A version tag, such as v1.2.3 or v0.2.0a4."""

APPROVABLE = ".[] | select(.current_user_can_approve) | .environment.id"
"""The waiting environments you can approve, from GitHub's pending deployments."""

USER_AGENT = f"prokopton-ml/{__version__} (release check)"
"""Sent with each web request: some sites (Cloudflare's browser check) refuse
Python's default user agent."""

RUN_STATE = (
    '(.status + " " + .conclusion), '
    '(.jobs[] | select(.status == "completed") | .conclusion + " " + .name)'
)
"""The run's state, then each finished job's conclusion and name, one a line."""

FOLLOW_CHECKS = 360
"""How many times to check the run, 10 seconds apart (an hour)."""

LIVE_VERSION = re.compile(r"<span data-version>([^<]*)</span>")
"""Where a live page shows its version."""


class ReleaseError(RuntimeError):
    """Raised when a release cannot go ahead or did not succeed."""


@dataclass(frozen=True)
class Release:
    """What to release, and what to check."""

    tag: str
    name: str
    """For the tag's message, such as ``prokopton-ml`` or ``DNAbarcode.org``."""
    version_file: Path | None = None
    pypi: str | None = None
    check_url: str | None = None
    workflow: str = "release.yml"
    dry_run: bool = False

    @property
    def version(self) -> str:
        return self.tag.removeprefix("v")


def _run(command: Sequence[str]) -> str:
    """Run a command and return its output.

    Raises:
        ReleaseError: If the command fails.
    """
    done = subprocess.run(command, capture_output=True, text=True, check=False)
    if done.returncode != 0:
        detail = done.stderr.strip() or done.stdout.strip()
        msg = f"{' '.join(command)} failed: {detail}"
        raise ReleaseError(msg)
    return done.stdout.strip()


def _fetch(url: str) -> tuple[int, str]:
    """GET a URL: its HTTP status and text (status 0 if it can't be reached)."""
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.status, response.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as error:
        return error.code, ""
    except OSError:
        return 0, ""


def _ask(question: str) -> bool:
    """Ask a yes-or-no question (no means no, as does no answer)."""
    try:
        return input(f"{question} [y/N] ").strip().lower() in ("y", "yes")
    except EOFError:
        return False


@dataclass(frozen=True)
class Tools:
    """How the release reaches the outside world (replaced in tests)."""

    run: Callable[[Sequence[str]], str] = _run
    fetch: Callable[[str], tuple[int, str]] = _fetch
    ask: Callable[[str], bool] = _ask
    say: Callable[[str], None] = print
    sleep: Callable[[float], None] = time.sleep


def _package_version(path: Path) -> str:
    try:
        data = tomllib.loads(path.read_text("utf-8"))
        return str(data["project"]["version"])
    except (OSError, tomllib.TOMLDecodeError, KeyError) as error:
        msg = f"Could not read the version from {path}: {error}"
        raise ReleaseError(msg) from error


def _preflight(spec: Release, tools: Tools) -> None:
    run = tools.run
    if not TAG.fullmatch(spec.tag):
        msg = f"Not a version like v1.2.3 or v1.2.3a1: {spec.tag}"
        raise ReleaseError(msg)
    if spec.version_file is not None:
        found = _package_version(spec.version_file)
        if found != spec.version:
            msg = f"{spec.version_file} says version {found}, not {spec.version}"
            raise ReleaseError(msg)
    if run(["git", "branch", "--show-current"]) != "main":
        msg = "Not on main (git switch main)"
        raise ReleaseError(msg)
    if run(["git", "status", "--porcelain"]):
        msg = "There are uncommitted changes (see git status)"
        raise ReleaseError(msg)
    run(["git", "fetch", "--quiet", "--tags", "origin"])
    if run(["git", "rev-parse", "HEAD"]) != run(["git", "rev-parse", "origin/main"]):
        msg = "main differs from GitHub (git pull)"
        raise ReleaseError(msg)
    if run(["git", "tag", "--list", spec.tag]):
        msg = f"{spec.tag} already exists"
        raise ReleaseError(msg)
    released = run(["git", "tag", "--points-at", "HEAD", "--list", "v*"])
    if released:
        msg = f"This commit is already released as {' '.join(released.split())}"
        raise ReleaseError(msg)


def _changes(tools: Tools) -> str:
    try:
        last = tools.run(["git", "describe", "--tags", "--abbrev=0", "--match", "v*"])
    except ReleaseError:
        last = ""
    log = tools.run(["git", "log", "--oneline", f"{last}..HEAD" if last else "HEAD"])
    return f"Changes since {last or 'the first commit'}:\n{log}"


def _find_run(spec: Release, tools: Tools) -> str:
    for _ in range(24):
        found = tools.run(
            [
                "gh",
                "run",
                "list",
                "--workflow",
                spec.workflow,
                "--branch",
                spec.tag,
                "--json",
                "databaseId",
                "--jq",
                ".[0].databaseId // empty",
            ]
        )
        if found:
            return found
        tools.sleep(5)
    msg = (
        f"No Release run started for {spec.tag} "
        f"(gh run list --workflow={spec.workflow})"
    )
    raise ReleaseError(msg)


def _follow(spec: Release, run_id: str, tools: Tools) -> None:
    """Wait for the run to finish, offering to approve PyPI when it asks."""
    repo = tools.run(
        ["gh", "repo", "view", "--json", "nameWithOwner", "--jq", ".nameWithOwner"]
    )
    pending = f"repos/{repo}/actions/runs/{run_id}/pending_deployments"
    url = tools.run(["gh", "run", "view", run_id, "--json", "url", "--jq", ".url"])
    tools.say(f"Release run {run_id}: {url}")
    offered = spec.pypi is None
    finished: set[str] = set()
    for _ in range(FOLLOW_CHECKS):
        view = ["gh", "run", "view", run_id, "--json", "status,conclusion,jobs"]
        lines = tools.run([*view, "--jq", RUN_STATE]).splitlines()
        status, _, conclusion = (lines[0] if lines else "").partition(" ")
        for line in lines[1:]:
            outcome, _, job = line.partition(" ")
            if job not in finished:
                finished.add(job)
                mark = {"success": "✓", "skipped": "-"}.get(outcome, "✗")
                tools.say(
                    f"  {mark} {job}"
                    + ("" if outcome == "success" else f" ({outcome})")
                )
        if status == "completed":
            if conclusion != "success":
                outcome = conclusion or "unknown"
                msg = f"The release run ended: {outcome} (gh run view {run_id})"
                raise ReleaseError(msg)
            return
        if not offered and status == "waiting":
            waiting = tools.run(["gh", "api", pending, "--jq", APPROVABLE])
            if waiting:
                offered = True
                _approve(spec, pending, waiting.split(), url, tools)
        tools.sleep(10)
    msg = f"The release is still running after an hour; see {url}"
    raise ReleaseError(msg)


def _approve(
    spec: Release, pending: str, environments: list[str], url: str, tools: Tools
) -> None:
    question = f"The run is waiting. Publish {spec.pypi} {spec.version} to PyPI?"
    if not tools.ask(question):
        msg = f"Not approved; the run is waiting. Approve or reject it at {url}"
        raise ReleaseError(msg)
    command = ["gh", "api", "--method", "POST", pending]
    for environment in environments:
        command += ["-F", f"environment_ids[]={environment}"]
    command += ["-f", "state=approved", "-f", f"comment=Release {spec.tag}"]
    tools.run(command)
    tools.say("Approved; publishing to PyPI")


def pypi_lists(package: str, version: str, tools: Tools) -> bool:
    """Whether PyPI lists a package version (it can take a few seconds)."""
    for attempt in range(6):
        status, _ = tools.fetch(f"https://pypi.org/pypi/{package}/{version}/json")
        if status == 200:
            return True
        if attempt < 5:
            tools.sleep(10)
    return False


def _live_version(url: str, tools: Tools) -> tuple[int, str]:
    code, page = tools.fetch(url)
    found = LIVE_VERSION.search(page)
    return code, found.group(1) if found else ""


def release(spec: Release, tools: Tools | None = None) -> str:
    """Make a release (see the module docstring) and say what was released.

    Raises:
        ReleaseError: At the first problem.
    """
    tools = tools or Tools()
    _preflight(spec, tools)
    tools.say(_changes(tools))
    if spec.dry_run:
        return f"Ready to release {spec.name} {spec.version} (dry run: not tagged)"
    tools.run(["git", "tag", "-a", spec.tag, "-m", f"{spec.name} {spec.version}"])
    tools.run(["git", "push", "--quiet", "origin", spec.tag])
    _follow(spec, _find_run(spec, tools), tools)
    done = [f"Released {spec.name} {spec.version}"]
    if spec.pypi is not None:
        if not pypi_lists(spec.pypi, spec.version, tools):
            msg = f"PyPI does not list {spec.pypi} {spec.version} yet"
            raise ReleaseError(msg)
        done.append(f"https://pypi.org/project/{spec.pypi}/{spec.version}/")
    if spec.check_url is not None:
        commit = tools.run(["git", "rev-parse", "--short=7", "HEAD"])
        expected = f"{spec.version}/{commit}"
        code, live = 0, ""
        for attempt in range(6):
            code, live = _live_version(spec.check_url, tools)
            if live == expected:
                break
            if attempt < 5:
                tools.sleep(10)
        if live != expected:
            shown = f"shows {live}" if live else f"gave HTTP {code} with no version"
            msg = f"{spec.check_url} {shown}, not {expected}"
            raise ReleaseError(msg)
        done.append(f"{spec.check_url} shows {live}")
    return "; ".join(done)
