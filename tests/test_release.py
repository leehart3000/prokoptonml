"""Tests for releases: the checks before tagging, following the run (with
the PyPI approval), and the checks afterwards. Commands, web requests and
questions are replaced by fakes; nothing is tagged or pushed."""

import builtins
import io
import re
import sys
import urllib.error
import urllib.request
from collections.abc import Sequence
from email.message import Message
from pathlib import Path

import pytest

from prokopton.ml import cli
from prokopton.ml import release as module
from prokopton.ml.release import Release, ReleaseError, Tools, pypi_lists, release

RUN_VIEW = ("gh", "run", "view", "42", "--json", "status,conclusion", "--jq")
PENDING = "repos/leehart3000/prokoptonml/actions/runs/42/pending_deployments"


def tag_list(tag: str) -> tuple[str, ...]:
    return ("git", "tag", "--list", tag)


class Fake:
    """Scripted answers to commands, web requests and questions.

    An answer is text, a list of texts given one per call (the last one
    repeats), or a ``ReleaseError`` to raise.
    """

    def __init__(self, answers: dict[tuple[str, ...], object]) -> None:
        self.answers = dict(answers)
        self.commands: list[tuple[str, ...]] = []
        self.said: list[str] = []
        self.asked: list[str] = []
        self.slept: list[float] = []
        self.pages: dict[str, list[tuple[int, str]]] = {}
        self.yes = True

    def run(self, command: Sequence[str]) -> str:
        key = tuple(command)
        self.commands.append(key)
        for prefix in sorted(self.answers, key=len, reverse=True):
            if key[: len(prefix)] == prefix:
                answer = self.answers[prefix]
                if isinstance(answer, ReleaseError):
                    raise answer
                if isinstance(answer, list):
                    return str(answer.pop(0) if len(answer) > 1 else answer[0])
                return str(answer)
        msg = f"Unexpected command: {key}"
        raise AssertionError(msg)

    def fetch(self, url: str) -> tuple[int, str]:
        pages = self.pages[url]
        return pages.pop(0) if len(pages) > 1 else pages[0]

    def ask(self, question: str) -> bool:
        self.asked.append(question)
        return self.yes

    def tools(self) -> Tools:
        return Tools(
            run=self.run,
            fetch=self.fetch,
            ask=self.ask,
            say=self.said.append,
            sleep=self.slept.append,
        )


def ready(tag: str = "v0.2.0a4") -> Fake:
    """A repository ready to release, whose run passes straight away."""
    answers: dict[tuple[str, ...], object] = {
        ("git", "branch", "--show-current"): "main",
        ("git", "status", "--porcelain"): "",
        ("git", "fetch"): "",
        ("git", "rev-parse", "HEAD"): "abc123",
        ("git", "rev-parse", "origin/main"): "abc123",
        ("git", "rev-parse", "--short=7", "HEAD"): "abc1234",
        tag_list(tag): "",
        ("git", "tag", "--points-at", "HEAD", "--list", "v*"): "",
        ("git", "describe"): "v0.2.0a3",
        ("git", "log"): "e1dfb58 feat: a change",
        ("git", "tag", "-a"): "",
        ("git", "push"): "",
        ("gh", "run", "list"): "42",
        ("gh", "repo", "view"): "leehart3000/prokoptonml",
        ("gh", "run", "view", "42", "--json", "url"): "https://github.com/run/42",
        RUN_VIEW: "completed success",
    }
    return Fake(answers)


def pyproject(tmp_path: Path, version: str = "0.2.0a4") -> Path:
    path = tmp_path / "pyproject.toml"
    path.write_text(f'[project]\nname = "x"\nversion = "{version}"\n', "utf-8")
    return path


# --- A whole release ---


def test_a_platform_release_is_approved_here_and_checked_on_pypi(
    tmp_path: Path,
) -> None:
    fake = ready()
    fake.answers[RUN_VIEW] = ["in_progress ", "waiting ", "completed success"]
    fake.answers[("gh", "api", PENDING, "--jq")] = "27309708932"
    fake.answers[("gh", "api", "--method", "POST")] = ""
    url = "https://pypi.org/pypi/prokopton-ml/0.2.0a4/json"
    fake.pages[url] = [(404, ""), (200, "{}")]
    spec = Release(
        "v0.2.0a4",
        "prokopton-ml",
        version_file=pyproject(tmp_path),
        pypi="prokopton-ml",
    )

    done = release(spec, fake.tools())

    assert done == (
        "Released prokopton-ml 0.2.0a4; https://pypi.org/project/prokopton-ml/0.2.0a4/"
    )
    assert ("git", "tag", "-a", "v0.2.0a4", "-m", "prokopton-ml 0.2.0a4") in (
        fake.commands
    )
    assert ("git", "push", "--quiet", "origin", "v0.2.0a4") in fake.commands
    assert fake.asked == ["The run is waiting. Publish prokopton-ml 0.2.0a4 to PyPI?"]
    approval = next(
        c for c in fake.commands if c[:4] == ("gh", "api", "--method", "POST")
    )
    assert approval == (
        "gh", "api", "--method", "POST", PENDING,
        "-F", "environment_ids[]=27309708932",
        "-f", "state=approved",
        "-f", "comment=Release v0.2.0a4",
    )  # fmt: skip
    assert "Changes since v0.2.0a3:\ne1dfb58 feat: a change" in fake.said
    assert "Approved; publishing to PyPI" in fake.said


def test_a_product_release_checks_the_live_page() -> None:
    fake = ready("v1.0.15")
    fake.pages["https://dnabarcode.org/"] = [
        (200, "<span data-version>1.0.14/c6c9fc3</span>"),
        (200, "<footer><span data-version>1.0.15/abc1234</span></footer>"),
    ]
    spec = Release("v1.0.15", "DNAbarcode.org", check_url="https://dnabarcode.org/")

    done = release(spec, fake.tools())

    assert done == (
        "Released DNAbarcode.org 1.0.15; https://dnabarcode.org/ shows 1.0.15/abc1234"
    )
    assert fake.slept == [10]
    assert not any(command[:2] == ("gh", "api") for command in fake.commands)


def test_a_waiting_run_is_not_offered_for_approval_without_pypi() -> None:
    fake = ready("v1.0.15")
    fake.answers[RUN_VIEW] = ["waiting ", "completed success"]

    release(Release("v1.0.15", "app"), fake.tools())

    assert fake.asked == []


def test_nothing_is_offered_until_an_approval_is_possible(tmp_path: Path) -> None:
    fake = ready()
    fake.answers[RUN_VIEW] = ["waiting ", "waiting ", "completed success"]
    fake.answers[("gh", "api", PENDING, "--jq")] = ["", "27309708932"]
    fake.answers[("gh", "api", "--method", "POST")] = ""
    fake.pages["https://pypi.org/pypi/p/0.2.0a4/json"] = [(200, "{}")]

    release(Release("v0.2.0a4", "p", pypi="p"), fake.tools())

    assert len(fake.asked) == 1


def test_the_first_release_lists_every_commit() -> None:
    fake = ready("v1.0.0")
    fake.answers[("git", "describe")] = ReleaseError("no tags")

    release(Release("v1.0.0", "app"), fake.tools())

    assert ("git", "log", "--oneline", "HEAD") in fake.commands
    assert fake.said[0].startswith("Changes since the first commit:")


def test_a_dry_run_checks_and_lists_the_changes_but_does_not_tag() -> None:
    fake = ready("v1.0.15")

    done = release(Release("v1.0.15", "app", dry_run=True), fake.tools())

    assert done == "Ready to release app 1.0.15 (dry run: not tagged)"
    assert fake.said[0].startswith("Changes since v0.2.0a3:")
    assert not any(c[:3] == ("git", "tag", "-a") for c in fake.commands)
    assert not any(c[0] == "gh" or c[:2] == ("git", "push") for c in fake.commands)


def test_the_version_is_the_tag_without_its_v() -> None:
    assert Release("v1.2.3rc1", "app").version == "1.2.3rc1"


# --- Checks before tagging ---


@pytest.mark.parametrize("tag", ["1.0.0", "v1.0", "v1.0.0-beta", "v1.0.0a"])
def test_a_tag_must_be_a_version(tag: str) -> None:
    with pytest.raises(ReleaseError, match="Not a version"):
        release(Release(tag, "app"))  # Stops before using any tools


def test_the_tag_must_match_the_package_version(tmp_path: Path) -> None:
    spec = Release("v0.2.0a4", "p", version_file=pyproject(tmp_path, "0.2.0a3"))

    with pytest.raises(
        ReleaseError, match=re.escape("says version 0.2.0a3, not 0.2.0a4")
    ):
        release(spec, ready().tools())


def test_an_unreadable_version_file_is_an_error(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\n", "utf-8")
    spec = Release("v0.2.0a4", "p", version_file=tmp_path / "pyproject.toml")

    with pytest.raises(ReleaseError, match="Could not read the version"):
        release(spec, ready().tools())


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({("git", "branch", "--show-current"): "feat/x"}, "Not on main"),
        ({("git", "status", "--porcelain"): " M README.md"}, "uncommitted"),
        ({("git", "rev-parse", "origin/main"): "def456"}, "differs from GitHub"),
        ({tag_list("v1.0.15"): "v1.0.15"}, "v1.0.15 already exists"),
        (
            {("git", "tag", "--points-at", "HEAD", "--list", "v*"): "v1.0.14\nv1.0.13"},
            "already released as v1.0.14 v1.0.13",
        ),
    ],
)
def test_nothing_is_tagged_unless_main_is_ready(
    change: dict[tuple[str, ...], object], message: str
) -> None:
    fake = ready("v1.0.15")
    fake.answers.update(change)

    with pytest.raises(ReleaseError, match=message):
        release(Release("v1.0.15", "app"), fake.tools())

    assert not any(command[:3] == ("git", "tag", "-a") for command in fake.commands)


# --- Following the run ---


def test_a_run_that_never_starts_is_an_error() -> None:
    fake = ready("v1.0.15")
    fake.answers[("gh", "run", "list")] = ""

    with pytest.raises(
        ReleaseError, match=re.escape("No Release run started for v1.0.15")
    ):
        release(Release("v1.0.15", "app"), fake.tools())

    assert fake.slept == [5] * 24


def test_a_failed_run_is_an_error() -> None:
    fake = ready("v1.0.15")
    fake.answers[RUN_VIEW] = "completed failure"

    with pytest.raises(ReleaseError, match="The release run ended: failure"):
        release(Release("v1.0.15", "app"), fake.tools())


def test_a_run_without_a_conclusion_is_an_error() -> None:
    fake = ready("v1.0.15")
    fake.answers[RUN_VIEW] = "completed "

    with pytest.raises(ReleaseError, match="ended: unknown"):
        release(Release("v1.0.15", "app"), fake.tools())


def test_a_run_that_does_not_finish_is_an_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(module, "FOLLOW_CHECKS", 3)
    fake = ready("v1.0.15")
    fake.answers[RUN_VIEW] = "in_progress "

    with pytest.raises(ReleaseError, match="still running after an hour"):
        release(Release("v1.0.15", "app"), fake.tools())

    assert fake.slept == [10, 10, 10]


def test_declining_the_approval_leaves_the_run_waiting() -> None:
    fake = ready()
    fake.answers[RUN_VIEW] = "waiting "
    fake.answers[("gh", "api", PENDING, "--jq")] = "27309708932"
    fake.yes = False

    with pytest.raises(ReleaseError, match="Not approved; the run is waiting"):
        release(Release("v0.2.0a4", "p", pypi="p"), fake.tools())

    assert not any(c[:4] == ("gh", "api", "--method", "POST") for c in fake.commands)


# --- Checks afterwards ---


def test_a_version_pypi_never_lists_is_an_error() -> None:
    fake = ready()
    fake.pages["https://pypi.org/pypi/p/0.2.0a4/json"] = [(404, "")]

    with pytest.raises(ReleaseError, match=re.escape("PyPI does not list p 0.2.0a4")):
        release(Release("v0.2.0a4", "p", pypi="p"), fake.tools())

    assert fake.slept == [10] * 5


def test_pypi_is_asked_until_it_lists_the_version() -> None:
    fake = Fake({})
    fake.pages["https://pypi.org/pypi/p/1/json"] = [(404, ""), (404, ""), (200, "")]

    assert pypi_lists("p", "1", fake.tools())
    assert fake.slept == [10, 10]


def test_a_live_page_without_the_new_version_is_an_error() -> None:
    fake = ready("v1.0.15")
    fake.pages["https://x.test/"] = [(503, "")]

    with pytest.raises(
        ReleaseError, match=re.escape("shows no version, not 1.0.15/abc1234")
    ):
        release(Release("v1.0.15", "app", check_url="https://x.test/"), fake.tools())


# --- The real tools ---


def test_a_command_gives_its_output() -> None:
    assert module._run([sys.executable, "-c", "print(' hello ')"]) == "hello"


def test_a_failing_command_is_an_error_with_its_message() -> None:
    command = [sys.executable, "-c", "import sys; sys.exit('went wrong')"]

    with pytest.raises(ReleaseError, match="failed: went wrong"):
        module._run(command)


def test_a_failing_command_without_errors_shows_its_output() -> None:
    command = [sys.executable, "-c", "import sys; print('said'); sys.exit(3)"]

    with pytest.raises(ReleaseError, match="failed: said"):
        module._run(command)


class FakeResponse(io.BytesIO):
    status = 200


def test_pages_are_fetched_with_the_standard_library(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def urlopen(url: str, timeout: float) -> FakeResponse:
        assert timeout == 30
        if url.endswith("/missing"):
            raise urllib.error.HTTPError(url, 404, "Not Found", Message(), None)
        if url.endswith("/down"):
            raise urllib.error.URLError("no route")
        return FakeResponse(b"<p>ok</p>")

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)

    assert module._fetch("https://x.test/") == (200, "<p>ok</p>")
    assert module._fetch("https://x.test/missing") == (404, "")
    assert module._fetch("https://x.test/down") == (0, "")


@pytest.mark.parametrize(
    ("typed", "answer"), [("y", True), (" Yes ", True), ("", False), ("n", False)]
)
def test_only_yes_approves(
    monkeypatch: pytest.MonkeyPatch, typed: str, answer: bool
) -> None:
    monkeypatch.setattr(builtins, "input", lambda _: typed)

    assert module._ask("Publish?") is answer


def test_no_answer_does_not_approve(monkeypatch: pytest.MonkeyPatch) -> None:
    def closed(_: str) -> str:
        raise EOFError

    monkeypatch.setattr(builtins, "input", closed)

    assert module._ask("Publish?") is False


# --- The command line ---


@pytest.mark.parametrize(
    ("extra", "name"),
    [
        (["--pypi", "prokopton-ml"], "prokopton-ml"),
        (["--name", "DNAbarcode.org"], "DNAbarcode.org"),
        ([], "here"),
    ],
)
def test_the_command_makes_a_release(
    extra: list[str],
    name: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "here").mkdir()
    monkeypatch.chdir(tmp_path / "here")
    made: list[Release] = []

    def fake_release(spec: Release) -> str:
        made.append(spec)
        return "released"

    monkeypatch.setattr(cli, "release", fake_release)

    assert (
        cli.main(["release", "v1.0.15", "--check-url", "https://x.test/", *extra]) == 0
    )
    assert made[0].name == name
    assert made[0].check_url == "https://x.test/"
    assert made[0].workflow == "release.yml"
    assert made[0].dry_run is False


def test_the_command_passes_on_a_dry_run(monkeypatch: pytest.MonkeyPatch) -> None:
    made: list[Release] = []

    def fake_release(spec: Release) -> str:
        made.append(spec)
        return "ready"

    monkeypatch.setattr(cli, "release", fake_release)

    assert cli.main(["release", "v1.0.15", "--dry-run"]) == 0
    assert made[0].dry_run is True


def test_the_command_reports_a_stopped_release(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    def stopped(_: Release) -> str:
        raise ReleaseError("Not on main (git switch main)")

    monkeypatch.setattr(cli, "release", stopped)

    assert cli.main(["release", "v1.0.15"]) == 1
    assert "Not on main" in caplog.text
