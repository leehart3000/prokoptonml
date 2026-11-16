"""Command-line interface for ProkoptonML.

    prokopton-ml stages                      the pipeline, and when each stage ran
    prokopton-ml run STAGE [STAGE ...]       run stages (or --all, or --from STAGE)
    prokopton-ml identify --input FILE       identify one sequence
    prokopton-ml migrate-config [--write]    rewrite a schema 3 config as schema 4
    prokopton-ml check BUNDLE CHECKS [--url URL]
                                             run known-answer checks
    prokopton-ml release vVERSION [...]      tag main and follow the release

The config is ``prokopton.toml`` in this folder or a parent folder, unless
``--config PATH`` is given. ``check``, ``release`` and ``identify --bundle``
need no config.
"""

import argparse
import json
import logging
import os
import sys
import tomllib
from collections.abc import Sequence
from pathlib import Path

from prokopton.core.config import ConfigError, load_config

from prokopton.ml.checks import (
    CheckError,
    bundle_config,
    read_checks,
    report,
    run_local,
    run_remote,
)
from prokopton.ml.config import AppConfig, UnknownClassifierError
from prokopton.ml.inference.bundle import BundleError, load_bundle
from prokopton.ml.inference.load import LoadError, load_identifier
from prokopton.ml.migrate import (
    CURRENT_SCHEMA,
    MigrationError,
    migrate_text,
    schema_of,
)
from prokopton.ml.release import Release, ReleaseError, release
from prokopton.ml.stages import (
    Registry,
    StageError,
    core_kinds,
    entry_point_kinds,
    plan,
    read_record,
    run_stage,
    settings_hash,
)

__all__ = ["CONFIG_NAME", "ENV_PREFIX", "main"]

CONFIG_NAME = "prokopton.toml"
"""The app's config file, found in the current folder or a parent folder."""

ENV_PREFIX = "PROKOPTON_"
"""Prefix for environment variables that override config values, e.g.
``PROKOPTON_PATHS__MODELS_DIR``."""

logger = logging.getLogger("prokopton.ml")


def registry() -> Registry:
    """Every stage kind available: the core's, and every installed package's
    (the platform's own ``bio`` kinds among them) through entry points."""
    return Registry([*core_kinds(), *entry_point_kinds()])


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="prokopton-ml",
        description="Build, deploy and monitor scientific ML applications.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help=f"The app's config file (default: {CONFIG_NAME} in this folder or a "
        "parent folder, whose folder then becomes the working folder).",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser(
        "stages",
        help="List the pipeline's stages, when each last ran, and whether its "
        "settings have changed since.",
    )
    run = commands.add_parser("run", help="Run pipeline stages.")
    run.add_argument("stages", nargs="*", help="Stages to run, in this order.")
    run.add_argument("--all", action="store_true", help="Run every stage, in order.")
    run.add_argument("--from", dest="start", help="Run the pipeline from this stage.")
    identify = commands.add_parser(
        "identify",
        help="Identify one sequence and print the result, with its decision "
        "trace, as JSON.",
    )
    identify.add_argument(
        "--input",
        type=Path,
        required=True,
        help="A file with one submission (for sequence apps, one sequence); "
        "'-' for stdin.",
    )
    identify.add_argument(
        "--bundle",
        type=Path,
        default=None,
        help="Use a serving bundle instead of the MLflow registry (no config needed).",
    )
    migrate = commands.add_parser(
        "migrate-config",
        help="Rewrite a schema 3 config file as the current schema, keeping "
        "its comments (prints it; --write replaces the file).",
    )
    migrate.add_argument("--write", action="store_true", help="Replace the file.")
    check = commands.add_parser(
        "check",
        help="Run known-answer checks (<name>.json requests and <name>.expected "
        "answers) against a serving bundle, or a running service; fails if any "
        "answer differs.",
    )
    check.add_argument("bundle", type=Path, help="The serving bundle.")
    check.add_argument("checks", type=Path, help="The folder of checks.")
    check.add_argument(
        "--url",
        default=None,
        help="Send the checks to the service at this URL instead (its API key, "
        "if any, from PROKOPTON_API_KEY).",
    )
    make = commands.add_parser(
        "release",
        help="Tag main with a new version, follow the Release workflow and check "
        "the result (see prokopton.ml.release).",
    )
    make.add_argument("tag", help="The version tag, such as v1.2.3 or v0.2.0a4.")
    make.add_argument(
        "--name",
        default=None,
        help="The name in the tag's message (default: the PyPI package, or "
        "this folder's name).",
    )
    make.add_argument(
        "--version-file",
        type=Path,
        default=None,
        help="A pyproject.toml whose version the tag must match.",
    )
    make.add_argument(
        "--pypi",
        default=None,
        metavar="PACKAGE",
        help="Offer to approve publishing to PyPI, then check PyPI lists it.",
    )
    make.add_argument(
        "--check-url",
        default=None,
        help="A live page that must show the new version (<span data-version>).",
    )
    make.add_argument(
        "--workflow", default="release.yml", help="The Release workflow's file."
    )
    make.add_argument(
        "--dry-run",
        action="store_true",
        help="Make the checks and list the changes, but do not tag.",
    )
    return parser


def _find_config(given: Path | None) -> Path:
    """The config file: as given, or found in this folder or a parent folder
    (whose folder then becomes the working folder, as paths in the config are
    relative to it)."""
    if given is not None:
        return given
    here = Path.cwd()
    for folder in (here, *here.parents):
        if (folder / CONFIG_NAME).is_file():
            if folder != here:
                os.chdir(folder)
                logger.info(
                    "Using %s (working folder %s)", folder / CONFIG_NAME, folder
                )
            return folder / CONFIG_NAME
    msg = f"No {CONFIG_NAME} in {here} or its parent folders; use --config PATH"
    raise ConfigError(msg)


def _load(path: Path, stages: Registry) -> AppConfig:
    _warn_if_schema_3(path)
    config = load_config(path, AppConfig, env_prefix=ENV_PREFIX)
    stages.check(config)
    return config


def _list_stages(config: AppConfig, stages: Registry) -> str:
    for number, name in enumerate(config.pipeline.stages, start=1):
        stage = config.stages[name]
        target = f" ({stage.model})" if stage.model else ""
        record = read_record(config, name)
        if record is None:
            status = "never run"
        else:
            changed = record.settings_sha256 != settings_hash(
                config, name, stages[stage.kind]
            )
            status = f"last run {record.finished_at:%Y-%m-%d %H:%M} UTC with " + (
                f"{record.platform_version}"
                + ("; settings changed since" if changed else "")
            )
        print(f"{number:3}  {name:<24} {stage.kind}{target}  [{status}]")
    return f"{len(config.pipeline.stages)} stages"


def _run_stages(config: AppConfig, stages: Registry, names: list[str]) -> str:
    for name in names:
        logger.info("Running %s (%s)", name, config.stages[name].kind)
        logger.info("%s: %s", name, run_stage(stages, config, name))
    return f"ran {len(names)} stage{'s' if len(names) != 1 else ''}"


def _identify(config: AppConfig | None, source: Path, bundle: Path | None) -> str:
    text = sys.stdin.read() if str(source) == "-" else source.read_text("utf-8")
    if bundle is not None:
        loaded = load_bundle(bundle)
    else:
        assert config is not None  # A config is loaded when there is no bundle
        loaded = load_identifier(config)
    result = loaded.identifier.identify(text)
    output = {"versions": loaded.versions, **result.model_dump(mode="json")}
    print(json.dumps(output, indent=2))
    return f"{result.status}: {result.answer.text if result.answer else result.reason}"


def _check(bundle: Path, folder: Path, url: str | None) -> str:
    checks = read_checks(folder)
    if url is None:
        outcomes = run_local(load_bundle(bundle).identifier, checks)
    else:
        endpoint = bundle_config(bundle).serving.endpoint
        key = os.environ.get("PROKOPTON_API_KEY", "")
        outcomes = run_remote(url, endpoint, checks, key=key)
    lines, failed = report(outcomes)
    for line in lines:
        print(line)
    if failed:
        msg = f"{failed} of {len(outcomes)} checks gave a different answer"
        raise CheckError(msg)
    return f"all {len(outcomes)} checks gave their expected answers"


def _release(args: argparse.Namespace) -> str:
    spec = Release(
        tag=args.tag,
        name=args.name or args.pypi or Path.cwd().name,
        version_file=args.version_file,
        pypi=args.pypi,
        check_url=args.check_url,
        workflow=args.workflow,
        dry_run=args.dry_run,
    )
    return release(spec)


def _migrate(path: Path, write: bool) -> str:
    """Rewrite an older config as the current schema: print it, or replace
    the file."""
    result = migrate_text(path.read_text("utf-8"))
    done = f"{path} rewritten as schema {CURRENT_SCHEMA}"
    if not write:
        sys.stdout.write(result)
        return f"{done} (printed; --write replaces the file)"
    path.write_text(result, "utf-8")
    return done


def _warn_if_schema_3(path: Path) -> None:
    try:
        data = tomllib.loads(path.read_text("utf-8"))
    except (OSError, tomllib.TOMLDecodeError):
        return  # load_config reports these
    if schema_of(data) == 3:  # Older schemas are refused by load_config
        logger.warning(
            "%s is schema 3, read by converting it; a later version will stop "
            "reading it. Run: prokopton-ml migrate-config --write",
            path,
        )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command line interface and return an exit code."""
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    expected: tuple[type[Exception], ...] = (
        ConfigError,
        StageError,
        UnknownClassifierError,
        MigrationError,
        LoadError,
        BundleError,
        CheckError,
        ReleaseError,
        OSError,
    )
    try:
        stages = registry()
        expected += stages.errors
        if args.command == "check":
            summary = _check(args.bundle, args.checks, args.url)
        elif args.command == "release":
            summary = _release(args)
        elif args.command == "identify" and args.bundle is not None:
            summary = _identify(None, args.input, args.bundle)
        else:
            path = _find_config(args.config)
            if args.command == "migrate-config":
                summary = _migrate(path, args.write)
            else:
                config = _load(path, stages)
                if args.command == "stages":
                    summary = _list_stages(config, stages)
                elif args.command == "identify":
                    summary = _identify(config, args.input, None)
                else:
                    names = plan(
                        config, args.stages, run_all=args.all, start=args.start
                    )
                    summary = _run_stages(config, stages, names)
    except expected as error:
        logger.error("%s", error)
        return 1
    logger.info("Done: %s", summary)
    return 0
