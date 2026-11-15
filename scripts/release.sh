#!/usr/bin/env bash
# Release prokopton-ml: tag main with the version in pyproject.toml, then
# follow the Release workflow (TestPyPI, then PyPI after your approval) and
# check that PyPI has the new version. Stops at the first problem (nothing
# after it runs).
#
#   scripts/release.sh v0.2.0a1
#
# Before tagging, it checks that you are on main with no uncommitted changes,
# that main matches GitHub, that the version matches pyproject.toml and is
# new, and that this commit has not been released already.
set -euo pipefail

stop() { echo "STOP: $*" >&2; exit 1; }

[ $# -eq 1 ] || stop "usage: scripts/release.sh vVERSION (for example v0.2.0a1)"
tag=$1
[[ $tag =~ ^v[0-9]+\.[0-9]+\.[0-9]+((a|b|rc)[0-9]+)?$ ]] \
  || stop "not a version like v1.2.3 or v1.2.3a1: $tag"

cd "$(git rev-parse --show-toplevel)"
version=$(sed -n 's/^version = "\(.*\)"$/\1/p' pyproject.toml | head -1)
[ "$tag" = "v$version" ] || stop "pyproject.toml says version $version, not ${tag#v}"
[ "$(git branch --show-current)" = main ] || stop "not on main (git switch main)"
[ -z "$(git status --porcelain)" ] || stop "uncommitted changes (see git status)"
git fetch --quiet --tags origin
[ "$(git rev-parse HEAD)" = "$(git rev-parse origin/main)" ] \
  || stop "main differs from GitHub (git pull)"
! git rev-parse --quiet --verify "refs/tags/$tag" > /dev/null || stop "$tag already exists"
released=$(git tag --points-at HEAD --list 'v*' | paste -sd ' ' -)
[ -z "$released" ] || stop "this commit is already released as $released"
last=$(git describe --tags --abbrev=0 --match 'v*' 2> /dev/null || true)
echo "Changes since ${last:-the first commit}:"
git log --oneline "${last:+$last..}HEAD"

git tag -a "$tag" -m "prokopton-ml $version"
git push --quiet origin "$tag"
run=
for _ in $(seq 1 24); do
  run=$(gh run list --workflow=release.yml --branch "$tag" --json databaseId --jq '.[0].databaseId // empty')
  [ -n "$run" ] && break
  sleep 5
done
[ -n "$run" ] || stop "no Release run started for $tag (gh run list --workflow=release.yml)"
echo "Release run $run: after TestPyPI, approve the PyPI step here:"
echo "  $(gh run view "$run" --json url --jq .url)"
gh run watch "$run" --exit-status > /dev/null || stop "the release failed or was rejected (gh run view $run --log-failed)"

for _ in 1 2 3 4 5 6; do
  status=$(curl --silent --output /dev/null --write-out '%{http_code}' "https://pypi.org/pypi/prokopton-ml/$version/json")
  [ "$status" = 200 ] && break
  sleep 10
done
[ "$status" = 200 ] || stop "PyPI does not list prokopton-ml $version yet (HTTP $status)"
echo "Released prokopton-ml $version: https://pypi.org/project/prokopton-ml/$version/"
