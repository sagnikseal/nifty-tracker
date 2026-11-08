#!/usr/bin/env bash
# Data-branch plumbing for the collector workflows (.github/workflows/collect_*.yml).
#
#   bash nifty_engine/collect/gha.sh checkout   # shallow, partial, sparse checkout of the data branch
#   bash nifty_engine/collect/gha.sh push       # commit + push, retrying on conflicts
#
# The branch grows by ~1.5-2 MB per session, so CI only checks out what a run reads or writes: today
# and the previous 5 IST days (news dedupe looks 4 days back, expired options up to 4 days) plus tomorrow, meta/, runs/ and the root files (MANIFEST.json, README.md). Blobs of other
# days are never downloaded (--filter=blob:none + sparse cone).
#
# Push conflicts: if the remote moved (e.g. a manual push), the working tree is reset to the new
# remote tip and this run's staging folder is merged again with `python -m nifty_engine.collect merge`
# (every merge rule is idempotent), then the push is retried. That is a conflict-free equivalent of
# `git pull --rebase` for append-only data.
#
# Environment: DATA_TOKEN (required; GITHUB_TOKEN or a PAT for DATA_REPO), DATA_REPO (owner/name,
# default: this repository), DATA_BRANCH (default: data), DATA_DIR_PATH (default: databranch),
# STAGING (push only: the collector's --out folder), MODE (commit message).
set -euo pipefail

cmd="${1:?usage: gha.sh checkout|push}"
WS="${GITHUB_WORKSPACE:-$PWD}"
DIR="${DATA_DIR_PATH:-databranch}"
case "$DIR" in /*) ;; *) DIR="$WS/$DIR" ;; esac
BRANCH="${DATA_BRANCH:-data}"
REPO="${DATA_REPO:-${GITHUB_REPOSITORY:?}}"
URL="${DATA_URL:-https://github.com/${REPO}.git}"   # DATA_URL: override for local testing
: "${DATA_TOKEN:?DATA_TOKEN is not set}"

B64="$(printf 'x-access-token:%s' "$DATA_TOKEN" | base64 | tr -d '\n')"
echo "::add-mask::${B64}"
AUTH="AUTHORIZATION: basic ${B64}"

sparse_dirs() {
  local i
  echo "meta runs"
  for i in -1 0 1 2 3 4 5; do
    TZ=Asia/Kolkata date -d "$((-i)) day" +"days/%F"
  done
}

identity() {
  git -C "$DIR" config user.name "github-actions[bot]"
  git -C "$DIR" config user.email "41898282+github-actions[bot]@users.noreply.github.com"
  git -C "$DIR" config http.https://github.com/.extraheader "$AUTH"
}

checkout() {
  local rc=0
  git -c http.extraheader="$AUTH" ls-remote --exit-code --heads "$URL" "$BRANCH" >/dev/null 2>&1 || rc=$?
  if [ "$rc" -eq 0 ]; then
    git -c http.extraheader="$AUTH" clone -q --depth 1 --filter=blob:none --no-checkout \
        --single-branch --branch "$BRANCH" "$URL" "$DIR"
    identity
    # shellcheck disable=SC2046
    git -C "$DIR" sparse-checkout set --cone $(sparse_dirs)
    git -C "$DIR" checkout -q "$BRANCH"
    echo "checked out ${REPO}@${BRANCH} (sparse: last 6 days) into ${DIR}"
  elif [ "$rc" -eq 2 ]; then
    # first run: the branch does not exist yet -> start an orphan branch (no history shared with main)
    git init -q "$DIR"
    git -C "$DIR" symbolic-ref HEAD "refs/heads/${BRANCH}"
    git -C "$DIR" remote add origin "$URL"
    identity
    echo "branch '${BRANCH}' not found in ${REPO}: creating it as an orphan branch"
  else
    echo "::error title=Data branch::cannot reach ${REPO} (git ls-remote exit ${rc}); check DATA_REPO / DATA_REPO_TOKEN"
    exit 1
  fi
}

commit() {
  git -C "$DIR" add -A
  if git -C "$DIR" diff --cached --quiet; then
    return 1
  fi
  git -C "$DIR" commit -q -m "collect: ${MODE:-run} $(TZ=Asia/Kolkata date +'%F %H:%M') IST [skip ci]"
}

push() {
  [ -d "$DIR/.git" ] || { echo "no data-branch checkout at $DIR"; exit 0; }
  if ! commit; then
    echo "nothing new to commit"
    exit 0
  fi
  local i
  for i in 1 2 3 4 5 6; do
    if git -C "$DIR" push -q origin "HEAD:refs/heads/${BRANCH}"; then
      echo "pushed to ${REPO}@${BRANCH} (attempt ${i})"
      exit 0
    fi
    echo "push rejected (attempt ${i}); re-applying this run's data on the new remote tip"
    sleep $((i * 5 + RANDOM % 5))
    git -C "$DIR" fetch -q --depth 1 origin "$BRANCH"
    git -C "$DIR" reset -q --hard FETCH_HEAD
    if [ -n "${STAGING:-}" ] && [ -d "$STAGING" ]; then
      (cd "$WS" && python -m nifty_engine.collect merge --out "$STAGING" --data-branch-dir "$DIR" >/dev/null)
    fi
    if ! commit; then
      echo "nothing left to push after re-merge"
      exit 0
    fi
  done
  echo "::error title=Data branch push::could not push after 6 attempts"
  exit 1
}

case "$cmd" in
  checkout) checkout ;;
  push) push ;;
  *) echo "unknown command: $cmd"; exit 2 ;;
esac
