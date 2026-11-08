#!/usr/bin/env bash
# Linux launcher: same as start.command (macOS double-click launcher).
exec "$(dirname "$0")/start.command" "$@"
