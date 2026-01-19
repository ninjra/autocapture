#!/usr/bin/env bash
set -euo pipefail

python -m autocapture.graph.cli_worker "$@"
