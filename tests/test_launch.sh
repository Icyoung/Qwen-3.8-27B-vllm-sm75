#!/usr/bin/env bash
# Public launcher regression entry; tests substitute Docker with an argv logger.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
python3 -m unittest discover -s tests -p test_release_scripts.py -v
