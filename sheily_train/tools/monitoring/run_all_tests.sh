#!/usr/bin/env bash
set -euo pipefail
pytest -v tests_sheily_suite/ --maxfail=1
