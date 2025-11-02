#!/usr/bin/env bash
set -euo pipefail
python -m src.training.train_highway --config configs/config.yaml
