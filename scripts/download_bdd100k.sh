#!/usr/bin/env bash
set -euo pipefail

# Simple unauthenticated CURL download as requested
# Note: Kaggle API usually requires credentials; this may return 403 without them.
# This script saves the zip to data/ and unzips into data/

DEST_DIR="data"
ZIP_PATH="data/solesensei_bdd100k.zip"
URL="https://www.kaggle.com/api/v1/datasets/download/solesensei/solesensei_bdd100k"

mkdir -p "$DEST_DIR"

echo "Downloading to $ZIP_PATH ..."
curl -L -o "$ZIP_PATH" "$URL"

echo "Unzipping to $DEST_DIR ..."
unzip -o "$ZIP_PATH" -d "$DEST_DIR"

echo "Done. Files are in $DEST_DIR"
