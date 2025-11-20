#!/usr/bin/env bash
set -euo pipefail

# Create a small subset from BDD100K 10k split for quick Grad-CAM demos
# Usage: scripts/sample_bdd100k_subset.sh [train_count] [val_count]
# Defaults: train_count=800, val_count=200

TRAIN_SRC="data/bdd100k/bdd100k/images/10k/train"
VAL_SRC="data/bdd100k/bdd100k/images/10k/val"
TRAIN_DST="data/bdd100k_subset/train"
VAL_DST="data/bdd100k_subset/val"

TRAIN_N="${1:-800}"
VAL_N="${2:-200}"

mkdir -p "$TRAIN_DST" "$VAL_DST"

# macOS shuf alternative if shuf not available: gshuf from coreutils; otherwise Python fallback
have_shuf=true
if ! command -v shuf >/dev/null 2>&1; then
  if command -v gshuf >/dev/null 2>&1; then
    alias shuf=gshuf
  else
    have_shuf=false
  fi
fi

if [ "$have_shuf" = true ]; then
  # Copy samples without overwriting existing using shuf
  find "$TRAIN_SRC" -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) | shuf -n "$TRAIN_N" | xargs -I{} cp -n "{}" "$TRAIN_DST" || true
  find "$VAL_SRC" -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) | shuf -n "$VAL_N" | xargs -I{} cp -n "{}" "$VAL_DST" || true
else
  echo "shuf not found; using Python fallback for sampling. (brew install coreutils for gshuf)"
  echo "TRAIN_SRC=$TRAIN_SRC"
  echo "VAL_SRC=$VAL_SRC"
  echo "TRAIN_DST=$TRAIN_DST"
  echo "VAL_DST=$VAL_DST"
  TRAIN_SRC="$TRAIN_SRC" \
  VAL_SRC="$VAL_SRC" \
  TRAIN_DST="$TRAIN_DST" \
  VAL_DST="$VAL_DST" \
  TRAIN_N="$TRAIN_N" \
  VAL_N="$VAL_N" \
  python3 - <<'PY'
import os, random, shutil, sys
train_src = os.environ.get('TRAIN_SRC')
val_src = os.environ.get('VAL_SRC')
train_dst = os.environ.get('TRAIN_DST')
val_dst = os.environ.get('VAL_DST')
train_n = int(os.environ.get('TRAIN_N','800'))
val_n = int(os.environ.get('VAL_N','200'))

def list_images(root):
    exts = {'.jpg','.jpeg','.png','.JPG','.JPEG','.PNG'}
    paths = []
    for dp,_,fnames in os.walk(root):
        for f in fnames:
            if os.path.splitext(f)[1] in exts:
                paths.append(os.path.join(dp,f))
    return paths

def sample_and_copy(src, dst, n):
    os.makedirs(dst, exist_ok=True)
    files = list_images(src)
    if not files:
        print(f"No images found in {src}")
        return
    k = min(n, len(files))
    for p in random.sample(files, k):
        try:
            shutil.copy2(p, os.path.join(dst, os.path.basename(p)))
        except Exception as e:
            pass

sample_and_copy(train_src, train_dst, train_n)
sample_and_copy(val_src, val_dst, val_n)
print(f"Python fallback: subset created at {train_dst} and {val_dst}")
PY
fi

echo "Subset created at $TRAIN_DST and $VAL_DST"}
