#!/usr/bin/env bash
set -euo pipefail

REPO="JacobNewmes/CLARA-CoreML"
DEST="$(cd "$(dirname "$0")/.." && pwd)/CLARA/Resources"

models=(
    "MedASR.mlpackage"
    "MedSigLIP_VisionEncoder.mlpackage"
)

for model in "${models[@]}"; do
    if [ -d "$DEST/$model" ]; then
        echo "✓ $model already exists, skipping"
        continue
    fi
    echo "↓ Downloading $model …"
    huggingface-cli download "$REPO" "$model" \
        --repo-type model \
        --local-dir "$DEST" \
        --include "$model/*"
    echo "✓ $model downloaded"
done

echo "Done. Models are in $DEST"
