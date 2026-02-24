#!/usr/bin/env bash
set -euo pipefail

REPO="JacobNewmes/coreml-medsiglip-448"
DEST="$(cd "$(dirname "$0")/.." && pwd)/CLARA/Resources"

models=(
    "MedSigLIP_VisionEncoder.mlpackage"
)

for model in "${models[@]}"; do
    if [ -d "$DEST/$model" ]; then
        echo "✓ $model already exists, skipping"
        continue
    fi
    echo "↓ Downloading $model …"
    python3 -m huggingface_hub.commands.huggingface_cli download "$REPO" \
        --repo-type model \
        --local-dir "$DEST" \
        --include "$model/*"
    echo "✓ $model downloaded"
done

echo "Done. Models are in $DEST"
