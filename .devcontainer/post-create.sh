#!/bin/bash
set -e

echo "=== Restoring dependencies ==="
dotnet restore

echo "=== Building solution ==="
dotnet build --no-restore

echo ""
echo "=== Setup complete! ==="
echo ""
echo "To download models, install huggingface-cli:"
echo "  pip install huggingface-hub"
echo ""
echo "Then download a model, e.g.:"
echo "  huggingface-cli download google/vit-base-patch16-224 --include 'onnx/*' --local-dir models/vit-base"
echo ""
