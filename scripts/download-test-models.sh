#!/bin/bash
# Downloads small test ONNX models for development and CI testing.
set -e

MODEL_DIR="${1:-models}"

download_model() {
    local url="$1"
    local outpath="$2"
    local dir=$(dirname "$outpath")
    
    mkdir -p "$dir"
    
    if [ -f "$outpath" ]; then
        echo "  Already exists: $outpath"
        return
    fi
    
    echo "  Downloading: $url"
    echo "  To: $outpath"
    curl -L -o "$outpath" "$url"
    local size=$(du -h "$outpath" | cut -f1)
    echo "  Done ($size)"
}

echo "=== Downloading Test Models ==="
echo ""

echo "[1/2] SqueezeNet 1.0 (Classification)"
download_model \
    "https://huggingface.co/onnxmodelzoo/squeezenet1.0-7/resolve/main/squeezenet1.0-7.onnx" \
    "$MODEL_DIR/squeezenet/model.onnx"

echo "[2/2] ImageNet Labels"
download_model \
    "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt" \
    "$MODEL_DIR/squeezenet/imagenet_classes.txt"

echo ""
echo "=== All models downloaded ==="
echo "Models directory: $MODEL_DIR"
find "$MODEL_DIR" -type f -exec ls -lh {} \;
