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

echo "[1/5] SqueezeNet 1.0 (Classification)"
download_model \
    "https://huggingface.co/onnxmodelzoo/squeezenet1.0-7/resolve/main/squeezenet1.0-7.onnx" \
    "$MODEL_DIR/squeezenet/model.onnx"

echo "[2/5] ImageNet Labels"
download_model \
    "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt" \
    "$MODEL_DIR/squeezenet/imagenet_classes.txt"

# ── 3. DINOv2 ViT-S/14 — self-supervised vision embeddings (~83MB, 384-dim output)
# Input:  input   [1, 3, 224, 224]  float
# Output: output  [1, 384]          float
echo "[3/5] DINOv2 ViT-S/14 (Embeddings)"
download_model \
    "https://huggingface.co/sefaburak/dinov2-small-onnx/resolve/main/dinov2_vits14.onnx" \
    "$MODEL_DIR/dinov2/dinov2_vits14.onnx"

# ── 4. ResNet-50 v1.7 — image classification (~98MB, 1000 ImageNet classes)
# Input:  data                [N, 3, 224, 224]  float
# Output: resnetv17_dense0_fwd  [N, 1000]       float
echo "[4/5] ResNet-50 v1.7 (Classification)"
download_model \
    "https://huggingface.co/onnxmodelzoo/resnet50-v1-7/resolve/main/resnet50-v1-7.onnx" \
    "$MODEL_DIR/resnet50/resnet50-v1-7.onnx"

# ── 5. DeepLabV3-ResNet50 — semantic segmentation (~160MB, 21 VOC classes)
# Input:  pixel_values  [batch, 3, 520, 520]  float
# Output: logits        [batch, 21, H, W]     float
echo "[5/5] DeepLabV3-ResNet50 (Segmentation)"
DEEPLAB_DIR="$MODEL_DIR/deeplabv3"
DEEPLAB_ONNX="$DEEPLAB_DIR/deeplabv3_resnet50.onnx"
mkdir -p "$DEEPLAB_DIR"
if [ -f "$DEEPLAB_ONNX" ]; then
    echo "  Already exists: $DEEPLAB_ONNX"
else
    echo "  Exporting DeepLabV3-ResNet50 via torch.onnx.export..."
    python3 -c "
import torch
import torchvision.models.segmentation as seg
model = seg.deeplabv3_resnet50(weights=seg.DeepLabV3_ResNet50_Weights.DEFAULT)
model.eval()
dummy = torch.randn(1, 3, 520, 520)
torch.onnx.export(model, dummy, '$DEEPLAB_ONNX',
    input_names=['pixel_values'], output_names=['logits'],
    dynamic_axes={'pixel_values': {0: 'batch'}, 'logits': {0: 'batch'}},
    opset_version=17, do_constant_folding=True)
print('Done')
"
    if [ -f "$DEEPLAB_ONNX" ]; then
        size=$(du -h "$DEEPLAB_ONNX" | cut -f1)
        echo "  Done ($size)"
    else
        echo "  ERROR: DeepLabV3 export failed"
        exit 1
    fi
fi

echo ""
echo "=== All models downloaded ==="
echo "Models directory: $MODEL_DIR"
find "$MODEL_DIR" -type f -exec ls -lh {} \;
