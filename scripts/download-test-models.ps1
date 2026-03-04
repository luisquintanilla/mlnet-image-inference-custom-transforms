#!/usr/bin/env pwsh
# Downloads test ONNX models for development and CI testing.
# Models are sourced from ONNX Model Zoo, HuggingFace, and Ultralytics.
#
# Some models require Python with 'optimum[onnxruntime]' and/or 'ultralytics'
# packages when direct downloads are unavailable.
#
# Usage: ./scripts/download-test-models.ps1 [-ModelDir <path>]

param(
    [string]$ModelDir = "models"
)

$ErrorActionPreference = "Stop"

function Download-Model {
    param([string]$Url, [string]$OutPath)
    
    $dir = Split-Path $OutPath -Parent
    if (!(Test-Path $dir)) { New-Item -ItemType Directory -Path $dir -Force | Out-Null }
    
    if (Test-Path $OutPath) {
        Write-Host "  Already exists: $OutPath" -ForegroundColor Yellow
        return
    }
    
    Write-Host "  Downloading: $Url"
    Write-Host "  To: $OutPath"
    Invoke-WebRequest -Uri $Url -OutFile $OutPath -UseBasicParsing
    Write-Host "  Done ($([math]::Round((Get-Item $OutPath).Length / 1MB, 1)) MB)" -ForegroundColor Green
}

function Export-OnnxModel {
    param([string]$HfModel, [string]$OutDir, [string]$Label)
    
    if (!(Test-Path $OutDir)) { New-Item -ItemType Directory -Path $OutDir -Force | Out-Null }
    
    $onnxFile = Join-Path $OutDir "model.onnx"
    if (Test-Path $onnxFile) {
        Write-Host "  Already exists: $onnxFile" -ForegroundColor Yellow
        return
    }
    
    Write-Host "  Exporting $HfModel -> $OutDir"
    python -m optimum.exporters.onnx --model $HfModel $OutDir 2>&1 | ForEach-Object {
        if ($_ -match "export success|saved at") { Write-Host "  $_" -ForegroundColor Green }
    }
    
    if (Test-Path $onnxFile) {
        Write-Host "  Done ($([math]::Round((Get-Item $onnxFile).Length / 1MB, 1)) MB)" -ForegroundColor Green
    } else {
        Write-Host "  ERROR: Export did not produce $onnxFile" -ForegroundColor Red
        exit 1
    }
}

$totalSteps = 7
$step = 0

Write-Host "=== Downloading Test Models ===" -ForegroundColor Cyan
Write-Host ""

# ── 1. SqueezeNet 1.0 — tiny image classification model (~5MB, 1000 ImageNet classes)
# Input:  data_0        [1, 3, 224, 224]  float
# Output: softmaxout_1  [1, 1000, 1, 1]   float
$step++; Write-Host "[$step/$totalSteps] SqueezeNet 1.0 (Classification)" -ForegroundColor White
Download-Model `
    -Url "https://huggingface.co/onnxmodelzoo/squeezenet1.0-7/resolve/main/squeezenet1.0-7.onnx" `
    -OutPath "$ModelDir/squeezenet/model.onnx"

# ── 2. ImageNet class labels (used by SqueezeNet and MobileNet)
$step++; Write-Host "[$step/$totalSteps] ImageNet Labels" -ForegroundColor White
Download-Model `
    -Url "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt" `
    -OutPath "$ModelDir/squeezenet/imagenet_classes.txt"

# ── 3. YOLOv8n — object detection (~12MB, 80 COCO classes)
# Input:  images   [1, 3, 640, 640]  float
# Output: output0  [1, 84, 8400]     float  (84 = 4 bbox coords + 80 class scores)
$step++; Write-Host "[$step/$totalSteps] YOLOv8n (Object Detection)" -ForegroundColor White
$yoloDir = "$ModelDir/yolov8"
$yoloOnnx = "$yoloDir/model.onnx"
if (!(Test-Path $yoloDir)) { New-Item -ItemType Directory -Path $yoloDir -Force | Out-Null }
if (Test-Path $yoloOnnx) {
    Write-Host "  Already exists: $yoloOnnx" -ForegroundColor Yellow
} else {
    # Try direct download first; fall back to ultralytics Python export
    $yoloUrl = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.onnx"
    try {
        Write-Host "  Downloading: $yoloUrl"
        Invoke-WebRequest -Uri $yoloUrl -OutFile $yoloOnnx -UseBasicParsing -ErrorAction Stop
        Write-Host "  Done ($([math]::Round((Get-Item $yoloOnnx).Length / 1MB, 1)) MB)" -ForegroundColor Green
    } catch {
        Write-Host "  Direct download failed, exporting via ultralytics Python package..." -ForegroundColor Yellow
        python -c "from ultralytics import YOLO; m = YOLO('yolov8n.pt'); m.export(format='onnx')" 2>&1 | Out-Null
        $exportedPath = "yolov8n.onnx"
        if (Test-Path $exportedPath) {
            Move-Item -Path $exportedPath -Destination $yoloOnnx -Force
            if (Test-Path "yolov8n.pt") { Remove-Item "yolov8n.pt" }
            Write-Host "  Done ($([math]::Round((Get-Item $yoloOnnx).Length / 1MB, 1)) MB)" -ForegroundColor Green
        } else {
            Write-Host "  ERROR: YOLOv8n export failed" -ForegroundColor Red
            exit 1
        }
    }
}

# ── 4. MobileNetV2 — image classification (~14MB, 1001 classes)
# Input:  pixel_values  [batch_size, 3, 224, 224]  float
# Output: logits        [batch_size, 1001]          float
$step++; Write-Host "[$step/$totalSteps] MobileNetV2 (Classification)" -ForegroundColor White
$mobilenetDir = "$ModelDir/mobilenet"
$mobilenetUrl = "https://github.com/onnx/models/raw/main/validated/vision/classification/mobilenet/model/mobilenetv2-12.onnx"
if (!(Test-Path $mobilenetDir)) { New-Item -ItemType Directory -Path $mobilenetDir -Force | Out-Null }
$mobilenetOnnx = "$mobilenetDir/model.onnx"
if (Test-Path $mobilenetOnnx) {
    Write-Host "  Already exists: $mobilenetOnnx" -ForegroundColor Yellow
} else {
    try {
        Write-Host "  Downloading: $mobilenetUrl"
        Invoke-WebRequest -Uri $mobilenetUrl -OutFile $mobilenetOnnx -UseBasicParsing -ErrorAction Stop
        Write-Host "  Done ($([math]::Round((Get-Item $mobilenetOnnx).Length / 1MB, 1)) MB)" -ForegroundColor Green
    } catch {
        Write-Host "  Direct download failed, exporting via optimum..." -ForegroundColor Yellow
        Export-OnnxModel -HfModel "google/mobilenet_v2_1.0_224" -OutDir $mobilenetDir -Label "MobileNetV2"
    }
}

# ── 5. SegFormer-b0 — semantic segmentation (~15MB, 150 ADE20K classes)
# Input:  pixel_values  [batch_size, num_channels, height, width]  float
# Output: logits        [batch_size, 150, 56, 56]                  float
$step++; Write-Host "[$step/$totalSteps] SegFormer-b0 (Segmentation)" -ForegroundColor White
Export-OnnxModel -HfModel "nvidia/segformer-b0-finetuned-ade-512-512" -OutDir "$ModelDir/segformer" -Label "SegFormer-b0"

# ── 6. CLIP ViT-B/32 — combined vision+text model for embeddings & zero-shot (~578MB)
# Inputs:
#   input_ids       [text_batch_size, sequence_length]               int64
#   pixel_values    [batch_size, num_channels, height, width]        float
#   attention_mask  [text_batch_size, sequence_length]               int64
# Outputs:
#   logits_per_image  [image_batch_size, text_batch_size]  float
#   logits_per_text   [text_batch_size, image_batch_size]  float
#   text_embeds       [text_batch_size, 512]               float
#   image_embeds      [image_batch_size, 512]              float
$step++; Write-Host "[$step/$totalSteps] CLIP ViT-B/32 (Embeddings + Zero-Shot)" -ForegroundColor White
Export-OnnxModel -HfModel "openai/clip-vit-base-patch32" -OutDir "$ModelDir/clip" -Label "CLIP ViT-B/32"

# ── 7. CLIP tokenizer files (vocab.json, merges.txt)
$step++; Write-Host "[$step/$totalSteps] CLIP Tokenizer Files" -ForegroundColor White
$clipDir = "$ModelDir/clip"
Download-Model `
    -Url "https://huggingface.co/openai/clip-vit-base-patch32/resolve/main/vocab.json" `
    -OutPath "$clipDir/vocab.json"
Download-Model `
    -Url "https://huggingface.co/openai/clip-vit-base-patch32/resolve/main/merges.txt" `
    -OutPath "$clipDir/merges.txt"

Write-Host ""
Write-Host "=== All models downloaded ===" -ForegroundColor Green
Write-Host "Models directory: $ModelDir"
Get-ChildItem $ModelDir -Recurse -File | ForEach-Object { 
    Write-Host "  $($_.FullName.Replace((Resolve-Path $ModelDir).Path, $ModelDir)) ($([math]::Round($_.Length / 1MB, 1)) MB)" 
}
