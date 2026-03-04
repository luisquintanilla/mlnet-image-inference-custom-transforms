#!/usr/bin/env pwsh
# Downloads small test ONNX models for development and CI testing.
# These are pre-exported models from ONNX Model Zoo / HuggingFace.
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

Write-Host "=== Downloading Test Models ===" -ForegroundColor Cyan
Write-Host ""

# SqueezeNet 1.0 — tiny image classification model (~5MB, 1000 ImageNet classes)
Write-Host "[1/2] SqueezeNet 1.0 (Classification)" -ForegroundColor White
Download-Model `
    -Url "https://huggingface.co/onnxmodelzoo/squeezenet1.0-7/resolve/main/squeezenet1.0-7.onnx" `
    -OutPath "$ModelDir/squeezenet/model.onnx"

# Also download ImageNet class labels
Write-Host "[2/2] ImageNet Labels" -ForegroundColor White
Download-Model `
    -Url "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt" `
    -OutPath "$ModelDir/squeezenet/imagenet_classes.txt"

Write-Host ""
Write-Host "=== All models downloaded ===" -ForegroundColor Green
Write-Host "Models directory: $ModelDir"
Get-ChildItem $ModelDir -Recurse -File | ForEach-Object { 
    Write-Host "  $($_.FullName.Replace((Resolve-Path $ModelDir).Path, $ModelDir)) ($([math]::Round($_.Length / 1MB, 1)) MB)" 
}
