using Microsoft.Extensions.Logging;
using MLNet.Image.Core;

namespace MLNet.ImageGeneration.OnnxGenAI;

/// <summary>
/// Options for text-to-image generation using ONNX models (e.g., Stable Diffusion).
/// </summary>
public class OnnxImageGenerationOptions
{
    /// <summary>Path to the ONNX model directory (contains multiple model files for SD pipeline).</summary>
    public required string ModelDirectory { get; init; }

    /// <summary>Number of inference steps. Default: 20 for SDXL, 50 for SD 1.5.</summary>
    public int NumInferenceSteps { get; init; } = 20;

    /// <summary>Classifier-free guidance scale. Default: 7.5.</summary>
    public float GuidanceScale { get; init; } = 7.5f;

    /// <summary>Output image width. Default: 512.</summary>
    public int Width { get; init; } = 512;

    /// <summary>Output image height. Default: 512.</summary>
    public int Height { get; init; } = 512;

    /// <summary>Random seed for reproducibility. Null for random.</summary>
    public int? Seed { get; init; }

    /// <summary>Negative prompt for classifier-free guidance.</summary>
    public string? NegativePrompt { get; init; }

    /// <summary>
    /// Path to CLIP vocab.json file for text tokenization. 
    /// When null, uses a simple SOT+EOT tokenizer (no real text encoding).
    /// </summary>
    public string? VocabPath { get; init; }

    /// <summary>
    /// Path to CLIP merges.txt file for text tokenization.
    /// Required when VocabPath is set.
    /// </summary>
    public string? MergesPath { get; init; }

    /// <summary>
    /// Execution provider for hardware acceleration. Default: CPU.
    /// CUDA dramatically speeds up Stable Diffusion inference (minutes → seconds).
    /// </summary>
    public OnnxExecutionProvider ExecutionProvider { get; init; } = OnnxExecutionProvider.CPU;

    /// <summary>GPU device ID for CUDA/DirectML/TensorRT. Default: 0.</summary>
    public int GpuDeviceId { get; init; } = 0;

    /// <summary>
    /// If true, silently falls back to CPU when the requested GPU execution provider fails.
    /// Default: true.
    /// </summary>
    public bool FallbackToCpu { get; init; } = true;

    /// <summary>
    /// Optional logger for diagnostics (model load timing, denoising steps, GPU fallback).
    /// When null, no logging is performed.
    /// </summary>
    public ILogger? Logger { get; init; }
}
