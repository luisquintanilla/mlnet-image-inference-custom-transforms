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
}
