using Microsoft.ML.Data;

namespace MLNet.ImageGeneration.OnnxGenAI;

/// <summary>
/// Transformer for text-to-image generation using ONNX Stable Diffusion models.
/// 
/// STATUS: Stub implementation. Full implementation requires:
/// - Microsoft.ML.OnnxRuntimeGenAI package for net10.0
/// - Stable Diffusion ONNX pipeline (text encoder, UNet, VAE decoder, safety checker)
/// 
/// See: https://github.com/elbruno/ElBruno.Text2Image for reference implementation.
/// </summary>
public sealed class OnnxImageGenerationTransformer : IDisposable
{
    private readonly OnnxImageGenerationOptions _options;

    public OnnxImageGenerationTransformer(OnnxImageGenerationOptions options)
    {
        _options = options;
    }

    /// <summary>
    /// Generate an image from a text prompt.
    /// </summary>
    /// <param name="prompt">The text prompt describing the desired image.</param>
    /// <returns>Generated image as a byte array (PNG format).</returns>
    public byte[] Generate(string prompt)
    {
        // TODO: Implement when OnnxRuntimeGenAI is available for net10.0
        // Pipeline: prompt → CLIP text encoder → UNet denoising loop → VAE decoder → image
        throw new NotImplementedException(
            "Text-to-image generation requires Microsoft.ML.OnnxRuntimeGenAI which " +
            "is not yet available for net10.0. This is a placeholder for future implementation.");
    }

    public void Dispose()
    {
        // TODO: Dispose ONNX sessions when implemented
    }
}
