using Microsoft.Extensions.AI;

namespace MLNet.ImageGeneration.OnnxGenAI.MEAI;

/// <summary>
/// Experimental IImageGenerator implementation for text-to-image generation via ONNX.
/// 
/// STATUS: Stub implementation. IImageGenerator is experimental in MEAI.
/// Full implementation pending:
/// - Microsoft.ML.OnnxRuntimeGenAI availability for net10.0
/// - Stabilization of IImageGenerator interface in Microsoft.Extensions.AI
/// </summary>
public sealed class OnnxImageGenerator : IDisposable
{
    // NOTE: IImageGenerator is experimental and may not be in the current MEAI package.
    // When it stabilizes, this class will implement IImageGenerator.
    // For now, we provide a compatible API shape.

    private readonly OnnxImageGenerationTransformer _transformer;

    public OnnxImageGenerator(OnnxImageGenerationOptions options)
    {
        _transformer = new OnnxImageGenerationTransformer(options);
    }

    /// <summary>
    /// Generate an image from a text prompt.
    /// </summary>
    public Task<byte[]> GenerateAsync(
        string prompt,
        CancellationToken cancellationToken = default)
    {
        // TODO: Implement async generation
        throw new NotImplementedException(
            "Text-to-image generation is not yet implemented. See OnnxImageGenerationTransformer.");
    }

    public void Dispose()
    {
        _transformer?.Dispose();
    }
}
