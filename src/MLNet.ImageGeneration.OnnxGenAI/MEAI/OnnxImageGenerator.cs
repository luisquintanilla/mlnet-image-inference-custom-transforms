using Microsoft.Extensions.AI;
using Microsoft.ML.Data;

namespace MLNet.ImageGeneration.OnnxGenAI.MEAI;

/// <summary>
/// Experimental IImageGenerator-compatible class for text-to-image generation via ONNX.
/// Wraps <see cref="OnnxImageGenerationTransformer"/> with an async API shape.
/// When IImageGenerator stabilizes in MEAI, this class will implement that interface.
/// </summary>
public sealed class OnnxImageGenerator : IDisposable
{
    private readonly OnnxImageGenerationTransformer _transformer;

    public OnnxImageGenerator(OnnxImageGenerationOptions options)
    {
        _transformer = new OnnxImageGenerationTransformer(options);
    }

    /// <summary>
    /// Generate an image from a text prompt using the Stable Diffusion pipeline.
    /// </summary>
    public async Task<MLImage> GenerateAsync(
        string prompt,
        int? seed = null,
        CancellationToken cancellationToken = default)
    {
        return await Task.Run(() => _transformer.Generate(prompt, seed), cancellationToken);
    }

    public void Dispose()
    {
        _transformer?.Dispose();
    }
}
