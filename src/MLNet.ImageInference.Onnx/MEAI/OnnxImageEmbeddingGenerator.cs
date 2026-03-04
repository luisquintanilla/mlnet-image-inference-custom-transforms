using Microsoft.Extensions.AI;
using Microsoft.ML.Data;

namespace MLNet.ImageInference.Onnx.MEAI;

/// <summary>
/// IEmbeddingGenerator implementation for image embeddings via ONNX.
/// Bridges the ML.NET image embedding transformer to the Microsoft.Extensions.AI abstraction.
/// Uses MLImage directly as TInput — the ML.NET image primitive.
/// </summary>
public sealed class OnnxImageEmbeddingGenerator : IEmbeddingGenerator<MLImage, Embedding<float>>
{
    private readonly Embeddings.OnnxImageEmbeddingTransformer _transformer;

    public EmbeddingGeneratorMetadata Metadata { get; }

    public OnnxImageEmbeddingGenerator(Embeddings.OnnxImageEmbeddingTransformer transformer, string? modelId = null)
    {
        ArgumentNullException.ThrowIfNull(transformer);
        _transformer = transformer;

        Metadata = new EmbeddingGeneratorMetadata(
            providerName: "MLNet.ImageInference.Onnx",
            defaultModelId: modelId,
            defaultModelDimensions: transformer.EmbeddingDimension);
    }

    /// <summary>
    /// Create an OnnxImageEmbeddingGenerator from a model path.
    /// </summary>
    public OnnxImageEmbeddingGenerator(string modelPath, Embeddings.OnnxImageEmbeddingOptions? options = null)
        : this(new Embeddings.OnnxImageEmbeddingTransformer(
            options ?? new Embeddings.OnnxImageEmbeddingOptions { ModelPath = modelPath }),
            modelId: Path.GetFileNameWithoutExtension(modelPath))
    {
    }

    public Task<GeneratedEmbeddings<Embedding<float>>> GenerateAsync(
        IEnumerable<MLImage> values,
        EmbeddingGenerationOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        var imageList = values.ToList();
        var embeddings = _transformer.GenerateEmbeddings(imageList);

        var result = new GeneratedEmbeddings<Embedding<float>>(
            embeddings.Select(e => new Embedding<float>(e)).ToList());

        return Task.FromResult(result);
    }

    public object? GetService(Type serviceType, object? serviceKey = null)
    {
        if (serviceType == typeof(OnnxImageEmbeddingGenerator))
            return this;
        return null;
    }

    public void Dispose()
    {
        _transformer?.Dispose();
    }
}
