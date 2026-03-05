using MLNet.Image.Core;

namespace MLNet.ImageInference.Onnx.Embeddings;

/// <summary>
/// Options for the OnnxImageEmbedding facade estimator.
/// </summary>
public class OnnxImageEmbeddingOptions : Shared.IOnnxImageOptions
{
    /// <summary>Path to the ONNX model file.</summary>
    public required string ModelPath { get; init; }

    /// <summary>Name of the input column containing MLImage values.</summary>
    public string InputColumnName { get; init; } = "Image";

    /// <summary>Name of the output column for the embedding vector.</summary>
    public string EmbeddingColumnName { get; init; } = "Embedding";

    /// <summary>Preprocessing configuration. Defaults to CLIP.</summary>
    public PreprocessorConfig PreprocessorConfig { get; init; } = PreprocessorConfig.CLIP;

    /// <summary>Whether to L2-normalize the output embedding. Default: true.</summary>
    public bool Normalize { get; init; } = true;

    /// <summary>Pooling strategy for the model output. Default: CLS token.</summary>
    public PoolingStrategy Pooling { get; init; } = PoolingStrategy.ClsToken;

    /// <summary>
    /// Gets or sets the batch size for IDataView cursor lookahead batching.
    /// Higher values reduce the number of ONNX inference calls but use more memory.
    /// Default is 32.
    /// </summary>
    public int BatchSize { get; set; } = 32;
}

/// <summary>
/// Pooling strategy for extracting a fixed-size embedding from model output.
/// </summary>
public enum PoolingStrategy
{
    /// <summary>Use the [CLS] token output (first token). Used by ViT, CLIP.</summary>
    ClsToken,

    /// <summary>Mean pooling across all token outputs. Used by DINOv2.</summary>
    MeanPooling
}
