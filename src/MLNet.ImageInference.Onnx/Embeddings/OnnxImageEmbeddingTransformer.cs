using Microsoft.ML;
using Microsoft.ML.Data;
using MLNet.Image.Core;
using MLNet.ImageInference.Onnx.Shared;
using System.Numerics.Tensors;

namespace MLNet.ImageInference.Onnx.Embeddings;

/// <summary>
/// Transformer that produces image embeddings: MLImage → preprocessed tensor → ONNX → pooling → float[] vector.
/// Composes ImagePreprocessingTransformer + OnnxImageScoringTransformer + pooling/normalization post-processing.
/// </summary>
public sealed class OnnxImageEmbeddingTransformer : ITransformer, IDisposable
{
    private readonly OnnxImageEmbeddingOptions _options;
    private readonly ImagePreprocessingTransformer _preprocessor;
    private readonly OnnxImageScoringTransformer _scorer;

    public bool IsRowToRowMapper => true;

    /// <summary>
    /// Dimension of the output embedding vector.
    /// </summary>
    public int EmbeddingDimension { get; }

    public OnnxImageEmbeddingTransformer(OnnxImageEmbeddingOptions options)
        : this(options,
              new ImagePreprocessingTransformer(new ImagePreprocessingOptions
              {
                  InputColumnName = options.InputColumnName,
                  PreprocessorConfig = options.PreprocessorConfig
              }),
              new OnnxImageScoringTransformer(new OnnxImageScoringOptions
              {
                  ModelPath = options.ModelPath,
                  ImageHeight = options.PreprocessorConfig.ImageSize.Height,
                  ImageWidth = options.PreprocessorConfig.ImageSize.Width,
                  BatchSize = options.BatchSize
              }))
    {
    }

    internal OnnxImageEmbeddingTransformer(
        OnnxImageEmbeddingOptions options,
        ImagePreprocessingTransformer preprocessor,
        OnnxImageScoringTransformer scorer)
    {
        _options = options;
        _preprocessor = preprocessor;
        _scorer = scorer;

        // Discover embedding dimension from output shape
        EmbeddingDimension = (int)scorer.Metadata.OutputShapes[0][^1];
    }

    /// <summary>
    /// Generate embeddings for a list of images.
    /// </summary>
    public float[][] GenerateEmbeddings(IReadOnlyList<MLImage> images, CancellationToken cancellationToken = default)
    {
        var results = new float[images.Count][];
        for (int i = 0; i < images.Count; i++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            results[i] = GenerateEmbedding(images[i], cancellationToken);
        }
        return results;
    }

    /// <summary>
    /// Generates embeddings for a batch of images. Uses true tensor batching if the model supports it.
    /// </summary>
    public float[][] GenerateEmbeddingBatch(IReadOnlyList<MLImage> images, CancellationToken cancellationToken = default)
    {
        if (images == null || images.Count == 0)
            return Array.Empty<float[]>();

        if (_scorer.IsBatchDynamic)
        {
            return GenerateEmbeddingBatchDynamic(images);
        }
        else
        {
            var results = new float[images.Count][];
            for (int i = 0; i < images.Count; i++)
            {
                cancellationToken.ThrowIfCancellationRequested();
                results[i] = GenerateEmbedding(images[i], cancellationToken);
            }
            return results;
        }
    }

    /// <summary>
    /// Generate an embedding for a single image.
    /// </summary>
    public float[] GenerateEmbedding(MLImage image, CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();

        // Stage 1: Preprocess
        var tensor = _preprocessor.Preprocess(image);

        // Stage 2: Score
        var output = _scorer.Score(tensor);

        // Stage 3: Post-process (pool + normalize)
        return PostProcess(output);
    }

    private float[][] GenerateEmbeddingBatchDynamic(IReadOnlyList<MLImage> images)
    {
        int n = images.Count;

        // Stage 1: Batch preprocess
        var batchData = _preprocessor.PreprocessBatch(images);

        // Stage 2: Batch score
        var (output, _) = _scorer.ScoreBatch(batchData, n);

        int outputPerImage = output.Length / n;
        var embeddings = new float[n][];

        for (int i = 0; i < n; i++)
        {
            // Stage 3: Post-process each image's output
            var imageOutput = output.AsSpan(i * outputPerImage, outputPerImage).ToArray();
            embeddings[i] = PostProcess(imageOutput);
        }

        return embeddings;
    }

    private float[] PostProcess(float[] output)
    {
        // Pool the output
        float[] embedding = _options.Pooling switch
        {
            PoolingStrategy.ClsToken => ExtractClsToken(output),
            PoolingStrategy.MeanPooling => MeanPool(output),
            _ => ExtractClsToken(output)
        };

        // L2 normalize
        if (_options.Normalize)
        {
            float norm = TensorPrimitives.Norm(embedding);
            if (norm > 0)
            {
                TensorPrimitives.Divide(embedding, norm, embedding);
            }
        }

        return embedding;
    }

    private float[] ExtractClsToken(float[] output)
    {
        // CLS token is always the first EmbeddingDimension elements in the flat array.
        // Works for both [1, hidden_dim] and [1, seq_len, hidden_dim] layouts.
        return output[..EmbeddingDimension];
    }

    private float[] MeanPool(float[] output)
    {
        // Output shape: [1, seq_len, hidden_dim] or [1, hidden_dim]
        int seqLen = output.Length / EmbeddingDimension;
        var result = new float[EmbeddingDimension];

        for (int i = 0; i < EmbeddingDimension; i++)
        {
            float sum = 0;
            for (int s = 0; s < seqLen; s++)
            {
                sum += output[s * EmbeddingDimension + i];
            }
            result[i] = sum / seqLen;
        }

        return result;
    }

    internal OnnxImageEmbeddingOptions Options => _options;
    internal ImagePreprocessingTransformer Preprocessor => _preprocessor;
    internal OnnxImageScoringTransformer Scorer => _scorer;

    public IDataView Transform(IDataView input)
    {
        return new EmbeddingDataView(input, this);
    }

    public DataViewSchema GetOutputSchema(DataViewSchema inputSchema)
    {
        var builder = new DataViewSchema.Builder();
        for (int i = 0; i < inputSchema.Count; i++)
            builder.AddColumn(inputSchema[i].Name, inputSchema[i].Type, inputSchema[i].Annotations);
        builder.AddColumn(_options.EmbeddingColumnName,
            new VectorDataViewType(NumberDataViewType.Single, EmbeddingDimension));
        return builder.ToSchema();
    }

    public IRowToRowMapper GetRowToRowMapper(DataViewSchema inputSchema)
        => throw new InvalidOperationException(
            "Use Transform() to get an IDataView. Direct IRowToRowMapper is not supported.");

    void ICanSaveModel.Save(ModelSaveContext ctx)
        => throw new NotSupportedException("Use transformer-specific save/load instead of mlContext.Model.Save().");

    public void Dispose()
    {
        _scorer?.Dispose();
    }
}
