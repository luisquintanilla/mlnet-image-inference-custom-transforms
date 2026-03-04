using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using MLNet.Image.Core;
using MLNet.ImageInference.Onnx.Shared;
using System.Numerics.Tensors;

namespace MLNet.ImageInference.Onnx.Embeddings;

/// <summary>
/// Transformer that produces image embeddings: MLImage → preprocessed tensor → ONNX → pooling → float[] vector.
/// </summary>
public sealed class OnnxImageEmbeddingTransformer : ITransformer, IDisposable
{
    private readonly OnnxImageEmbeddingOptions _options;
    private readonly OnnxSessionPool _sessionPool;
    private readonly ModelMetadataDiscovery.ModelMetadata _metadata;

    public bool IsRowToRowMapper => true;

    /// <summary>
    /// Dimension of the output embedding vector.
    /// </summary>
    public int EmbeddingDimension { get; }

    public OnnxImageEmbeddingTransformer(OnnxImageEmbeddingOptions options)
    {
        _options = options;
        _sessionPool = new OnnxSessionPool(options.ModelPath);
        _metadata = ModelMetadataDiscovery.Discover(_sessionPool.Session);

        // Discover embedding dimension from output shape
        var outputShape = _metadata.OutputShapes[0];
        EmbeddingDimension = (int)outputShape[^1]; // Last dimension is embedding size
    }

    /// <summary>
    /// Generate embeddings for a list of images.
    /// </summary>
    public float[][] GenerateEmbeddings(IReadOnlyList<MLImage> images)
    {
        var results = new float[images.Count][];
        for (int i = 0; i < images.Count; i++)
        {
            results[i] = GenerateEmbedding(images[i]);
        }
        return results;
    }

    /// <summary>
    /// Generates embeddings for a batch of images. Uses true tensor batching if the model supports it.
    /// </summary>
    public float[][] GenerateEmbeddingBatch(IReadOnlyList<MLImage> images)
    {
        if (images == null || images.Count == 0)
            return Array.Empty<float[]>();

        if (_metadata.IsBatchDynamic)
        {
            return GenerateEmbeddingBatchDynamic(images);
        }
        else
        {
            var results = new float[images.Count][];
            for (int i = 0; i < images.Count; i++)
            {
                results[i] = GenerateEmbedding(images[i]);
            }
            return results;
        }
    }

    /// <summary>
    /// Generate an embedding for a single image.
    /// </summary>
    public float[] GenerateEmbedding(MLImage image)
    {
        var tensor = HuggingFaceImagePreprocessor.Preprocess(image, _options.PreprocessorConfig);
        int height = _options.PreprocessorConfig.ImageSize.Height;
        int width = _options.PreprocessorConfig.ImageSize.Width;

        // Create ONNX input tensor [1, 3, H, W]
        var inputTensor = new DenseTensor<float>(tensor, [1, 3, height, width]);
        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor(_metadata.InputNames[0], inputTensor)
        };

        // Run inference
        using var results = _sessionPool.Session.Run(inputs);
        var output = results.First().AsTensor<float>();

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

    private float[][] GenerateEmbeddingBatchDynamic(IReadOnlyList<MLImage> images)
    {
        int n = images.Count;
        int height = _options.PreprocessorConfig.ImageSize.Height;
        int width = _options.PreprocessorConfig.ImageSize.Width;

        var batchData = HuggingFaceImagePreprocessor.PreprocessBatch(images, _options.PreprocessorConfig);
        var inputTensor = new DenseTensor<float>(batchData, [n, 3, height, width]);
        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor(_metadata.InputNames[0], inputTensor)
        };

        using var results = _sessionPool.Session.Run(inputs);
        var output = results.First().AsTensor<float>();

        var embeddings = new float[n][];
        for (int i = 0; i < n; i++)
        {
            float[] embedding = _options.Pooling switch
            {
                PoolingStrategy.ClsToken => ExtractClsTokenAtIndex(output, i),
                PoolingStrategy.MeanPooling => MeanPoolAtIndex(output, i),
                _ => ExtractClsTokenAtIndex(output, i)
            };

            if (_options.Normalize)
            {
                float norm = TensorPrimitives.Norm(embedding);
                if (norm > 0)
                {
                    TensorPrimitives.Divide(embedding, norm, embedding);
                }
            }

            embeddings[i] = embedding;
        }

        return embeddings;
    }

    private float[] ExtractClsTokenAtIndex(Microsoft.ML.OnnxRuntime.Tensors.Tensor<float> output, int batchIndex)
    {
        int dims = output.Dimensions.Length;
        if (dims == 2)
        {
            // [N, hidden_dim]
            var result = new float[EmbeddingDimension];
            for (int i = 0; i < EmbeddingDimension; i++)
            {
                result[i] = output[batchIndex, i];
            }
            return result;
        }
        else
        {
            // [N, seq_len, hidden_dim] — take first token (CLS)
            var result = new float[EmbeddingDimension];
            for (int i = 0; i < EmbeddingDimension; i++)
            {
                result[i] = output[batchIndex, 0, i];
            }
            return result;
        }
    }

    private float[] MeanPoolAtIndex(Microsoft.ML.OnnxRuntime.Tensors.Tensor<float> output, int batchIndex)
    {
        int seqLen = output.Dimensions.Length > 2 ? (int)output.Dimensions[1] : 1;
        var result = new float[EmbeddingDimension];

        for (int i = 0; i < EmbeddingDimension; i++)
        {
            float sum = 0;
            for (int s = 0; s < seqLen; s++)
            {
                sum += output[batchIndex, s, i];
            }
            result[i] = sum / seqLen;
        }

        return result;
    }

    private float[] ExtractClsToken(Microsoft.ML.OnnxRuntime.Tensors.Tensor<float> output)
    {
        // Output shape is typically [1, seq_len, hidden_dim] or [1, hidden_dim]
        int dims = output.Dimensions.Length;
        if (dims == 2)
        {
            // [1, hidden_dim] — already a single vector
            return output.ToArray()[..EmbeddingDimension];
        }
        else
        {
            // [1, seq_len, hidden_dim] — take first token (CLS)
            var result = new float[EmbeddingDimension];
            for (int i = 0; i < EmbeddingDimension; i++)
            {
                result[i] = output[0, 0, i];
            }
            return result;
        }
    }

    private float[] MeanPool(Microsoft.ML.OnnxRuntime.Tensors.Tensor<float> output)
    {
        // Output shape: [1, seq_len, hidden_dim]
        int seqLen = output.Dimensions.Length > 2 ? (int)output.Dimensions[1] : 1;
        var result = new float[EmbeddingDimension];

        for (int i = 0; i < EmbeddingDimension; i++)
        {
            float sum = 0;
            for (int s = 0; s < seqLen; s++)
            {
                sum += output[0, s, i];
            }
            result[i] = sum / seqLen;
        }

        return result;
    }

    internal OnnxImageEmbeddingOptions Options => _options;

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
        _sessionPool?.Dispose();
    }
}
