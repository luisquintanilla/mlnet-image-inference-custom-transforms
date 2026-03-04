using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using MLNet.Image.Core;
using MLNet.ImageInference.Onnx.Shared;
using System.Numerics.Tensors;

namespace MLNet.ImageInference.Onnx.Classification;

/// <summary>
/// Transformer that performs image classification: MLImage → preprocessed tensor → ONNX → softmax → label.
/// </summary>
public sealed class OnnxImageClassificationTransformer : ITransformer, IDisposable
{
    private readonly OnnxImageClassificationOptions _options;
    private readonly OnnxSessionPool _sessionPool;
    private readonly ModelMetadataDiscovery.ModelMetadata _metadata;

    public bool IsRowToRowMapper => true;

    public OnnxImageClassificationTransformer(OnnxImageClassificationOptions options)
    {
        _options = options;
        _sessionPool = new OnnxSessionPool(options.ModelPath);
        _metadata = ModelMetadataDiscovery.Discover(_sessionPool.Session);
    }

    /// <summary>
    /// Classify a single image and return top predictions.
    /// </summary>
    public (string Label, float Probability)[] Classify(MLImage image)
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
        var output = results.First().AsEnumerable<float>().ToArray();

        // Apply softmax
        var probabilities = new float[output.Length];
        TensorPrimitives.SoftMax(output, probabilities);

        // Build results
        var predictions = new (string Label, float Probability)[output.Length];
        for (int i = 0; i < output.Length; i++)
        {
            string label = _options.Labels is not null && i < _options.Labels.Length
                ? _options.Labels[i]
                : i.ToString();
            predictions[i] = (label, probabilities[i]);
        }

        // Sort by probability descending
        Array.Sort(predictions, (a, b) => b.Probability.CompareTo(a.Probability));

        // Apply TopK if specified
        if (_options.TopK.HasValue && _options.TopK.Value < predictions.Length)
        {
            predictions = predictions[.._options.TopK.Value];
        }

        return predictions;
    }

    /// <summary>
    /// Classifies a batch of images. Uses true tensor batching if the model supports dynamic batch,
    /// otherwise loops individual inference calls.
    /// </summary>
    public (string Label, float Probability)[][] ClassifyBatch(IReadOnlyList<MLImage> images)
    {
        if (images == null || images.Count == 0)
            return Array.Empty<(string, float)[]>();

        if (_metadata.IsBatchDynamic)
        {
            return ClassifyBatchDynamic(images);
        }
        else
        {
            // Fixed batch model — loop individual calls
            var results = new (string Label, float Probability)[images.Count][];
            for (int i = 0; i < images.Count; i++)
            {
                results[i] = Classify(images[i]);
            }
            return results;
        }
    }

    private (string Label, float Probability)[][] ClassifyBatchDynamic(IReadOnlyList<MLImage> images)
    {
        int n = images.Count;
        int height = _options.PreprocessorConfig.ImageSize.Height;
        int width = _options.PreprocessorConfig.ImageSize.Width;

        // Preprocess all images into a contiguous [N, 3, H, W] float array
        var batchTensor = HuggingFaceImagePreprocessor.PreprocessBatch(images, _options.PreprocessorConfig);

        // Create ONNX input tensor [N, 3, H, W]
        var inputTensor = new DenseTensor<float>(batchTensor, [n, 3, height, width]);
        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor(_metadata.InputNames[0], inputTensor)
        };

        // Run inference once for the entire batch
        using var results = _sessionPool.Session.Run(inputs);
        var output = results.First().AsEnumerable<float>().ToArray();

        int numClasses = output.Length / n;
        var batchResults = new (string Label, float Probability)[n][];

        for (int i = 0; i < n; i++)
        {
            // Extract logits for this image
            var logits = output.AsSpan(i * numClasses, numClasses);
            var probabilities = new float[numClasses];
            TensorPrimitives.SoftMax(logits, probabilities);

            // Build predictions
            var predictions = new (string Label, float Probability)[numClasses];
            for (int j = 0; j < numClasses; j++)
            {
                string label = _options.Labels is not null && j < _options.Labels.Length
                    ? _options.Labels[j]
                    : j.ToString();
                predictions[j] = (label, probabilities[j]);
            }

            // Sort by probability descending
            Array.Sort(predictions, (a, b) => b.Probability.CompareTo(a.Probability));

            // Apply TopK if specified
            if (_options.TopK.HasValue && _options.TopK.Value < predictions.Length)
            {
                predictions = predictions[.._options.TopK.Value];
            }

            batchResults[i] = predictions;
        }

        return batchResults;
    }

    internal OnnxImageClassificationOptions Options => _options;

    public IDataView Transform(IDataView input)
    {
        return new ClassificationDataView(input, this);
    }

    public DataViewSchema GetOutputSchema(DataViewSchema inputSchema)
    {
        var builder = new DataViewSchema.Builder();
        for (int i = 0; i < inputSchema.Count; i++)
            builder.AddColumn(inputSchema[i].Name, inputSchema[i].Type, inputSchema[i].Annotations);
        builder.AddColumn(_options.PredictedLabelColumnName, TextDataViewType.Instance);
        builder.AddColumn(_options.ProbabilityColumnName, new VectorDataViewType(NumberDataViewType.Single));
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
