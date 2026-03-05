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
/// Composes ImagePreprocessingTransformer + OnnxImageScoringTransformer + classification post-processing.
/// </summary>
public sealed class OnnxImageClassificationTransformer : ITransformer, IDisposable
{
    private readonly OnnxImageClassificationOptions _options;
    private readonly ImagePreprocessingTransformer _preprocessor;
    private readonly OnnxImageScoringTransformer _scorer;

    public bool IsRowToRowMapper => true;

    public OnnxImageClassificationTransformer(OnnxImageClassificationOptions options)
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

    internal OnnxImageClassificationTransformer(
        OnnxImageClassificationOptions options,
        ImagePreprocessingTransformer preprocessor,
        OnnxImageScoringTransformer scorer)
    {
        _options = options;
        _preprocessor = preprocessor;
        _scorer = scorer;
    }

    /// <summary>
    /// Classify a single image and return top predictions.
    /// </summary>
    public (string Label, float Probability)[] Classify(MLImage image, CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();

        // Stage 1: Preprocess
        var tensor = _preprocessor.Preprocess(image);

        // Stage 2: Score
        var output = _scorer.Score(tensor);

        // Stage 3: Post-process (softmax + labels)
        return PostProcess(output);
    }

    /// <summary>
    /// Classifies a batch of images. Uses true tensor batching if the model supports dynamic batch,
    /// otherwise loops individual inference calls.
    /// </summary>
    public (string Label, float Probability)[][] ClassifyBatch(IReadOnlyList<MLImage> images, CancellationToken cancellationToken = default)
    {
        if (images == null || images.Count == 0)
            return Array.Empty<(string, float)[]>();

        if (_scorer.IsBatchDynamic)
        {
            return ClassifyBatchDynamic(images);
        }
        else
        {
            var results = new (string Label, float Probability)[images.Count][];
            for (int i = 0; i < images.Count; i++)
            {
                cancellationToken.ThrowIfCancellationRequested();
                results[i] = Classify(images[i], cancellationToken);
            }
            return results;
        }
    }

    private (string Label, float Probability)[][] ClassifyBatchDynamic(IReadOnlyList<MLImage> images)
    {
        int n = images.Count;

        // Stage 1: Batch preprocess
        var batchTensor = _preprocessor.PreprocessBatch(images);

        // Stage 2: Batch score
        var (output, _) = _scorer.ScoreBatch(batchTensor, n);

        int numClasses = output.Length / n;
        var batchResults = new (string Label, float Probability)[n][];

        for (int i = 0; i < n; i++)
        {
            // Stage 3: Post-process each image's output
            var logits = output.AsSpan(i * numClasses, numClasses).ToArray();
            batchResults[i] = PostProcess(logits);
        }

        return batchResults;
    }

    private (string Label, float Probability)[] PostProcess(float[] output)
    {
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
            predictions = predictions[.._options.TopK.Value];

        return predictions;
    }

    internal OnnxImageClassificationOptions Options => _options;
    internal ImagePreprocessingTransformer Preprocessor => _preprocessor;
    internal OnnxImageScoringTransformer Scorer => _scorer;

    /// <summary>
    /// Creates a composed IDataView pipeline: preprocess → score → post-process.
    /// </summary>
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
        _scorer?.Dispose();
    }
}
