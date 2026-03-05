using Microsoft.ML;
using Microsoft.ML.Data;
using MLNet.Image.Core;
using MLNet.ImageInference.Onnx.Shared;

namespace MLNet.ImageInference.Onnx.Detection;

/// <summary>
/// Transformer that performs object detection: MLImage → preprocessed tensor → ONNX → NMS → BoundingBox[].
/// Composes ImagePreprocessingTransformer + OnnxImageScoringTransformer + NMS post-processing.
/// </summary>
public sealed class OnnxObjectDetectionTransformer : ITransformer, IDisposable
{
    private readonly OnnxObjectDetectionOptions _options;
    private readonly ImagePreprocessingTransformer _preprocessor;
    private readonly OnnxImageScoringTransformer _scorer;

    public bool IsRowToRowMapper => true;

    public OnnxObjectDetectionTransformer(OnnxObjectDetectionOptions options)
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

    internal OnnxObjectDetectionTransformer(
        OnnxObjectDetectionOptions options,
        ImagePreprocessingTransformer preprocessor,
        OnnxImageScoringTransformer scorer)
    {
        _options = options;
        _preprocessor = preprocessor;
        _scorer = scorer;
    }

    /// <summary>
    /// Detect objects in a single image and return bounding boxes.
    /// </summary>
    public BoundingBox[] Detect(MLImage image, CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();

        // Stage 1: Preprocess
        var tensor = _preprocessor.Preprocess(image);

        // Stage 2: Score
        var output = _scorer.Score(tensor);

        // Stage 3: Post-process (NMS)
        return PostProcess(output);
    }

    /// <summary>
    /// Detects objects in a batch of images. Uses true tensor batching if the model supports dynamic batch,
    /// otherwise loops individual inference calls.
    /// </summary>
    public BoundingBox[][] DetectBatch(IReadOnlyList<MLImage> images, CancellationToken cancellationToken = default)
    {
        if (images == null || images.Count == 0)
            return Array.Empty<BoundingBox[]>();

        if (_scorer.IsBatchDynamic)
        {
            return DetectBatchDynamic(images);
        }
        else
        {
            var results = new BoundingBox[images.Count][];
            for (int i = 0; i < images.Count; i++)
            {
                cancellationToken.ThrowIfCancellationRequested();
                results[i] = Detect(images[i], cancellationToken);
            }
            return results;
        }
    }

    private BoundingBox[][] DetectBatchDynamic(IReadOnlyList<MLImage> images)
    {
        int n = images.Count;

        // Stage 1: Batch preprocess
        var batchTensor = _preprocessor.PreprocessBatch(images);

        // Stage 2: Batch score
        var (output, _) = _scorer.ScoreBatch(batchTensor, n);

        int outputPerImage = output.Length / n;
        var batchResults = new BoundingBox[n][];

        for (int i = 0; i < n; i++)
        {
            // Stage 3: Post-process each image's output
            var imageOutput = output.AsSpan(i * outputPerImage, outputPerImage).ToArray();
            batchResults[i] = PostProcess(imageOutput);
        }

        return batchResults;
    }

    private BoundingBox[] PostProcess(float[] output)
    {
        // Determine dimensions from model output shape: [1, numClasses+4, numBoxes]
        var outputShape = _scorer.Metadata.OutputShapes[0];
        int numClasses = (int)outputShape[1] - 4;
        int numBoxes = (int)outputShape[2];

        // Apply NMS post-processing
        var detections = NmsPostProcessor.Apply(
            output,
            numClasses,
            numBoxes,
            _options.ConfidenceThreshold,
            _options.IouThreshold,
            _options.Labels);

        // Apply MaxDetections if specified
        if (_options.MaxDetections.HasValue && _options.MaxDetections.Value < detections.Length)
        {
            detections = detections[.._options.MaxDetections.Value];
        }

        return detections;
    }

    internal OnnxObjectDetectionOptions Options => _options;
    internal ImagePreprocessingTransformer Preprocessor => _preprocessor;
    internal OnnxImageScoringTransformer Scorer => _scorer;

    public IDataView Transform(IDataView input)
    {
        return new DetectionDataView(input, this);
    }

    public DataViewSchema GetOutputSchema(DataViewSchema inputSchema)
    {
        var builder = new DataViewSchema.Builder();
        for (int i = 0; i < inputSchema.Count; i++)
            builder.AddColumn(inputSchema[i].Name, inputSchema[i].Type, inputSchema[i].Annotations);
        builder.AddColumn(_options.OutputColumnName + "_Boxes", new VectorDataViewType(NumberDataViewType.Single));
        builder.AddColumn(_options.OutputColumnName + "_Count", NumberDataViewType.Int32);
        return builder.ToSchema();
    }

    public IRowToRowMapper GetRowToRowMapper(DataViewSchema inputSchema)
        => throw new InvalidOperationException("This transformer does not support row-to-row mapping.");

    void ICanSaveModel.Save(ModelSaveContext ctx)
        => throw new NotSupportedException("Use transformer-specific save/load instead of mlContext.Model.Save().");

    public void Dispose()
    {
        _scorer?.Dispose();
    }
}
