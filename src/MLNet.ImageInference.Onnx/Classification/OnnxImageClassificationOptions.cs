using MLNet.Image.Core;

namespace MLNet.ImageInference.Onnx.Classification;

/// <summary>
/// Options for the OnnxImageClassification facade estimator.
/// </summary>
public class OnnxImageClassificationOptions : Shared.IOnnxImageOptions
{
    /// <summary>Path to the ONNX model file.</summary>
    public required string ModelPath { get; init; }

    /// <summary>Name of the input column containing MLImage values.</summary>
    public string InputColumnName { get; init; } = "Image";

    /// <summary>Name of the output column for the predicted label.</summary>
    public string PredictedLabelColumnName { get; init; } = "PredictedLabel";

    /// <summary>Name of the output column for class probabilities.</summary>
    public string ProbabilityColumnName { get; init; } = "Probability";

    /// <summary>Class labels. If null, uses integer class indices.</summary>
    public string[]? Labels { get; init; }

    /// <summary>Preprocessing configuration. Defaults to ImageNet.</summary>
    public PreprocessorConfig PreprocessorConfig { get; init; } = PreprocessorConfig.ImageNet;

    /// <summary>Number of top predictions to return. Default: all.</summary>
    public int? TopK { get; init; }

    /// <summary>
    /// Gets or sets the batch size for IDataView cursor lookahead batching.
    /// Higher values reduce the number of ONNX inference calls but use more memory.
    /// Default is 32.
    /// </summary>
    public int BatchSize { get; set; } = 32;
}
