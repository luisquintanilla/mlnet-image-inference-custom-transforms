using MLNet.Image.Core;

namespace MLNet.ImageInference.Onnx.ZeroShot;

/// <summary>
/// Options for zero-shot image classification using CLIP (vision + text encoders).
/// </summary>
public class OnnxZeroShotImageClassificationOptions : Shared.IOnnxImageOptions
{
    /// <summary>Path to the CLIP vision encoder ONNX model.</summary>
    public required string ImageModelPath { get; init; }

    /// <summary>Path to the CLIP text encoder ONNX model.</summary>
    public required string TextModelPath { get; init; }

    /// <summary>Path to the CLIP vocab.json file.</summary>
    public required string VocabPath { get; init; }

    /// <summary>Path to the CLIP merges.txt file.</summary>
    public required string MergesPath { get; init; }

    /// <summary>Name of the input column containing MLImage values.</summary>
    public string InputColumnName { get; init; } = "Image";

    /// <summary>Name of the output column for the predicted label.</summary>
    public string PredictedLabelColumnName { get; init; } = "PredictedLabel";

    /// <summary>Name of the output column for class probabilities.</summary>
    public string ProbabilityColumnName { get; init; } = "Probability";

    /// <summary>The zero-shot candidate class names (e.g. "a photo of a cat", "a photo of a dog").</summary>
    public required string[] CandidateLabels { get; init; }

    /// <summary>Preprocessing configuration. Defaults to CLIP.</summary>
    public PreprocessorConfig PreprocessorConfig { get; init; } = PreprocessorConfig.CLIP;

    /// <summary>Number of top predictions to return. Default: all.</summary>
    public int? TopK { get; init; }

    /// <summary>
    /// Gets or sets the batch size for IDataView cursor lookahead batching.
    /// Higher values reduce the number of ONNX inference calls but use more memory.
    /// Default is 32.
    /// </summary>
    public int BatchSize { get; set; } = 32;
}
