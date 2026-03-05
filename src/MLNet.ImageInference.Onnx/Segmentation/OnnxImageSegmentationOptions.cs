using MLNet.Image.Core;

namespace MLNet.ImageInference.Onnx.Segmentation;

/// <summary>
/// Options for the OnnxImageSegmentation facade estimator.
/// </summary>
public class OnnxImageSegmentationOptions : Shared.IOnnxImageOptions
{
    /// <summary>Path to the ONNX model file.</summary>
    public required string ModelPath { get; init; }

    /// <summary>Name of the input column containing MLImage values.</summary>
    public string InputColumnName { get; init; } = "Image";

    /// <summary>Name of the output column for the segmentation mask.</summary>
    public string OutputColumnName { get; init; } = "SegmentationMask";

    /// <summary>Class labels. If null, uses integer class indices.</summary>
    public string[]? Labels { get; init; }

    /// <summary>Preprocessing configuration. Defaults to SegFormer.</summary>
    public PreprocessorConfig PreprocessorConfig { get; init; } = PreprocessorConfig.SegFormer;

    /// <summary>Whether to resize the mask back to the original image dimensions.</summary>
    public bool ResizeToOriginal { get; init; } = true;

    /// <summary>
    /// Gets or sets the batch size for IDataView cursor lookahead batching.
    /// Higher values reduce the number of ONNX inference calls but use more memory.
    /// Default is 32.
    /// </summary>
    public int BatchSize { get; set; } = 32;
}
