using MLNet.Image.Core;

namespace MLNet.ImageInference.Onnx.Segmentation;

/// <summary>
/// Options for the OnnxImageSegmentation facade estimator.
/// </summary>
public class OnnxImageSegmentationOptions
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
}
