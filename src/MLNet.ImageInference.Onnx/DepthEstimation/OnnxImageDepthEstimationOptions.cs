using MLNet.Image.Core;
using MLNet.ImageInference.Onnx.Shared;

namespace MLNet.ImageInference.Onnx.DepthEstimation;

/// <summary>
/// Options for ONNX-based monocular depth estimation.
/// </summary>
public class OnnxImageDepthEstimationOptions : IOnnxImageOptions
{
    /// <summary>Path to the ONNX depth estimation model.</summary>
    public required string ModelPath { get; init; }

    /// <summary>Name of the input column containing MLImage values.</summary>
    public string InputColumnName { get; init; } = "Image";

    /// <summary>Name of the output column for depth map data.</summary>
    public string OutputColumnName { get; init; } = "DepthMap";

    /// <summary>Preprocessing configuration (mean, std, image size).</summary>
    public PreprocessorConfig PreprocessorConfig { get; init; } = PreprocessorConfig.DPT;

    /// <summary>Whether to resize the output depth map to the original image dimensions.</summary>
    public bool ResizeToOriginal { get; init; } = true;

    /// <summary>Batch size for lookahead batching in IDataView cursors.</summary>
    public int BatchSize { get; init; } = 32;
}
