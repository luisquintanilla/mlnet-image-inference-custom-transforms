using MLNet.Image.Core;

namespace MLNet.ImageInference.Onnx.SegmentAnything;

/// <summary>
/// Options for configuring the SAM2 (Segment Anything Model v2) transformer.
/// </summary>
public sealed class OnnxSegmentAnythingOptions
{
    /// <summary>
    /// Path to the SAM2 encoder ONNX model.
    /// The encoder processes the input image and produces image embeddings.
    /// </summary>
    public required string EncoderModelPath { get; init; }

    /// <summary>
    /// Path to the SAM2 decoder ONNX model.
    /// The decoder takes image embeddings + prompts and produces segmentation masks.
    /// </summary>
    public required string DecoderModelPath { get; init; }

    /// <summary>
    /// Name of the input column containing the image. Default: "Image".
    /// </summary>
    public string InputColumnName { get; init; } = "Image";

    /// <summary>
    /// Name of the output column for the segmentation mask. Default: "SegmentAnythingMask".
    /// </summary>
    public string OutputColumnName { get; init; } = "SegmentAnythingMask";

    /// <summary>
    /// Preprocessing configuration. Default: SAM2 preset (1024x1024, ImageNet normalization).
    /// </summary>
    public PreprocessorConfig PreprocessorConfig { get; init; } = PreprocessorConfig.SAM2;

    /// <summary>
    /// Batch size for encoding multiple images. Default: 1.
    /// Note: SAM2 encoder processes one image at a time (fixed batch=1 in ONNX).
    /// </summary>
    public int BatchSize { get; init; } = 1;

    /// <summary>
    /// Threshold for converting mask logits to binary mask. Default: 0.0.
    /// Pixels with values above this threshold are considered part of the segment.
    /// </summary>
    public float MaskThreshold { get; init; } = 0.0f;
}
