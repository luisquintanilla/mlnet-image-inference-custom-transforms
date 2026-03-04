using MLNet.Image.Core;

namespace MLNet.ImageInference.Onnx.Detection;

/// <summary>
/// Options for the OnnxObjectDetection facade estimator.
/// </summary>
public class OnnxObjectDetectionOptions
{
    /// <summary>Path to the ONNX model file.</summary>
    public required string ModelPath { get; init; }

    /// <summary>Name of the input column containing MLImage values.</summary>
    public string InputColumnName { get; init; } = "Image";

    /// <summary>Name of the output column for detected objects.</summary>
    public string OutputColumnName { get; init; } = "DetectedObjects";

    /// <summary>Class labels. If null, uses integer class indices.</summary>
    public string[]? Labels { get; init; }

    /// <summary>Preprocessing configuration. Defaults to YOLOv8.</summary>
    public PreprocessorConfig PreprocessorConfig { get; init; } = PreprocessorConfig.YOLOv8;

    /// <summary>Minimum confidence score to keep a detection.</summary>
    public float ConfidenceThreshold { get; init; } = 0.5f;

    /// <summary>IoU threshold for Non-Max Suppression.</summary>
    public float IouThreshold { get; init; } = 0.45f;

    /// <summary>Maximum number of detections to return. Null means all.</summary>
    public int? MaxDetections { get; init; }
}
