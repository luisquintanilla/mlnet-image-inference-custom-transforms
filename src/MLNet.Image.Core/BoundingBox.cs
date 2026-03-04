namespace MLNet.Image.Core;

/// <summary>
/// Represents a detected object's bounding box with label and confidence score.
/// </summary>
public record BoundingBox
{
    /// <summary>X coordinate of the top-left corner (pixels).</summary>
    public float X { get; init; }

    /// <summary>Y coordinate of the top-left corner (pixels).</summary>
    public float Y { get; init; }

    /// <summary>Width of the bounding box (pixels).</summary>
    public float Width { get; init; }

    /// <summary>Height of the bounding box (pixels).</summary>
    public float Height { get; init; }

    /// <summary>Predicted class label.</summary>
    public string Label { get; init; } = string.Empty;

    /// <summary>Class index in the model's label set.</summary>
    public int ClassId { get; init; }

    /// <summary>Confidence score [0.0, 1.0].</summary>
    public float Score { get; init; }

    public override string ToString() => $"{Label} ({Score:P1}) [{X:F0},{Y:F0} {Width:F0}x{Height:F0}]";
}
