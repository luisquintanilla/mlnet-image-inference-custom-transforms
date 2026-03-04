namespace MLNet.Image.Core;

/// <summary>
/// Represents a semantic segmentation mask where each pixel is assigned a class ID.
/// </summary>
public record SegmentationMask
{
    /// <summary>Per-pixel class IDs in row-major order [height * width].</summary>
    public int[] ClassIds { get; init; } = [];

    /// <summary>Width of the mask in pixels.</summary>
    public int Width { get; init; }

    /// <summary>Height of the mask in pixels.</summary>
    public int Height { get; init; }

    /// <summary>Label names indexed by class ID (optional).</summary>
    public string[]? Labels { get; init; }

    /// <summary>
    /// Get the class ID at a specific pixel coordinate.
    /// </summary>
    public int GetClassAt(int x, int y)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(x);
        ArgumentOutOfRangeException.ThrowIfNegative(y);
        ArgumentOutOfRangeException.ThrowIfGreaterThanOrEqual(x, Width);
        ArgumentOutOfRangeException.ThrowIfGreaterThanOrEqual(y, Height);
        return ClassIds[y * Width + x];
    }

    /// <summary>
    /// Get the label at a specific pixel coordinate, if labels are available.
    /// </summary>
    public string? GetLabelAt(int x, int y)
    {
        int classId = GetClassAt(x, y);
        return Labels is not null && classId >= 0 && classId < Labels.Length
            ? Labels[classId]
            : null;
    }
}
