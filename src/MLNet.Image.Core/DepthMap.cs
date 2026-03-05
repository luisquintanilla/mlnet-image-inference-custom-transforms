namespace MLNet.Image.Core;

/// <summary>
/// Represents a monocular depth estimation result — a per-pixel depth map.
/// Values are normalized to [0, 1] where 0 = farthest, 1 = nearest.
/// </summary>
public record DepthMap
{
    /// <summary>Per-pixel depth values in row-major order [Height × Width].</summary>
    public float[] Values { get; init; } = [];

    /// <summary>Width of the depth map in pixels.</summary>
    public int Width { get; init; }

    /// <summary>Height of the depth map in pixels.</summary>
    public int Height { get; init; }

    /// <summary>Minimum raw depth value before normalization.</summary>
    public float MinDepth { get; init; }

    /// <summary>Maximum raw depth value before normalization.</summary>
    public float MaxDepth { get; init; }

    /// <summary>Get the normalized depth at pixel (x, y). 0 = farthest, 1 = nearest.</summary>
    public float GetDepthAt(int x, int y)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(x);
        ArgumentOutOfRangeException.ThrowIfNegative(y);
        ArgumentOutOfRangeException.ThrowIfGreaterThanOrEqual(x, Width);
        ArgumentOutOfRangeException.ThrowIfGreaterThanOrEqual(y, Height);
        return Values[y * Width + x];
    }
}
