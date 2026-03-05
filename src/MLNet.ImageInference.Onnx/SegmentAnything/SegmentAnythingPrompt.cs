namespace MLNet.ImageInference.Onnx.SegmentAnything;

/// <summary>
/// Represents a prompt for the Segment Anything Model (SAM2).
/// Prompts define which objects to segment in the image using point coordinates and labels.
/// </summary>
public sealed class SegmentAnythingPrompt
{
    /// <summary>
    /// Point coordinates in original image space as (x, y) pairs.
    /// </summary>
    public float[,] PointCoords { get; }

    /// <summary>
    /// Labels for each point: 1 = foreground, 0 = background.
    /// </summary>
    public float[] PointLabels { get; }

    /// <summary>
    /// Optional previous mask input for iterative refinement (256x256).
    /// </summary>
    public float[,]? PreviousMask { get; init; }

    private SegmentAnythingPrompt(float[,] pointCoords, float[] pointLabels)
    {
        if (pointCoords.GetLength(0) != pointLabels.Length)
            throw new ArgumentException("Number of points must match number of labels.");

        PointCoords = pointCoords;
        PointLabels = pointLabels;
    }

    /// <summary>
    /// Create a prompt with a single foreground point.
    /// </summary>
    public static SegmentAnythingPrompt FromPoint(float x, float y, bool isForeground = true)
    {
        var coords = new float[,] { { x, y } };
        var labels = new[] { isForeground ? 1f : 0f };
        return new SegmentAnythingPrompt(coords, labels);
    }

    /// <summary>
    /// Create a prompt with multiple points and their labels.
    /// </summary>
    public static SegmentAnythingPrompt FromPoints(float[,] pointCoords, float[] pointLabels)
    {
        return new SegmentAnythingPrompt(pointCoords, pointLabels);
    }

    /// <summary>
    /// Create a prompt from a bounding box (x1, y1, x2, y2).
    /// SAM encodes boxes as two points: top-left (label=2) and bottom-right (label=3).
    /// </summary>
    public static SegmentAnythingPrompt FromBoundingBox(float x1, float y1, float x2, float y2)
    {
        var coords = new float[,] { { x1, y1 }, { x2, y2 } };
        var labels = new[] { 2f, 3f };
        return new SegmentAnythingPrompt(coords, labels);
    }

    /// <summary>
    /// Number of points in the prompt.
    /// </summary>
    public int NumPoints => PointLabels.Length;
}
