namespace MLNet.ImageInference.Onnx.SegmentAnything;

/// <summary>
/// Result from SAM2 segmentation containing masks and confidence scores.
/// </summary>
public sealed class SegmentAnythingResult
{
    /// <summary>
    /// Binary segmentation masks at original image resolution.
    /// Each mask is a 2D array where values > 0 indicate the segmented region.
    /// </summary>
    public float[][] Masks { get; }

    /// <summary>
    /// IoU (Intersection over Union) prediction for each mask.
    /// Higher values indicate more confident predictions.
    /// </summary>
    public float[] IoUPredictions { get; }

    /// <summary>
    /// Width of each mask (matches original image width).
    /// </summary>
    public int Width { get; }

    /// <summary>
    /// Height of each mask (matches original image height).
    /// </summary>
    public int Height { get; }

    /// <summary>
    /// Number of masks returned.
    /// </summary>
    public int NumMasks => Masks.Length;

    public SegmentAnythingResult(float[][] masks, float[] iouPredictions, int width, int height)
    {
        Masks = masks;
        IoUPredictions = iouPredictions;
        Width = width;
        Height = height;
    }

    /// <summary>
    /// Get the best mask (highest IoU prediction).
    /// </summary>
    public float[] GetBestMask()
    {
        int bestIdx = 0;
        float bestIou = IoUPredictions[0];
        for (int i = 1; i < IoUPredictions.Length; i++)
        {
            if (IoUPredictions[i] > bestIou)
            {
                bestIou = IoUPredictions[i];
                bestIdx = i;
            }
        }
        return Masks[bestIdx];
    }

    /// <summary>
    /// Get the best IoU score.
    /// </summary>
    public float GetBestIoU()
    {
        float best = IoUPredictions[0];
        for (int i = 1; i < IoUPredictions.Length; i++)
            if (IoUPredictions[i] > best) best = IoUPredictions[i];
        return best;
    }
}
