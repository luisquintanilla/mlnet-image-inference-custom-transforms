using MLNet.Image.Core;

namespace MLNet.ImageInference.Onnx.Segmentation;

/// <summary>
/// Static helper for argmax-based segmentation post-processing.
/// Converts raw model logits [1, numClasses, H, W] into a <see cref="SegmentationMask"/>.
/// </summary>
public static class MaskPostProcessor
{
    /// <summary>
    /// Apply argmax across the class dimension to produce a per-pixel class map,
    /// optionally resizing to the original image dimensions via nearest-neighbor interpolation.
    /// </summary>
    /// <param name="rawOutput">Raw model output logits in [1, numClasses, H, W] layout.</param>
    /// <param name="numClasses">Number of classes (C dimension).</param>
    /// <param name="height">Height of the model output (H dimension).</param>
    /// <param name="width">Width of the model output (W dimension).</param>
    /// <param name="originalWidth">Original image width for resize, or null to skip.</param>
    /// <param name="originalHeight">Original image height for resize, or null to skip.</param>
    /// <param name="labels">Optional class label names.</param>
    public static SegmentationMask Apply(
        float[] rawOutput,
        int numClasses,
        int height,
        int width,
        int? originalWidth,
        int? originalHeight,
        string[]? labels)
    {
        // Argmax per pixel across the class dimension
        // Layout: [1, C, H, W] => index = c * height * width + y * width + x
        var classIds = new int[height * width];
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                int bestClass = 0;
                float bestScore = rawOutput[0 * height * width + y * width + x];

                for (int c = 1; c < numClasses; c++)
                {
                    float score = rawOutput[c * height * width + y * width + x];
                    if (score > bestScore)
                    {
                        bestScore = score;
                        bestClass = c;
                    }
                }

                classIds[y * width + x] = bestClass;
            }
        }

        // Nearest-neighbor resize if original dimensions differ
        if (originalWidth.HasValue && originalHeight.HasValue &&
            (originalWidth.Value != width || originalHeight.Value != height))
        {
            int dstW = originalWidth.Value;
            int dstH = originalHeight.Value;
            var resized = new int[dstH * dstW];

            for (int dy = 0; dy < dstH; dy++)
            {
                int srcY = dy * height / dstH;
                for (int dx = 0; dx < dstW; dx++)
                {
                    int srcX = dx * width / dstW;
                    resized[dy * dstW + dx] = classIds[srcY * width + srcX];
                }
            }

            return new SegmentationMask
            {
                ClassIds = resized,
                Width = dstW,
                Height = dstH,
                Labels = labels
            };
        }

        return new SegmentationMask
        {
            ClassIds = classIds,
            Width = width,
            Height = height,
            Labels = labels
        };
    }
}
