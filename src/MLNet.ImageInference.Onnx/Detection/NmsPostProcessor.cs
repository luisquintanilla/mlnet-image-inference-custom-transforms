using MLNet.Image.Core;

namespace MLNet.ImageInference.Onnx.Detection;

/// <summary>
/// Non-Max Suppression post-processor for YOLO-style object detection output.
/// </summary>
public static class NmsPostProcessor
{
    /// <summary>
    /// Transpose YOLO output from [1, numClasses+4, numBoxes] to [numBoxes, numClasses+4].
    /// </summary>
    public static float[] TransposeYoloOutput(float[] rawOutput, int rows, int cols)
    {
        var transposed = new float[cols * rows];
        for (int r = 0; r < rows; r++)
        {
            for (int c = 0; c < cols; c++)
            {
                transposed[c * rows + r] = rawOutput[r * cols + c];
            }
        }
        return transposed;
    }

    /// <summary>
    /// Apply NMS to raw YOLO output and return detected bounding boxes.
    /// </summary>
    /// <param name="rawOutput">Raw model output in [1, numClasses+4, numBoxes] format.</param>
    /// <param name="numClasses">Number of object classes.</param>
    /// <param name="numBoxes">Number of candidate boxes (e.g. 8400).</param>
    /// <param name="confidenceThreshold">Minimum confidence to keep a detection.</param>
    /// <param name="iouThreshold">IoU threshold for suppression.</param>
    /// <param name="labels">Optional class labels.</param>
    /// <returns>Array of detected bounding boxes after NMS.</returns>
    public static BoundingBox[] Apply(
        float[] rawOutput,
        int numClasses,
        int numBoxes,
        float confidenceThreshold,
        float iouThreshold,
        string[]? labels)
    {
        int stride = numClasses + 4;

        // Transpose from [1, stride, numBoxes] to [numBoxes, stride]
        var data = TransposeYoloOutput(rawOutput, stride, numBoxes);

        // Collect candidates per class
        var candidatesPerClass = new Dictionary<int, List<(BoundingBox Box, float Score)>>();

        for (int i = 0; i < numBoxes; i++)
        {
            int offset = i * stride;

            float xCenter = data[offset];
            float yCenter = data[offset + 1];
            float w = data[offset + 2];
            float h = data[offset + 3];

            // Find best class
            int bestClass = -1;
            float bestScore = confidenceThreshold;
            for (int c = 0; c < numClasses; c++)
            {
                float score = data[offset + 4 + c];
                if (score > bestScore)
                {
                    bestScore = score;
                    bestClass = c;
                }
            }

            if (bestClass < 0)
                continue;

            // Convert from center format to top-left corner
            float x = xCenter - w / 2f;
            float y = yCenter - h / 2f;

            string label = labels is not null && bestClass < labels.Length
                ? labels[bestClass]
                : bestClass.ToString();

            var box = new BoundingBox
            {
                X = x,
                Y = y,
                Width = w,
                Height = h,
                Label = label,
                ClassId = bestClass,
                Score = bestScore
            };

            if (!candidatesPerClass.TryGetValue(bestClass, out var list))
            {
                list = [];
                candidatesPerClass[bestClass] = list;
            }
            list.Add((box, bestScore));
        }

        // Apply NMS per class
        var results = new List<BoundingBox>();

        foreach (var (_, candidates) in candidatesPerClass)
        {
            // Sort by confidence descending
            candidates.Sort((a, b) => b.Score.CompareTo(a.Score));

            var kept = new List<BoundingBox>();
            foreach (var (candidate, _) in candidates)
            {
                bool suppressed = false;
                foreach (var existing in kept)
                {
                    if (ComputeIoU(candidate, existing) >= iouThreshold)
                    {
                        suppressed = true;
                        break;
                    }
                }

                if (!suppressed)
                    kept.Add(candidate);
            }

            results.AddRange(kept);
        }

        // Sort final results by score descending
        results.Sort((a, b) => b.Score.CompareTo(a.Score));
        return results.ToArray();
    }

    private static float ComputeIoU(BoundingBox a, BoundingBox b)
    {
        float x1 = Math.Max(a.X, b.X);
        float y1 = Math.Max(a.Y, b.Y);
        float x2 = Math.Min(a.X + a.Width, b.X + b.Width);
        float y2 = Math.Min(a.Y + a.Height, b.Y + b.Height);

        float intersection = Math.Max(0, x2 - x1) * Math.Max(0, y2 - y1);
        float areaA = a.Width * a.Height;
        float areaB = b.Width * b.Height;
        float union = areaA + areaB - intersection;

        return union <= 0 ? 0 : intersection / union;
    }
}
