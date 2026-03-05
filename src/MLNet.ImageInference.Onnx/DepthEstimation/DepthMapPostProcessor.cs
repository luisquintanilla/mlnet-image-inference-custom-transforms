using MLNet.Image.Core;

namespace MLNet.ImageInference.Onnx.DepthEstimation;

/// <summary>
/// Post-processor that converts raw depth model output to a normalized DepthMap.
/// DPT/MiDaS models output inverse depth (disparity) — higher values = closer.
/// </summary>
internal static class DepthMapPostProcessor
{
    /// <summary>
    /// Convert raw model output [H*W] or [1*H*W] to a normalized DepthMap.
    /// </summary>
    public static DepthMap Apply(float[] rawOutput, int height, int width,
        int? originalWidth = null, int? originalHeight = null)
    {
        int pixelCount = height * width;
        int offset = rawOutput.Length - pixelCount; // Handle [1,H,W] or [H,W]

        // Find min/max for normalization
        float minVal = float.MaxValue, maxVal = float.MinValue;
        for (int i = 0; i < pixelCount; i++)
        {
            float v = rawOutput[offset + i];
            if (v < minVal) minVal = v;
            if (v > maxVal) maxVal = v;
        }

        float range = maxVal - minVal;
        if (range < 1e-8f) range = 1f; // Avoid division by zero

        // Min-max normalize to [0, 1] — higher raw value (closer) maps to higher normalized value
        var values = new float[pixelCount];
        for (int i = 0; i < pixelCount; i++)
        {
            values[i] = (rawOutput[offset + i] - minVal) / range;
        }

        // Optionally resize to original image dimensions
        if (originalWidth.HasValue && originalHeight.HasValue
            && (originalWidth.Value != width || originalHeight.Value != height))
        {
            values = ResizeBilinear(values, width, height, originalWidth.Value, originalHeight.Value);
            return new DepthMap
            {
                Values = values,
                Width = originalWidth.Value,
                Height = originalHeight.Value,
                MinDepth = minVal,
                MaxDepth = maxVal
            };
        }

        return new DepthMap
        {
            Values = values,
            Width = width,
            Height = height,
            MinDepth = minVal,
            MaxDepth = maxVal
        };
    }

    private static float[] ResizeBilinear(float[] src, int srcW, int srcH, int dstW, int dstH)
    {
        var dst = new float[dstW * dstH];
        for (int y = 0; y < dstH; y++)
        {
            float srcY = (y + 0.5f) * srcH / dstH - 0.5f;
            int y0 = Math.Clamp((int)MathF.Floor(srcY), 0, srcH - 1);
            int y1 = Math.Min(y0 + 1, srcH - 1);
            float yFrac = srcY - MathF.Floor(srcY);

            for (int x = 0; x < dstW; x++)
            {
                float srcX = (x + 0.5f) * srcW / dstW - 0.5f;
                int x0 = Math.Clamp((int)MathF.Floor(srcX), 0, srcW - 1);
                int x1 = Math.Min(x0 + 1, srcW - 1);
                float xFrac = srcX - MathF.Floor(srcX);

                float v00 = src[y0 * srcW + x0];
                float v10 = src[y0 * srcW + x1];
                float v01 = src[y1 * srcW + x0];
                float v11 = src[y1 * srcW + x1];

                dst[y * dstW + x] =
                    v00 * (1 - xFrac) * (1 - yFrac) +
                    v10 * xFrac * (1 - yFrac) +
                    v01 * (1 - xFrac) * yFrac +
                    v11 * xFrac * yFrac;
            }
        }
        return dst;
    }
}
