using Microsoft.ML.Data;

namespace MLNet.Image.Core;

/// <summary>
/// HuggingFace-compatible image preprocessor that provides per-channel
/// mean/std normalization — the key gap in ML.NET's built-in ExtractPixels
/// (which only supports global offset+scale, not per-channel normalization).
///
/// Pipeline: MLImage → extract pixels → rescale to [0,1] → normalize per-channel → CHW float tensor
/// </summary>
public static class HuggingFaceImagePreprocessor
{
    /// <summary>
    /// Preprocess an MLImage into a normalized float tensor in CHW format.
    /// Applies: resize → rescale → per-channel normalize → HWC to CHW reorder.
    ///
    /// This is the preprocessing step that HuggingFace vision models expect,
    /// and that ML.NET's ExtractPixels does not provide (it only does global offset+scale).
    /// </summary>
    /// <param name="image">The input MLImage (any size — will be resized to config.ImageSize).</param>
    /// <param name="config">Preprocessing configuration (mean, std, rescale settings).</param>
    /// <returns>Float array in CHW format [channels, height, width], normalized per-channel.</returns>
    public static float[] Preprocess(MLImage image, PreprocessorConfig config)
    {
        ArgumentNullException.ThrowIfNull(image);
        ArgumentNullException.ThrowIfNull(config);

        int srcWidth = image.Width;
        int srcHeight = image.Height;
        int targetHeight = config.ImageSize.Height;
        int targetWidth = config.ImageSize.Width;
        int channels = 3; // RGB
        bool needsResize = srcWidth != targetWidth || srcHeight != targetHeight;

        // Extract raw pixel bytes (BGRA or RGBA format from MLImage, 4 bytes per pixel)
        ReadOnlySpan<byte> pixels = image.Pixels;
        int bytesPerPixel = pixels.Length / (srcWidth * srcHeight);

        // Output tensor in CHW format at the target resolution
        float[] tensor = new float[channels * targetHeight * targetWidth];

        bool isBgra = image.PixelFormat == MLPixelFormat.Bgra32;

        for (int y = 0; y < targetHeight; y++)
        {
            for (int x = 0; x < targetWidth; x++)
            {
                // Map target coordinates back to source (bilinear interpolation)
                float r, g, b;
                if (needsResize)
                {
                    float srcX = (x + 0.5f) * srcWidth / targetWidth - 0.5f;
                    float srcY = (y + 0.5f) * srcHeight / targetHeight - 0.5f;
                    (r, g, b) = SampleBilinear(pixels, srcWidth, srcHeight, bytesPerPixel, isBgra, srcX, srcY);
                }
                else
                {
                    int srcIdx = (y * srcWidth + x) * bytesPerPixel;
                    if (isBgra)
                    {
                        b = pixels[srcIdx];
                        g = pixels[srcIdx + 1];
                        r = pixels[srcIdx + 2];
                    }
                    else
                    {
                        r = pixels[srcIdx];
                        g = pixels[srcIdx + 1];
                        b = pixels[srcIdx + 2];
                    }
                }

                // Rescale from [0, 255] to [0.0, 1.0]
                if (config.DoRescale)
                {
                    r *= config.RescaleFactor;
                    g *= config.RescaleFactor;
                    b *= config.RescaleFactor;
                }

                // Per-channel normalization: (value - mean) / std
                if (config.DoNormalize)
                {
                    r = (r - config.Mean[0]) / config.Std[0];
                    g = (g - config.Mean[1]) / config.Std[1];
                    b = (b - config.Mean[2]) / config.Std[2];
                }

                // Write in CHW format: [channel][height][width]
                int pixelOffset = y * targetWidth + x;
                tensor[0 * targetHeight * targetWidth + pixelOffset] = r; // R channel
                tensor[1 * targetHeight * targetWidth + pixelOffset] = g; // G channel
                tensor[2 * targetHeight * targetWidth + pixelOffset] = b; // B channel
            }
        }

        return tensor;
    }

    private static (float R, float G, float B) SampleBilinear(
        ReadOnlySpan<byte> pixels, int width, int height, int bpp, bool isBgra, float x, float y)
    {
        int x0 = Math.Clamp((int)MathF.Floor(x), 0, width - 1);
        int y0 = Math.Clamp((int)MathF.Floor(y), 0, height - 1);
        int x1 = Math.Min(x0 + 1, width - 1);
        int y1 = Math.Min(y0 + 1, height - 1);
        float xFrac = x - MathF.Floor(x);
        float yFrac = y - MathF.Floor(y);

        static (float r, float g, float b) ReadPixel(ReadOnlySpan<byte> p, int idx, bool bgra)
        {
            if (bgra) return (p[idx + 2], p[idx + 1], p[idx]);
            return (p[idx], p[idx + 1], p[idx + 2]);
        }

        var (r00, g00, b00) = ReadPixel(pixels, (y0 * width + x0) * bpp, isBgra);
        var (r10, g10, b10) = ReadPixel(pixels, (y0 * width + x1) * bpp, isBgra);
        var (r01, g01, b01) = ReadPixel(pixels, (y1 * width + x0) * bpp, isBgra);
        var (r11, g11, b11) = ReadPixel(pixels, (y1 * width + x1) * bpp, isBgra);

        float w00 = (1 - xFrac) * (1 - yFrac);
        float w10 = xFrac * (1 - yFrac);
        float w01 = (1 - xFrac) * yFrac;
        float w11 = xFrac * yFrac;

        return (
            r00 * w00 + r10 * w10 + r01 * w01 + r11 * w11,
            g00 * w00 + g10 * w10 + g01 * w01 + g11 * w11,
            b00 * w00 + b10 * w10 + b01 * w01 + b11 * w11
        );
    }

    /// <summary>
    /// Preprocesses a batch of images into a single contiguous tensor of shape [N, C, H, W].
    /// </summary>
    /// <param name="images">The input MLImages (already resized to target size).</param>
    /// <param name="config">Preprocessing configuration.</param>
    /// <returns>Float array of shape [N, 3, H, W] containing all images in CHW format.</returns>
    public static float[] PreprocessBatch(IReadOnlyList<MLImage> images, PreprocessorConfig config)
    {
        if (images == null || images.Count == 0)
            throw new ArgumentException("Images collection must not be null or empty.", nameof(images));

        int channels = 3;
        int height = config.CropSize.Height;
        int width = config.CropSize.Width;
        int singleImageSize = channels * height * width;
        var result = new float[images.Count * singleImageSize];

        for (int i = 0; i < images.Count; i++)
        {
            var singleResult = Preprocess(images[i], config);
            Array.Copy(singleResult, 0, result, i * singleImageSize, singleImageSize);
        }

        return result;
    }
}
