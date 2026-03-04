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
    /// Applies: rescale → per-channel normalize → HWC to CHW reorder.
    ///
    /// This is the preprocessing step that HuggingFace vision models expect,
    /// and that ML.NET's ExtractPixels does not provide (it only does global offset+scale).
    /// </summary>
    /// <param name="image">The input MLImage (already resized to target size).</param>
    /// <param name="config">Preprocessing configuration (mean, std, rescale settings).</param>
    /// <returns>Float array in CHW format [channels, height, width], normalized per-channel.</returns>
    public static float[] Preprocess(MLImage image, PreprocessorConfig config)
    {
        ArgumentNullException.ThrowIfNull(image);
        ArgumentNullException.ThrowIfNull(config);

        int width = image.Width;
        int height = image.Height;
        int channels = 3; // RGB

        // Extract raw pixel bytes (BGRA or RGBA format from MLImage, 4 bytes per pixel)
        ReadOnlySpan<byte> pixels = image.Pixels;
        int bytesPerPixel = pixels.Length / (width * height);

        // Output tensor in CHW format
        float[] tensor = new float[channels * height * width];

        bool isBgra = image.PixelFormat == MLPixelFormat.Bgra32;

        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                int srcIdx = (y * width + x) * bytesPerPixel;

                // Extract RGB values (handle both BGRA and RGBA pixel formats)
                float r, g, b;
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
                int pixelOffset = y * width + x;
                tensor[0 * height * width + pixelOffset] = r; // R channel
                tensor[1 * height * width + pixelOffset] = g; // G channel
                tensor[2 * height * width + pixelOffset] = b; // B channel
            }
        }

        return tensor;
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
