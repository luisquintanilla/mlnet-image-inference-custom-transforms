using Microsoft.Extensions.AI;
using Microsoft.ML.Data;

namespace MLNet.Image.Core;

/// <summary>
/// Extension methods for converting between MLImage (ML.NET) and DataContent (MEAI).
/// Bridges the ML.NET world (which uses MLImage) with the Microsoft.Extensions.AI world
/// (which uses DataContent with MIME types for image data).
/// </summary>
public static class MLImageExtensions
{
    /// <summary>
    /// Convert an MLImage to a MEAI DataContent with encoded image bytes.
    /// Use this to pass ML.NET images to MEAI interfaces like IChatClient or IImageGenerator.
    /// </summary>
    /// <param name="image">The MLImage to convert.</param>
    /// <param name="mediaType">The media type for encoding (default: "image/png").</param>
    /// <returns>A DataContent containing the encoded image bytes and MIME type.</returns>
    public static DataContent ToDataContent(this MLImage image, string mediaType = "image/png")
    {
        ArgumentNullException.ThrowIfNull(image);

        // Save to a temporary file to get proper encoding, then read bytes
        var tempPath = Path.Combine(Path.GetTempPath(), $"{Guid.NewGuid()}.{GetExtension(mediaType)}");
        try
        {
            image.Save(tempPath);
            byte[] bytes = File.ReadAllBytes(tempPath);
            return new DataContent(bytes, mediaType);
        }
        finally
        {
            if (File.Exists(tempPath))
                File.Delete(tempPath);
        }
    }

    /// <summary>
    /// Convert a MEAI DataContent back to an MLImage.
    /// Use this to consume images from MEAI interfaces in ML.NET pipelines.
    /// </summary>
    /// <param name="content">The DataContent containing image bytes.</param>
    /// <returns>An MLImage created from the DataContent bytes.</returns>
    public static MLImage ToMLImage(this DataContent content)
    {
        ArgumentNullException.ThrowIfNull(content);

        var data = content.Data;
        if (data.Length == 0)
            throw new ArgumentException("DataContent has no image data.", nameof(content));

        using var stream = new MemoryStream(data.ToArray());
        return MLImage.CreateFromStream(stream);
    }

    /// <summary>
    /// Preprocess an MLImage into a normalized CHW float tensor using a HuggingFace config.
    /// Convenience method combining MLImage with HuggingFaceImagePreprocessor.
    /// </summary>
    /// <param name="image">The input MLImage (should already be resized to target size).</param>
    /// <param name="config">Preprocessing configuration. Defaults to ImageNet if null.</param>
    /// <returns>Float array in CHW format [channels, height, width], normalized per-channel.</returns>
    public static float[] ToNormalizedTensor(this MLImage image, PreprocessorConfig? config = null)
    {
        return HuggingFaceImagePreprocessor.Preprocess(image, config ?? PreprocessorConfig.ImageNet);
    }

    private static string GetExtension(string mediaType) => mediaType switch
    {
        "image/jpeg" or "image/jpg" => "jpg",
        "image/bmp" => "bmp",
        "image/gif" => "gif",
        _ => "png"
    };
}
