namespace MLNet.Image.Core;

/// <summary>
/// Configuration for HuggingFace-compatible image preprocessing.
/// Maps directly to fields in a HuggingFace preprocessor_config.json file.
/// </summary>
public record PreprocessorConfig
{
    /// <summary>
    /// Per-channel mean values for normalization.
    /// Applied as: (pixel - mean[c]) / std[c]
    /// </summary>
    public float[] Mean { get; init; } = [0.485f, 0.456f, 0.406f];

    /// <summary>
    /// Per-channel standard deviation values for normalization.
    /// Applied as: (pixel - mean[c]) / std[c]
    /// </summary>
    public float[] Std { get; init; } = [0.229f, 0.224f, 0.225f];

    /// <summary>
    /// Whether to rescale pixel values from [0, 255] to [0.0, 1.0].
    /// </summary>
    public bool DoRescale { get; init; } = true;

    /// <summary>
    /// Scale factor for rescaling (typically 1/255).
    /// </summary>
    public float RescaleFactor { get; init; } = 1f / 255f;

    /// <summary>
    /// Whether to normalize using Mean and Std.
    /// </summary>
    public bool DoNormalize { get; init; } = true;

    /// <summary>
    /// Target image size (height, width) for resizing.
    /// </summary>
    public (int Height, int Width) ImageSize { get; init; } = (224, 224);

    /// <summary>
    /// Whether to center-crop the image after resizing.
    /// </summary>
    public bool DoCenterCrop { get; init; } = true;

    /// <summary>
    /// Crop size (height, width) for center cropping.
    /// </summary>
    public (int Height, int Width) CropSize { get; init; } = (224, 224);

    /// <summary>
    /// Whether to convert to RGB if image has alpha channel.
    /// </summary>
    public bool DoConvertRgb { get; init; } = true;

    /// <summary>
    /// ImageNet default preset: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], size=224x224.
    /// Used by ViT, ResNet, DeiT, and most vision transformers trained on ImageNet.
    /// </summary>
    public static PreprocessorConfig ImageNet => new();

    /// <summary>
    /// CLIP preset: mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711], size=224x224.
    /// Used by CLIP ViT-B/32 and similar models.
    /// </summary>
    public static PreprocessorConfig CLIP => new()
    {
        Mean = [0.48145466f, 0.4578275f, 0.40821073f],
        Std = [0.26862954f, 0.26130258f, 0.27577711f],
        ImageSize = (224, 224),
        CropSize = (224, 224)
    };

    /// <summary>
    /// DINOv2 preset: same as ImageNet defaults.
    /// </summary>
    public static PreprocessorConfig DINOv2 => ImageNet;

    /// <summary>
    /// YOLOv8 preset: no normalization, just rescale to [0, 1], size=640x640.
    /// </summary>
    public static PreprocessorConfig YOLOv8 => new()
    {
        DoRescale = true,
        DoNormalize = false,
        ImageSize = (640, 640),
        DoCenterCrop = false,
        CropSize = (640, 640)
    };

    /// <summary>
    /// SegFormer preset: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], size=512x512.
    /// </summary>
    public static PreprocessorConfig SegFormer => new()
    {
        ImageSize = (512, 512),
        CropSize = (512, 512),
        DoCenterCrop = false
    };
}
