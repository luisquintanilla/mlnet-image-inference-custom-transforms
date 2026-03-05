using Microsoft.ML;
using MLNet.Image.Core;
using MLNet.ImageInference.Onnx.Classification;
using MLNet.ImageInference.Onnx.DepthEstimation;
using MLNet.ImageInference.Onnx.Detection;
using MLNet.ImageInference.Onnx.Embeddings;
using MLNet.ImageInference.Onnx.ImageCaptioning;
using MLNet.ImageInference.Onnx.Segmentation;
using MLNet.ImageInference.Onnx.ZeroShot;

namespace MLNet.ImageInference.Onnx;

/// <summary>
/// Extension methods on MLContext.Transforms for image inference tasks.
/// Provides convenient entry points for building ML.NET image inference pipelines.
/// </summary>
public static class MLContextExtensions
{
    /// <summary>
    /// Create an image classification pipeline: preprocess → ONNX score → softmax → label.
    /// </summary>
    public static OnnxImageClassificationEstimator OnnxImageClassification(
        this TransformsCatalog catalog,
        OnnxImageClassificationOptions options)
    {
        ArgumentNullException.ThrowIfNull(options);
        return new OnnxImageClassificationEstimator(options);
    }

    /// <summary>
    /// Create an image embedding pipeline: preprocess → ONNX score → pooling → float[] vector.
    /// </summary>
    public static OnnxImageEmbeddingEstimator OnnxImageEmbedding(
        this TransformsCatalog catalog,
        OnnxImageEmbeddingOptions options)
    {
        ArgumentNullException.ThrowIfNull(options);
        return new OnnxImageEmbeddingEstimator(options);
    }

    /// <summary>
    /// Create an object detection pipeline: preprocess → ONNX score → NMS → bounding boxes.
    /// </summary>
    public static OnnxObjectDetectionEstimator OnnxObjectDetection(
        this TransformsCatalog catalog,
        OnnxObjectDetectionOptions options)
    {
        ArgumentNullException.ThrowIfNull(options);
        return new OnnxObjectDetectionEstimator(options);
    }

    /// <summary>
    /// Create a zero-shot image classification pipeline using CLIP: image encoder + text encoder → cosine similarity → softmax.
    /// </summary>
    public static OnnxZeroShotImageClassificationEstimator OnnxZeroShotImageClassification(
        this TransformsCatalog catalog,
        OnnxZeroShotImageClassificationOptions options)
    {
        ArgumentNullException.ThrowIfNull(options);
        return new OnnxZeroShotImageClassificationEstimator(options);
    }

    /// <summary>
    /// Create an image segmentation pipeline: preprocess → ONNX score → argmax → SegmentationMask.
    /// </summary>
    public static OnnxImageSegmentationEstimator OnnxImageSegmentation(
        this TransformsCatalog catalog,
        OnnxImageSegmentationOptions options)
    {
        ArgumentNullException.ThrowIfNull(options);
        return new OnnxImageSegmentationEstimator(options);
    }

    /// <summary>
    /// Create a depth estimation pipeline: preprocess → ONNX score → normalize → DepthMap.
    /// </summary>
    public static OnnxImageDepthEstimationEstimator OnnxImageDepthEstimation(
        this TransformsCatalog catalog,
        OnnxImageDepthEstimationOptions options)
    {
        ArgumentNullException.ThrowIfNull(options);
        return new OnnxImageDepthEstimationEstimator(options);
    }

    /// <summary>
    /// Create an image captioning pipeline: preprocess → vision encoder → autoregressive text decoder → caption.
    /// </summary>
    public static OnnxImageCaptioningEstimator OnnxImageCaptioning(
        this TransformsCatalog catalog,
        OnnxImageCaptioningOptions options)
    {
        ArgumentNullException.ThrowIfNull(options);
        return new OnnxImageCaptioningEstimator(options);
    }
}
