using Xunit;
using MLNet.ImageInference.Onnx.Shared;
using MLNet.ImageInference.Onnx.Classification;
using Microsoft.ML.OnnxRuntime;

namespace MLNet.ImageInference.Onnx.Tests;

/// <summary>
/// Tests for Phase 14 production hardening: GPU/CUDA configuration, CancellationToken, model validation.
/// </summary>
public class ProductionHardeningTests
{
    private const string SqueezeNetPath = "models/squeezenet/model.onnx";

    /// <summary>
    /// Check if the native OnnxRuntime library is available (not present on CI without native packages).
    /// </summary>
    private static bool IsOnnxRuntimeNativeAvailable()
    {
        try
        {
            _ = new SessionOptions();
            return true;
        }
        catch
        {
            return false;
        }
    }

    // --- GPU / CUDA Tests ---

    [Fact]
    public void CreateSessionOptions_CpuProvider_ReturnsNull()
    {
        var options = new OnnxImageClassificationOptions
        {
            ModelPath = SqueezeNetPath
        };
        // CPU provider should return null (default SessionOptions) — no native DLL needed
        var sessionOptions = OnnxSessionPool.CreateSessionOptions(options);
        Assert.Null(sessionOptions);
    }

    [Fact]
    public void CreateSessionOptions_CudaWithFallback_FallsBackToCpu()
    {
        if (!IsOnnxRuntimeNativeAvailable())
            return; // Native OnnxRuntime not available (e.g., Linux CI with Managed-only package)

        // Since we only have Microsoft.ML.OnnxRuntime.Managed (CPU-only),
        // requesting CUDA should throw — but FallbackToCpu catches it and returns null.
        var options = new CudaTestOptions
        {
            ModelPath = SqueezeNetPath,
            ExecutionProvider = OnnxExecutionProvider.CUDA,
            GpuDeviceId = 0,
            FallbackToCpu = true
        };
        var sessionOptions = OnnxSessionPool.CreateSessionOptions(options);
        // Should fall back to null (CPU) since CUDA is not available
        Assert.Null(sessionOptions);
    }

    [Fact]
    public void CreateSessionOptions_CudaWithoutFallback_Throws()
    {
        if (!IsOnnxRuntimeNativeAvailable())
            return; // Native OnnxRuntime not available (e.g., Linux CI with Managed-only package)

        // Without FallbackToCpu, requesting CUDA on CPU-only runtime should throw
        // EntryPointNotFoundException (no CUDA entry point in CPU-only onnxruntime DLL)
        var options = new CudaTestOptions
        {
            ModelPath = SqueezeNetPath,
            ExecutionProvider = OnnxExecutionProvider.CUDA,
            GpuDeviceId = 0,
            FallbackToCpu = false
        };
        Assert.Throws<EntryPointNotFoundException>(() => OnnxSessionPool.CreateSessionOptions(options));
    }

    [Fact]
    public void SessionPool_CudaWithFallback_CreatesWorkingSession()
    {
        if (!File.Exists(SqueezeNetPath))
            return;

        // Even when requesting CUDA, FallbackToCpu should let us create a working pool
        var options = new CudaTestOptions
        {
            ModelPath = SqueezeNetPath,
            ExecutionProvider = OnnxExecutionProvider.CUDA,
            GpuDeviceId = 0,
            FallbackToCpu = true
        };
        using var pool = new OnnxSessionPool(SqueezeNetPath, options);
        var session = pool.Session;
        Assert.NotNull(session);
    }

    // --- CancellationToken Tests ---

    [Fact]
    public void Classify_CancelledToken_ThrowsOperationCancelled()
    {
        if (!File.Exists(SqueezeNetPath))
            return;

        using var cts = new CancellationTokenSource();
        cts.Cancel(); // Pre-cancel

        using var transformer = new OnnxImageClassificationTransformer(
            new OnnxImageClassificationOptions { ModelPath = SqueezeNetPath });

        using var image = TestImageHelper.CreateSolidColorImage(224, 224, 128, 64, 200);
        Assert.Throws<OperationCanceledException>(() =>
            transformer.Classify(image, cts.Token));
    }

    [Fact]
    public void ClassifyBatch_CancelledToken_ThrowsOperationCancelled()
    {
        if (!File.Exists(SqueezeNetPath))
            return;

        using var cts = new CancellationTokenSource();
        cts.Cancel();

        using var transformer = new OnnxImageClassificationTransformer(
            new OnnxImageClassificationOptions { ModelPath = SqueezeNetPath });

        using var image = TestImageHelper.CreateSolidColorImage(224, 224, 128, 64, 200);
        Assert.Throws<OperationCanceledException>(() =>
            transformer.ClassifyBatch([image], cts.Token));
    }

    // --- Model Validation Tests ---

    [Fact]
    public void SessionPool_MissingModel_ThrowsFileNotFoundException()
    {
        var ex = Assert.Throws<FileNotFoundException>(() =>
            new OnnxSessionPool("nonexistent/path/model.onnx"));
        Assert.Contains("model.onnx", ex.Message);
    }

    [Fact]
    public void SessionPool_NullPath_ThrowsArgumentException()
    {
        Assert.Throws<ArgumentException>(() => new OnnxSessionPool(""));
    }

    // --- Helper class for CUDA test options ---
    private class CudaTestOptions : IOnnxImageOptions
    {
        public required string ModelPath { get; init; }
        public string InputColumnName => "Image";
        public int BatchSize => 1;
        public OnnxExecutionProvider ExecutionProvider { get; init; } = OnnxExecutionProvider.CPU;
        public int GpuDeviceId { get; init; } = 0;
        public bool FallbackToCpu { get; init; } = true;
    }

    // --- Test image helper ---
    private static class TestImageHelper
    {
        public static Microsoft.ML.Data.MLImage CreateSolidColorImage(int w, int h, byte r, byte g, byte b)
        {
            var pixels = new byte[w * h * 4];
            for (int i = 0; i < pixels.Length; i += 4)
            {
                pixels[i] = r;
                pixels[i + 1] = g;
                pixels[i + 2] = b;
                pixels[i + 3] = 255;
            }
            return Microsoft.ML.Data.MLImage.CreateFromPixels(w, h, Microsoft.ML.Data.MLPixelFormat.Rgba32, pixels);
        }
    }
}
