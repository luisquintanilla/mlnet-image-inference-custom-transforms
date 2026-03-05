using Microsoft.Extensions.Logging;

namespace MLNet.ImageInference.Onnx.Shared;

/// <summary>
/// Common properties shared by all ONNX image inference options classes.
/// </summary>
public interface IOnnxImageOptions
{
    /// <summary>Name of the IDataView column containing the input MLImage.</summary>
    string InputColumnName { get; }

    /// <summary>Maximum number of images to batch in a single ONNX inference call.</summary>
    int BatchSize { get; }

    /// <summary>
    /// Execution provider for hardware acceleration. Default: CPU.
    /// Requires the corresponding OnnxRuntime package (e.g., Microsoft.ML.OnnxRuntime.GPU for CUDA).
    /// </summary>
    OnnxExecutionProvider ExecutionProvider => OnnxExecutionProvider.CPU;

    /// <summary>GPU device ID for CUDA/DirectML/TensorRT. Default: 0.</summary>
    int GpuDeviceId => 0;

    /// <summary>
    /// If true, silently falls back to CPU when the requested GPU execution provider fails to initialize.
    /// Default: true.
    /// </summary>
    bool FallbackToCpu => true;

    /// <summary>
    /// Optional logger for diagnostics (model load, inference timing, GPU fallback).
    /// When null, no logging is performed.
    /// </summary>
    ILogger? Logger => null;
}
