namespace MLNet.ImageGeneration.OnnxGenAI;

/// <summary>
/// ONNX Runtime execution provider for hardware acceleration.
/// </summary>
public enum OnnxExecutionProvider
{
    /// <summary>Default CPU execution provider.</summary>
    CPU = 0,

    /// <summary>NVIDIA CUDA execution provider (requires CUDA toolkit + OnnxRuntime.GPU package).</summary>
    CUDA,

    /// <summary>DirectML execution provider for Windows GPU acceleration (AMD, Intel, NVIDIA).</summary>
    DirectML,

    /// <summary>NVIDIA TensorRT execution provider for optimized inference.</summary>
    TensorRT
}
