using Microsoft.Extensions.Logging;
using Microsoft.ML.OnnxRuntime;

namespace MLNet.ImageInference.Onnx.Shared;

/// <summary>
/// Thread-safe ONNX InferenceSession management.
/// Uses ThreadLocal to ensure each thread gets its own session instance,
/// avoiding concurrency issues with InferenceSession.Run().
/// </summary>
public sealed class OnnxSessionPool : IDisposable
{
    private readonly string _modelPath;
    private readonly SessionOptions? _sessionOptions;
    private readonly ThreadLocal<InferenceSession> _sessions;
    private bool _disposed;

    public OnnxSessionPool(string modelPath, SessionOptions? sessionOptions = null)
    {
        ArgumentException.ThrowIfNullOrEmpty(modelPath);

        if (!File.Exists(modelPath))
            throw new FileNotFoundException(
                $"ONNX model file not found: '{modelPath}'. Ensure the model has been downloaded and the path is correct.",
                modelPath);

        _modelPath = modelPath;
        _sessionOptions = sessionOptions;
        _sessions = new ThreadLocal<InferenceSession>(
            () => _sessionOptions is not null
                ? new InferenceSession(_modelPath, _sessionOptions)
                : new InferenceSession(_modelPath),
            trackAllValues: true);
    }

    /// <summary>
    /// Create a session pool with execution provider configuration from options.
    /// </summary>
    public OnnxSessionPool(string modelPath, IOnnxImageOptions options)
        : this(modelPath, CreateSessionOptions(options))
    {
    }

    /// <summary>
    /// Create SessionOptions configured with the specified execution provider.
    /// Falls back to CPU on initialization failure if FallbackToCpu is true.
    /// </summary>
    public static SessionOptions? CreateSessionOptions(IOnnxImageOptions options)
    {
        if (options.ExecutionProvider == OnnxExecutionProvider.CPU)
            return null;

        var sessionOptions = new SessionOptions();
        try
        {
            switch (options.ExecutionProvider)
            {
                case OnnxExecutionProvider.CUDA:
                    sessionOptions.AppendExecutionProvider_CUDA(options.GpuDeviceId);
                    break;
                case OnnxExecutionProvider.DirectML:
                    sessionOptions.AppendExecutionProvider_DML(options.GpuDeviceId);
                    break;
                case OnnxExecutionProvider.TensorRT:
                    sessionOptions.AppendExecutionProvider_Tensorrt(options.GpuDeviceId);
                    break;
            }
            options.Logger?.LogInformation("Configured {Provider} execution provider (device {DeviceId})",
                options.ExecutionProvider, options.GpuDeviceId);
            return sessionOptions;
        }
        catch (OnnxRuntimeException ex) when (options.FallbackToCpu)
        {
            options.Logger?.LogWarning(ex, "{Provider} initialization failed, falling back to CPU", options.ExecutionProvider);
            sessionOptions.Dispose();
            return null; // Fall back to CPU
        }
    }

    /// <summary>
    /// Get the InferenceSession for the current thread.
    /// </summary>
    public InferenceSession Session
    {
        get
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            return _sessions.Value!;
        }
    }

    /// <summary>
    /// The model path this pool was created with.
    /// </summary>
    public string ModelPath => _modelPath;

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;

        if (_sessions.IsValueCreated)
        {
            foreach (var session in _sessions.Values)
            {
                session?.Dispose();
            }
        }
        _sessions.Dispose();
    }
}
