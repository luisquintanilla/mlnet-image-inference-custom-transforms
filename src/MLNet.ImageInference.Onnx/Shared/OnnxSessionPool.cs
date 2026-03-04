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
        _modelPath = modelPath;
        _sessionOptions = sessionOptions;
        _sessions = new ThreadLocal<InferenceSession>(
            () => _sessionOptions is not null
                ? new InferenceSession(_modelPath, _sessionOptions)
                : new InferenceSession(_modelPath),
            trackAllValues: true);
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
