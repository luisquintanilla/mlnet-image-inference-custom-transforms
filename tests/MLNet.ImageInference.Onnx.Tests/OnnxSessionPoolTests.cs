using MLNet.ImageInference.Onnx.Shared;

namespace MLNet.ImageInference.Onnx.Tests;

public class OnnxSessionPoolTests
{
    /// <summary>
    /// Constructor with invalid path throws on first Session access.
    /// </summary>
    [Fact]
    public void Constructor_InvalidPath_ThrowsOnAccess()
    {
        var pool = new OnnxSessionPool("nonexistent_model.onnx");
        Assert.Throws<Microsoft.ML.OnnxRuntime.OnnxRuntimeException>(() => _ = pool.Session);
    }

    /// <summary>
    /// Dispose doesn't throw when session was never accessed.
    /// </summary>
    [Fact]
    public void Dispose_WithoutAccess_DoesNotThrow()
    {
        var pool = new OnnxSessionPool("nonexistent_model.onnx");
        var ex = Record.Exception(() => pool.Dispose());
        Assert.Null(ex);
    }

    /// <summary>
    /// Double dispose is safe.
    /// </summary>
    [Fact]
    public void DoubleDispose_DoesNotThrow()
    {
        var pool = new OnnxSessionPool("nonexistent_model.onnx");
        var ex = Record.Exception(() =>
        {
            pool.Dispose();
            pool.Dispose();
        });
        Assert.Null(ex);
    }
}
