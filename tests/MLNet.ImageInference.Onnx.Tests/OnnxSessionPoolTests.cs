using Xunit;
using MLNet.ImageInference.Onnx.Shared;

namespace MLNet.ImageInference.Onnx.Tests;

public class OnnxSessionPoolTests
{
    private const string SqueezeNetPath = "models/squeezenet/model.onnx";

    /// <summary>
    /// Constructor with invalid path throws FileNotFoundException eagerly.
    /// </summary>
    [Fact]
    public void Constructor_InvalidPath_ThrowsFileNotFoundException()
    {
        Assert.Throws<FileNotFoundException>(() => new OnnxSessionPool("nonexistent_model.onnx"));
    }

    /// <summary>
    /// Dispose doesn't throw when pool was created with a valid model.
    /// </summary>
    [Fact]
    public void Dispose_WithoutAccess_DoesNotThrow()
    {
        if (!File.Exists(SqueezeNetPath))
            return;

        var pool = new OnnxSessionPool(SqueezeNetPath);
        var ex = Record.Exception(() => pool.Dispose());
        Assert.Null(ex);
    }

    /// <summary>
    /// Double dispose is safe.
    /// </summary>
    [Fact]
    public void DoubleDispose_DoesNotThrow()
    {
        if (!File.Exists(SqueezeNetPath))
            return;

        var pool = new OnnxSessionPool(SqueezeNetPath);
        var ex = Record.Exception(() =>
        {
            pool.Dispose();
            pool.Dispose();
        });
        Assert.Null(ex);
    }
}
