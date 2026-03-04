using MLNet.ImageInference.Onnx.Shared;
using Xunit;

namespace MLNet.ImageInference.Onnx.Tests;

/// <summary>
/// Tests for <see cref="ModelMetadataDiscovery"/> metadata properties.
/// </summary>
public class ModelMetadataTests
{
    private const string MobileNetModelPath = "models/mobilenet/model.onnx";
    private const string SqueezeNetModelPath = "models/squeezenet/model.onnx";

    [Fact]
    public void IsBatchDynamic_WithDynamicModel_ReturnsTrue()
    {
        if (!File.Exists(MobileNetModelPath)) return;

        var meta = ModelMetadataDiscovery.Discover(MobileNetModelPath);

        Assert.True(meta.IsBatchDynamic);
    }

    [Fact]
    public void IsBatchDynamic_WithFixedModel_ReturnsFalse()
    {
        if (!File.Exists(SqueezeNetModelPath)) return;

        var meta = ModelMetadataDiscovery.Discover(SqueezeNetModelPath);

        Assert.False(meta.IsBatchDynamic);
    }
}
