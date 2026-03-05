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

    [SkippableFact]
    public void IsBatchDynamic_WithDynamicModel_ReturnsTrue()
    {
        Skip.Unless(File.Exists(MobileNetModelPath), "Model file not available - run scripts/download-test-models.ps1");

        var meta = ModelMetadataDiscovery.Discover(MobileNetModelPath);

        Assert.True(meta.IsBatchDynamic);
    }

    [SkippableFact]
    public void IsBatchDynamic_WithFixedModel_ReturnsFalse()
    {
        Skip.Unless(File.Exists(SqueezeNetModelPath), "Model file not available - run scripts/download-test-models.ps1");

        var meta = ModelMetadataDiscovery.Discover(SqueezeNetModelPath);

        Assert.False(meta.IsBatchDynamic);
    }
}
