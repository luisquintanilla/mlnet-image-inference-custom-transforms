using Xunit;
using MLNet.Image.Core;

namespace MLNet.Image.Core.Tests;

public class DepthMapTests
{
    [Fact]
    public void GetDepthAt_ReturnsCorrectValueFromFlatArray()
    {
        // 3x2 depth map:
        // Row 0: [0.1, 0.2, 0.3]
        // Row 1: [0.4, 0.5, 0.6]
        var map = new DepthMap
        {
            Values = [0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f],
            Width = 3,
            Height = 2,
            MinDepth = 100f,
            MaxDepth = 600f
        };

        Assert.Equal(0.1f, map.GetDepthAt(0, 0));
        Assert.Equal(0.2f, map.GetDepthAt(1, 0));
        Assert.Equal(0.3f, map.GetDepthAt(2, 0));
        Assert.Equal(0.4f, map.GetDepthAt(0, 1));
        Assert.Equal(0.5f, map.GetDepthAt(1, 1));
        Assert.Equal(0.6f, map.GetDepthAt(2, 1));
    }

    [Fact]
    public void GetDepthAt_ThrowsOnOutOfRange()
    {
        var map = new DepthMap
        {
            Values = [0.5f, 0.5f, 0.5f, 0.5f],
            Width = 2,
            Height = 2
        };

        Assert.Throws<ArgumentOutOfRangeException>(() => map.GetDepthAt(-1, 0));
        Assert.Throws<ArgumentOutOfRangeException>(() => map.GetDepthAt(0, -1));
        Assert.Throws<ArgumentOutOfRangeException>(() => map.GetDepthAt(2, 0));
        Assert.Throws<ArgumentOutOfRangeException>(() => map.GetDepthAt(0, 2));
    }

    [Fact]
    public void DepthMap_PreservesMinMaxMetadata()
    {
        var map = new DepthMap
        {
            Values = [0f, 0.5f, 1f],
            Width = 3,
            Height = 1,
            MinDepth = 42.5f,
            MaxDepth = 1200.0f
        };

        Assert.Equal(42.5f, map.MinDepth);
        Assert.Equal(1200.0f, map.MaxDepth);
    }

    [Fact]
    public void DepthMap_EmptyDefaults()
    {
        var map = new DepthMap();

        Assert.Empty(map.Values);
        Assert.Equal(0, map.Width);
        Assert.Equal(0, map.Height);
        Assert.Equal(0f, map.MinDepth);
        Assert.Equal(0f, map.MaxDepth);
    }
}
