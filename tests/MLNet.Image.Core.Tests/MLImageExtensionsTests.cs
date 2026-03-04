using Xunit;
using Microsoft.ML.Data;
using MLNet.Image.Core;

namespace MLNet.Image.Core.Tests;

public class MLImageExtensionsTests
{
    private static MLImage CreateTestImage(int width, int height)
    {
        var pixels = new byte[width * height * 4];
        for (int i = 0; i < width * height; i++)
        {
            pixels[i * 4 + 0] = 100; // R
            pixels[i * 4 + 1] = 150; // G
            pixels[i * 4 + 2] = 200; // B
            pixels[i * 4 + 3] = 255; // A
        }
        return MLImage.CreateFromPixels(width, height, MLPixelFormat.Rgba32, pixels);
    }

    [Fact]
    public void ToDataContent_ReturnsNonEmptyDataWithMediaType()
    {
        using var image = CreateTestImage(8, 8);

        var content = image.ToDataContent();

        Assert.True(content.Data.Length > 0, "DataContent should have non-empty data");
        Assert.Equal("image/png", content.MediaType);
    }

    [Fact]
    public void ToDataContent_CustomMediaType_UsesSpecifiedType()
    {
        using var image = CreateTestImage(8, 8);

        var content = image.ToDataContent("image/jpeg");

        Assert.True(content.Data.Length > 0);
        Assert.Equal("image/jpeg", content.MediaType);
    }

    [Fact]
    public void RoundTrip_ToDataContentAndBack_PreservesDimensions()
    {
        using var image = CreateTestImage(16, 12);

        var content = image.ToDataContent();
        using var roundTripped = content.ToMLImage();

        Assert.Equal(image.Width, roundTripped.Width);
        Assert.Equal(image.Height, roundTripped.Height);
    }
}
