using Xunit;
using Microsoft.ML.Data;
using MLNet.Image.Core;

namespace MLNet.Image.Core.Tests;

public class HuggingFaceImagePreprocessorTests
{
    private static MLImage CreateSolidColorImage(int width, int height, byte r, byte g, byte b)
    {
        // MLImage.CreateFromPixels expects pixel data in the specified format
        var pixels = new byte[width * height * 4]; // RGBA32
        for (int i = 0; i < width * height; i++)
        {
            pixels[i * 4 + 0] = r;
            pixels[i * 4 + 1] = g;
            pixels[i * 4 + 2] = b;
            pixels[i * 4 + 3] = 255; // Alpha
        }
        return MLImage.CreateFromPixels(width, height, MLPixelFormat.Rgba32, pixels);
    }

    [Fact]
    public void Preprocess_SolidRed_OutputTensorHasCorrectDimensions()
    {
        using var image = CreateSolidColorImage(4, 4, 255, 0, 0);
        var config = PreprocessorConfig.ImageNet;

        float[] tensor = HuggingFaceImagePreprocessor.Preprocess(image, config);

        Assert.Equal(3 * 4 * 4, tensor.Length);
    }

    [Fact]
    public void Preprocess_SolidRed_ChannelValuesAreCorrect()
    {
        using var image = CreateSolidColorImage(4, 4, 255, 0, 0);
        var config = PreprocessorConfig.ImageNet;

        float[] tensor = HuggingFaceImagePreprocessor.Preprocess(image, config);

        int hw = 4 * 4;

        // R channel: (255/255.0 - 0.485) / 0.229 ≈ 2.2489
        float expectedR = (1.0f - 0.485f) / 0.229f;
        Assert.Equal(expectedR, tensor[0], precision: 3);

        // G channel: (0/255.0 - 0.456) / 0.224 ≈ -2.0357
        float expectedG = (0.0f - 0.456f) / 0.224f;
        Assert.Equal(expectedG, tensor[hw], precision: 3);

        // B channel: (0/255.0 - 0.406) / 0.225 ≈ -1.8044
        float expectedB = (0.0f - 0.406f) / 0.225f;
        Assert.Equal(expectedB, tensor[2 * hw], precision: 3);
    }

    [Fact]
    public void Preprocess_SolidRed_AllPixelsInChannelAreUniform()
    {
        using var image = CreateSolidColorImage(4, 4, 255, 0, 0);
        var config = PreprocessorConfig.ImageNet;

        float[] tensor = HuggingFaceImagePreprocessor.Preprocess(image, config);

        int hw = 4 * 4;

        // All R channel values should be identical for a solid color image
        for (int i = 1; i < hw; i++)
        {
            Assert.Equal(tensor[0], tensor[i]);
        }
        for (int i = hw + 1; i < 2 * hw; i++)
        {
            Assert.Equal(tensor[hw], tensor[i]);
        }
        for (int i = 2 * hw + 1; i < 3 * hw; i++)
        {
            Assert.Equal(tensor[2 * hw], tensor[i]);
        }
    }

    [Fact]
    public void Preprocess_CHWFormat_ChannelsAreSeparated()
    {
        using var image = CreateSolidColorImage(4, 4, 255, 0, 0);
        var config = PreprocessorConfig.ImageNet;

        float[] tensor = HuggingFaceImagePreprocessor.Preprocess(image, config);

        int hw = 4 * 4;

        // R channel values (positive for red) should differ from G/B (negative)
        Assert.True(tensor[0] > 0, "R channel for red pixel should be positive");
        Assert.True(tensor[hw] < 0, "G channel for red pixel should be negative");
        Assert.True(tensor[2 * hw] < 0, "B channel for red pixel should be negative");
    }

    [Fact]
    public void Preprocess_DifferentConfigs_ProduceDifferentResults()
    {
        using var image = CreateSolidColorImage(4, 4, 128, 64, 32);

        float[] imageNetResult = HuggingFaceImagePreprocessor.Preprocess(image, PreprocessorConfig.ImageNet);
        float[] clipResult = HuggingFaceImagePreprocessor.Preprocess(image, PreprocessorConfig.CLIP);

        // Different configs should produce different normalized values
        Assert.NotEqual(imageNetResult[0], clipResult[0]);
    }
}
