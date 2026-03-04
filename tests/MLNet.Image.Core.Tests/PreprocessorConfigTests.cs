using Xunit;
using MLNet.Image.Core;

namespace MLNet.Image.Core.Tests;

public class PreprocessorConfigTests
{
    [Theory]
    [InlineData(nameof(PreprocessorConfig.ImageNet))]
    [InlineData(nameof(PreprocessorConfig.CLIP))]
    [InlineData(nameof(PreprocessorConfig.DINOv2))]
    [InlineData(nameof(PreprocessorConfig.YOLOv8))]
    [InlineData(nameof(PreprocessorConfig.SegFormer))]
    public void Preset_HasThreeChannelMeanAndStd(string presetName)
    {
        var config = GetPreset(presetName);
        Assert.Equal(3, config.Mean.Length);
        Assert.Equal(3, config.Std.Length);
    }

    [Theory]
    [InlineData(nameof(PreprocessorConfig.ImageNet))]
    [InlineData(nameof(PreprocessorConfig.CLIP))]
    [InlineData(nameof(PreprocessorConfig.DINOv2))]
    [InlineData(nameof(PreprocessorConfig.SegFormer))]
    public void Preset_MeanValuesInZeroOneRange(string presetName)
    {
        var config = GetPreset(presetName);
        foreach (var mean in config.Mean)
        {
            Assert.InRange(mean, 0f, 1f);
        }
    }

    [Theory]
    [InlineData(nameof(PreprocessorConfig.ImageNet))]
    [InlineData(nameof(PreprocessorConfig.CLIP))]
    [InlineData(nameof(PreprocessorConfig.DINOv2))]
    [InlineData(nameof(PreprocessorConfig.SegFormer))]
    public void Preset_StdValuesArePositive(string presetName)
    {
        var config = GetPreset(presetName);
        foreach (var std in config.Std)
        {
            Assert.True(std > 0, $"Std value {std} should be positive");
        }
    }

    [Theory]
    [InlineData(nameof(PreprocessorConfig.ImageNet))]
    [InlineData(nameof(PreprocessorConfig.CLIP))]
    [InlineData(nameof(PreprocessorConfig.DINOv2))]
    [InlineData(nameof(PreprocessorConfig.YOLOv8))]
    [InlineData(nameof(PreprocessorConfig.SegFormer))]
    public void Preset_ImageSizeIsPositive(string presetName)
    {
        var config = GetPreset(presetName);
        Assert.True(config.ImageSize.Height > 0);
        Assert.True(config.ImageSize.Width > 0);
    }

    [Theory]
    [InlineData(nameof(PreprocessorConfig.ImageNet))]
    [InlineData(nameof(PreprocessorConfig.CLIP))]
    [InlineData(nameof(PreprocessorConfig.DINOv2))]
    [InlineData(nameof(PreprocessorConfig.YOLOv8))]
    [InlineData(nameof(PreprocessorConfig.SegFormer))]
    public void Preset_RescaleFactorIsPositive(string presetName)
    {
        var config = GetPreset(presetName);
        Assert.True(config.RescaleFactor > 0);
    }

    private static PreprocessorConfig GetPreset(string name) => name switch
    {
        nameof(PreprocessorConfig.ImageNet) => PreprocessorConfig.ImageNet,
        nameof(PreprocessorConfig.CLIP) => PreprocessorConfig.CLIP,
        nameof(PreprocessorConfig.DINOv2) => PreprocessorConfig.DINOv2,
        nameof(PreprocessorConfig.YOLOv8) => PreprocessorConfig.YOLOv8,
        nameof(PreprocessorConfig.SegFormer) => PreprocessorConfig.SegFormer,
        _ => throw new ArgumentException($"Unknown preset: {name}")
    };
}
