using Xunit;
using MLNet.Image.Core;

namespace MLNet.Image.Core.Tests;

public class SegmentationMaskTests
{
    [Fact]
    public void GetClassAt_ReturnsCorrectClassFromFlatArray()
    {
        // 3x2 mask:
        // [0, 1, 2]
        // [3, 4, 5]
        var mask = new SegmentationMask
        {
            ClassIds = [0, 1, 2, 3, 4, 5],
            Width = 3,
            Height = 2
        };

        Assert.Equal(0, mask.GetClassAt(0, 0));
        Assert.Equal(1, mask.GetClassAt(1, 0));
        Assert.Equal(2, mask.GetClassAt(2, 0));
        Assert.Equal(3, mask.GetClassAt(0, 1));
        Assert.Equal(4, mask.GetClassAt(1, 1));
        Assert.Equal(5, mask.GetClassAt(2, 1));
    }

    [Fact]
    public void GetLabelAt_WithLabels_ReturnsCorrectLabel()
    {
        var mask = new SegmentationMask
        {
            ClassIds = [0, 1, 2, 1],
            Width = 2,
            Height = 2,
            Labels = ["background", "road", "sky"]
        };

        Assert.Equal("background", mask.GetLabelAt(0, 0));
        Assert.Equal("road", mask.GetLabelAt(1, 0));
        Assert.Equal("sky", mask.GetLabelAt(0, 1));
        Assert.Equal("road", mask.GetLabelAt(1, 1));
    }

    [Fact]
    public void GetLabelAt_WithoutLabels_ReturnsNull()
    {
        var mask = new SegmentationMask
        {
            ClassIds = [0, 1, 2, 3],
            Width = 2,
            Height = 2,
            Labels = null
        };

        Assert.Null(mask.GetLabelAt(0, 0));
        Assert.Null(mask.GetLabelAt(1, 1));
    }
}
