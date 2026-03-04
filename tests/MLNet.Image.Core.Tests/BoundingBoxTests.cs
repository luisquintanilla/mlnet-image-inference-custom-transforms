using Xunit;
using MLNet.Image.Core;

namespace MLNet.Image.Core.Tests;

public class BoundingBoxTests
{
    [Fact]
    public void RecordEquality_SameValues_AreEqual()
    {
        var a = new BoundingBox { X = 10, Y = 20, Width = 100, Height = 50, Label = "cat", ClassId = 1, Score = 0.95f };
        var b = new BoundingBox { X = 10, Y = 20, Width = 100, Height = 50, Label = "cat", ClassId = 1, Score = 0.95f };

        Assert.Equal(a, b);
    }

    [Fact]
    public void RecordEquality_DifferentValues_AreNotEqual()
    {
        var a = new BoundingBox { X = 10, Y = 20, Width = 100, Height = 50, Label = "cat", ClassId = 1, Score = 0.95f };
        var b = new BoundingBox { X = 30, Y = 40, Width = 100, Height = 50, Label = "dog", ClassId = 2, Score = 0.80f };

        Assert.NotEqual(a, b);
    }

    [Fact]
    public void Properties_AreAccessible()
    {
        var box = new BoundingBox { X = 5, Y = 10, Width = 200, Height = 150, Label = "person", ClassId = 0, Score = 0.99f };

        Assert.Equal(5f, box.X);
        Assert.Equal(10f, box.Y);
        Assert.Equal(200f, box.Width);
        Assert.Equal(150f, box.Height);
        Assert.Equal("person", box.Label);
        Assert.Equal(0, box.ClassId);
        Assert.Equal(0.99f, box.Score);
    }
}
