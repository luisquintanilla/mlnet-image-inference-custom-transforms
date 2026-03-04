using Xunit;
using MLNet.ImageInference.Onnx.Segmentation;

namespace MLNet.ImageInference.Onnx.Tests;

public class MaskPostProcessorTests
{
    /// <summary>
    /// 2-class, 2x2 image → known class assignments via argmax.
    /// Layout [1,2,2,2]: class0 logits then class1 logits.
    /// </summary>
    [Fact]
    public void BasicArgmax_KnownClassAssignments()
    {
        // [1, 2, 2, 2] layout: c * H * W + y * W + x
        // class0: (0,0)=5 (1,0)=1 (0,1)=1 (1,1)=5
        // class1: (0,0)=1 (1,0)=5 (0,1)=5 (1,1)=1
        float[] raw =
        [
            // class 0, row0: (x=0,y=0), (x=1,y=0)
            5f, 1f,
            // class 0, row1: (x=0,y=1), (x=1,y=1)
            1f, 5f,
            // class 1, row0
            1f, 5f,
            // class 1, row1
            5f, 1f
        ];

        var mask = MaskPostProcessor.Apply(raw, numClasses: 2, height: 2, width: 2,
            originalWidth: null, originalHeight: null, labels: null);

        Assert.Equal(2, mask.Width);
        Assert.Equal(2, mask.Height);
        // (0,0) → class 0, (1,0) → class 1, (0,1) → class 1, (1,1) → class 0
        Assert.Equal(0, mask.GetClassAt(0, 0));
        Assert.Equal(1, mask.GetClassAt(1, 0));
        Assert.Equal(1, mask.GetClassAt(0, 1));
        Assert.Equal(0, mask.GetClassAt(1, 1));
    }

    /// <summary>
    /// All pixels same class → all ClassIds equal.
    /// </summary>
    [Fact]
    public void UniformClass_AllSameClassId()
    {
        // 2 classes, 2x2. Class 1 always has higher logit.
        float[] raw =
        [
            // class 0
            0f, 0f,
            0f, 0f,
            // class 1
            1f, 1f,
            1f, 1f
        ];

        var mask = MaskPostProcessor.Apply(raw, numClasses: 2, height: 2, width: 2,
            originalWidth: null, originalHeight: null, labels: null);

        Assert.All(mask.ClassIds, id => Assert.Equal(1, id));
    }

    /// <summary>
    /// When labels provided, GetLabelAt returns correct label.
    /// </summary>
    [Fact]
    public void Labels_GetLabelAtReturnsCorrectLabel()
    {
        string[] labels = ["background", "person"];
        // 1x1 image, class 1 wins
        float[] raw = [0f, 1f];

        var mask = MaskPostProcessor.Apply(raw, numClasses: 2, height: 1, width: 1,
            originalWidth: null, originalHeight: null, labels: labels);

        Assert.Equal("person", mask.GetLabelAt(0, 0));
    }

    /// <summary>
    /// 2x2 mask resized to 4x4 via nearest-neighbor → each original pixel covers 2x2 area.
    /// </summary>
    [Fact]
    public void Resize_2x2To4x4_NearestNeighbor()
    {
        // 2 classes, 2x2 input
        // class 0 = top-left/bottom-right, class 1 = top-right/bottom-left
        float[] raw =
        [
            // class 0
            5f, 1f,
            1f, 5f,
            // class 1
            1f, 5f,
            5f, 1f
        ];

        var mask = MaskPostProcessor.Apply(raw, numClasses: 2, height: 2, width: 2,
            originalWidth: 4, originalHeight: 4, labels: null);

        Assert.Equal(4, mask.Width);
        Assert.Equal(4, mask.Height);

        // Top-left 2x2 block = class 0
        Assert.Equal(0, mask.GetClassAt(0, 0));
        Assert.Equal(0, mask.GetClassAt(1, 0));
        Assert.Equal(0, mask.GetClassAt(0, 1));
        Assert.Equal(0, mask.GetClassAt(1, 1));

        // Top-right 2x2 block = class 1
        Assert.Equal(1, mask.GetClassAt(2, 0));
        Assert.Equal(1, mask.GetClassAt(3, 0));
        Assert.Equal(1, mask.GetClassAt(2, 1));
        Assert.Equal(1, mask.GetClassAt(3, 1));

        // Bottom-left 2x2 block = class 1
        Assert.Equal(1, mask.GetClassAt(0, 2));
        Assert.Equal(1, mask.GetClassAt(1, 2));
        Assert.Equal(1, mask.GetClassAt(0, 3));
        Assert.Equal(1, mask.GetClassAt(1, 3));

        // Bottom-right 2x2 block = class 0
        Assert.Equal(0, mask.GetClassAt(2, 2));
        Assert.Equal(0, mask.GetClassAt(3, 2));
        Assert.Equal(0, mask.GetClassAt(2, 3));
        Assert.Equal(0, mask.GetClassAt(3, 3));
    }

    /// <summary>
    /// No resize when originalWidth/Height match model dimensions.
    /// </summary>
    [Fact]
    public void NoResize_WhenDimensionsMatch()
    {
        float[] raw =
        [
            // class 0
            5f, 1f,
            // class 1
            1f, 5f
        ];

        var mask = MaskPostProcessor.Apply(raw, numClasses: 2, height: 1, width: 2,
            originalWidth: 2, originalHeight: 1, labels: null);

        Assert.Equal(2, mask.Width);
        Assert.Equal(1, mask.Height);
        Assert.Equal(0, mask.GetClassAt(0, 0));
        Assert.Equal(1, mask.GetClassAt(1, 0));
    }
}
