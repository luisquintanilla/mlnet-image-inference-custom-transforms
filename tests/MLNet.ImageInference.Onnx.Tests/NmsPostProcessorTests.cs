using MLNet.ImageInference.Onnx.Detection;

namespace MLNet.ImageInference.Onnx.Tests;

public class NmsPostProcessorTests
{
    /// <summary>
    /// 3 non-overlapping boxes, all above threshold → all kept.
    /// Layout: 2 classes, 3 boxes => stride=6, raw shape [1,6,3].
    /// After transpose each row is [xc, yc, w, h, class0, class1].
    /// </summary>
    [Fact]
    public void NonOverlappingBoxes_AllKept()
    {
        // 3 boxes, 2 classes, stride = 6
        // Raw layout [1, 6, 3]: row-major across boxes first
        // row0 (xc): 50, 200, 400
        // row1 (yc): 50, 200, 400
        // row2 (w):  20, 20,  20
        // row3 (h):  20, 20,  20
        // row4 (c0): 0.9, 0.8, 0.7
        // row5 (c1): 0.1, 0.1, 0.1
        float[] raw =
        [
            50, 200, 400,   // xc
            50, 200, 400,   // yc
            20,  20,  20,   // w
            20,  20,  20,   // h
            0.9f, 0.8f, 0.7f, // class0
            0.1f, 0.1f, 0.1f  // class1
        ];

        var results = NmsPostProcessor.Apply(raw, numClasses: 2, numBoxes: 3,
            confidenceThreshold: 0.5f, iouThreshold: 0.45f, labels: null);

        Assert.Equal(3, results.Length);
        // All are class 0
        Assert.All(results, b => Assert.Equal(0, b.ClassId));
        // Sorted by score descending
        Assert.Equal(0.9f, results[0].Score);
        Assert.Equal(0.8f, results[1].Score);
        Assert.Equal(0.7f, results[2].Score);
    }

    /// <summary>
    /// 2 boxes same class, high IoU → only highest confidence kept.
    /// </summary>
    [Fact]
    public void OverlappingBoxes_SameClass_HighestKept()
    {
        // 2 overlapping boxes at nearly the same location, 1 class
        // stride = 5 (4 coords + 1 class), raw shape [1,5,2]
        float[] raw =
        [
            100, 102,   // xc (nearly identical)
            100, 102,   // yc
             50,  50,   // w
             50,  50,   // h
            0.9f, 0.7f  // class0
        ];

        var results = NmsPostProcessor.Apply(raw, numClasses: 1, numBoxes: 2,
            confidenceThreshold: 0.5f, iouThreshold: 0.45f, labels: null);

        Assert.Single(results);
        Assert.Equal(0.9f, results[0].Score);
    }

    /// <summary>
    /// Box below threshold → filtered out.
    /// </summary>
    [Fact]
    public void ConfidenceThreshold_FiltersLowScores()
    {
        // 2 boxes, 1 class. One above threshold, one below.
        float[] raw =
        [
            100, 300,  // xc
            100, 300,  // yc
             20,  20,  // w
             20,  20,  // h
            0.9f, 0.3f // class0
        ];

        var results = NmsPostProcessor.Apply(raw, numClasses: 1, numBoxes: 2,
            confidenceThreshold: 0.5f, iouThreshold: 0.45f, labels: null);

        Assert.Single(results);
        Assert.Equal(0.9f, results[0].Score);
    }

    /// <summary>
    /// 2 overlapping boxes with different classes → both kept (NMS is per-class).
    /// </summary>
    [Fact]
    public void MultiClass_OverlappingDifferentClasses_BothKept()
    {
        // 2 boxes at same location, different best class
        // stride = 6 (4 + 2 classes), raw [1,6,2]
        float[] raw =
        [
            100, 100,   // xc
            100, 100,   // yc
             50,  50,   // w
             50,  50,   // h
            0.9f, 0.1f, // class0 scores
            0.1f, 0.9f  // class1 scores
        ];

        var results = NmsPostProcessor.Apply(raw, numClasses: 2, numBoxes: 2,
            confidenceThreshold: 0.5f, iouThreshold: 0.45f, labels: null);

        Assert.Equal(2, results.Length);
        var classIds = results.Select(r => r.ClassId).OrderBy(c => c).ToArray();
        Assert.Equal([0, 1], classIds);
    }

    /// <summary>
    /// All below threshold → empty result.
    /// </summary>
    [Fact]
    public void AllBelowThreshold_EmptyResult()
    {
        float[] raw =
        [
            100, 200,
            100, 200,
             20,  20,
             20,  20,
            0.1f, 0.2f // both below 0.5
        ];

        var results = NmsPostProcessor.Apply(raw, numClasses: 1, numBoxes: 2,
            confidenceThreshold: 0.5f, iouThreshold: 0.45f, labels: null);

        Assert.Empty(results);
    }

    /// <summary>
    /// Verify TransposeYoloOutput with known small data.
    /// Input [1,6,3] (rows=6, cols=3) → transposed to [3,6].
    /// </summary>
    [Fact]
    public void TransposeYoloOutput_CorrectLayout()
    {
        // rows=6 (stride), cols=3 (boxes)
        // Input row-major: row0=[1,2,3], row1=[4,5,6], ...
        float[] input =
        [
            1, 2, 3,
            4, 5, 6,
            7, 8, 9,
            10, 11, 12,
            13, 14, 15,
            16, 17, 18
        ];

        var result = NmsPostProcessor.TransposeYoloOutput(input, rows: 6, cols: 3);

        // After transpose: [3, 6]
        // box0: [1, 4, 7, 10, 13, 16]
        // box1: [2, 5, 8, 11, 14, 17]
        // box2: [3, 6, 9, 12, 15, 18]
        Assert.Equal(18, result.Length);
        Assert.Equal([1, 4, 7, 10, 13, 16], result[0..6]);
        Assert.Equal([2, 5, 8, 11, 14, 17], result[6..12]);
        Assert.Equal([3, 6, 9, 12, 15, 18], result[12..18]);
    }

    /// <summary>
    /// When labels provided, BoundingBox.Label is set correctly.
    /// </summary>
    [Fact]
    public void Labels_AssignedCorrectly()
    {
        string[] labels = ["cat", "dog"];
        // 1 box, 2 classes, class1 is best
        float[] raw =
        [
            100,  // xc
            100,  // yc
             50,  // w
             50,  // h
            0.3f, // class0
            0.9f  // class1
        ];

        var results = NmsPostProcessor.Apply(raw, numClasses: 2, numBoxes: 1,
            confidenceThreshold: 0.5f, iouThreshold: 0.45f, labels: labels);

        Assert.Single(results);
        Assert.Equal("dog", results[0].Label);
        Assert.Equal(1, results[0].ClassId);
    }
}
