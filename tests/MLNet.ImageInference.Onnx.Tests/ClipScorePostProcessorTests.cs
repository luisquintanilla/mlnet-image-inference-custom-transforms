using Xunit;
using MLNet.ImageInference.Onnx.ZeroShot;

namespace MLNet.ImageInference.Onnx.Tests;

public class ClipScorePostProcessorTests
{
    /// <summary>
    /// Identical embeddings → highest similarity, probability near 1.0 when only candidate.
    /// </summary>
    [Fact]
    public void IdenticalEmbeddings_ProbabilityNearOne()
    {
        float[] image = [1f, 0f, 0f];
        float[][] text = [[1f, 0f, 0f]];

        var probs = ClipScorePostProcessor.ComputeScores(image, text);

        Assert.Single(probs);
        // softmax of single element is always 1.0
        Assert.Equal(1.0f, probs[0], 0.001f);
    }

    /// <summary>
    /// Orthogonal embeddings → low probability when competing with identical.
    /// </summary>
    [Fact]
    public void OrthogonalEmbeddings_LowProbability()
    {
        float[] image = [1f, 0f, 0f];
        float[][] text =
        [
            [1f, 0f, 0f], // identical → cosine 1.0
            [0f, 1f, 0f]  // orthogonal → cosine 0.0
        ];

        var probs = ClipScorePostProcessor.ComputeScores(image, text);

        Assert.Equal(2, probs.Length);
        // Identical should dominate over orthogonal
        Assert.True(probs[0] > probs[1]);
        // With logit_scale=100, exp(100) >> exp(0), so probs[0] ≈ 1
        Assert.True(probs[0] > 0.99f);
    }

    /// <summary>
    /// Multiple candidates: the matching one has highest probability.
    /// </summary>
    [Fact]
    public void MultipleCandidates_MatchingHasHighest()
    {
        float[] image = [1f, 0f, 0f];
        float[][] text =
        [
            [0f, 1f, 0f],  // orthogonal
            [1f, 0f, 0f],  // identical
            [0f, 0f, 1f]   // orthogonal
        ];

        var probs = ClipScorePostProcessor.ComputeScores(image, text);

        Assert.Equal(3, probs.Length);
        // Index 1 (identical) should have highest probability
        Assert.True(probs[1] > probs[0]);
        Assert.True(probs[1] > probs[2]);
    }

    /// <summary>
    /// Output probabilities always sum to ~1.0 (softmax property).
    /// </summary>
    [Fact]
    public void Probabilities_SumToOne()
    {
        float[] image = [0.6f, 0.8f, 0f];
        float[][] text =
        [
            [1f, 0f, 0f],
            [0f, 1f, 0f],
            [0f, 0f, 1f]
        ];

        var probs = ClipScorePostProcessor.ComputeScores(image, text);

        float sum = probs.Sum();
        Assert.Equal(1.0f, sum, 0.01f);
    }

    /// <summary>
    /// Verify logit_scale (100.0) is applied: scaled logits produce much sharper distribution.
    /// </summary>
    [Fact]
    public void TemperatureScaling_ProducesSharpDistribution()
    {
        // Two candidates with similar but not identical cosine similarities
        // image = [1,0,0], text0 = [0.9, 0.436, 0] (cos ~ 0.9), text1 = [0.8, 0.6, 0] (cos ~ 0.8)
        float[] image = [1f, 0f, 0f];

        // Normalized vectors
        float[] t0 = [0.9f, 0.4359f, 0f]; // norm ≈ 1.0, cos with image ≈ 0.9
        float[] t1 = [0.8f, 0.6f, 0f];    // norm ≈ 1.0, cos with image ≈ 0.8

        float[][] text = [t0, t1];

        var probs = ClipScorePostProcessor.ComputeScores(image, text);

        // With logit_scale=100, small differences in cosine get amplified
        // cos_diff ≈ 0.1, scaled diff = 10 → exp(10) ≈ 22000x
        // So probs[0] should be very close to 1.0
        Assert.True(probs[0] > 0.999f, $"Expected sharp distribution, got probs[0]={probs[0]}");
    }
}
