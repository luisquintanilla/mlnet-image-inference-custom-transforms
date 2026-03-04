using System.Numerics.Tensors;

namespace MLNet.ImageInference.Onnx.ZeroShot;

/// <summary>
/// Static helper for CLIP cosine similarity scoring between image and text embeddings.
/// </summary>
public static class ClipScorePostProcessor
{
    // CLIP uses logit_scale = ln(100) ≈ 4.605, so exp(logit_scale) = 100.0
    private const float LogitScale = 100.0f;

    /// <summary>
    /// Compute class probabilities from CLIP image and text embeddings.
    /// Calculates cosine similarity, scales by CLIP's logit_scale (100),
    /// then applies softmax to produce a probability distribution.
    /// </summary>
    /// <param name="imageEmbedding">L2-normalized image embedding from the vision encoder.</param>
    /// <param name="textEmbeddings">L2-normalized text embeddings from the text encoder (one per candidate label).</param>
    /// <returns>Probability per candidate label (sums to 1).</returns>
    public static float[] ComputeScores(float[] imageEmbedding, float[][] textEmbeddings)
    {
        ArgumentNullException.ThrowIfNull(imageEmbedding);
        ArgumentNullException.ThrowIfNull(textEmbeddings);

        float imageNorm = TensorPrimitives.Norm(imageEmbedding);
        var logits = new float[textEmbeddings.Length];

        for (int i = 0; i < textEmbeddings.Length; i++)
        {
            float dot = TensorPrimitives.Dot(imageEmbedding, textEmbeddings[i]);
            float textNorm = TensorPrimitives.Norm(textEmbeddings[i]);
            float cosineSim = (imageNorm > 0 && textNorm > 0) ? dot / (imageNorm * textNorm) : 0f;
            logits[i] = cosineSim * LogitScale;
        }

        // Numerically stable softmax: subtract max to prevent exp overflow
        float maxLogit = TensorPrimitives.Max(logits);
        var expValues = new float[textEmbeddings.Length];
        for (int i = 0; i < logits.Length; i++)
        {
            expValues[i] = MathF.Exp(logits[i] - maxLogit);
        }

        float sumExp = TensorPrimitives.Sum(expValues);
        var probabilities = new float[textEmbeddings.Length];
        for (int i = 0; i < expValues.Length; i++)
        {
            probabilities[i] = expValues[i] / sumExp;
        }

        return probabilities;
    }
}
