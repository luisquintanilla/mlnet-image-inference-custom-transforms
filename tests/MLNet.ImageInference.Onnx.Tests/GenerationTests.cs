using Microsoft.ML.Data;
using MLNet.Image.Core;
using MLNet.ImageGeneration.OnnxGenAI;
using Xunit;

namespace MLNet.ImageInference.Onnx.Tests;

/// <summary>
/// Unit tests for the image generation pipeline components (scheduler, options).
/// These tests validate logic that does not require a real Stable Diffusion model.
/// </summary>
public class GenerationTests
{
    // --- EulerDiscreteScheduler tests ---

    [Theory]
    [InlineData(1)]
    [InlineData(4)]
    [InlineData(20)]
    [InlineData(50)]
    public void EulerDiscreteScheduler_SetTimesteps_CreatesCorrectCount(int steps)
    {
        var scheduler = new EulerDiscreteScheduler();

        scheduler.SetTimesteps(steps);

        Assert.Equal(steps, scheduler.Timesteps.Length);
        // Timesteps should be monotonically decreasing (from high noise to low noise)
        for (int i = 1; i < scheduler.Timesteps.Length; i++)
            Assert.True(scheduler.Timesteps[i] < scheduler.Timesteps[i - 1],
                $"Timesteps should be decreasing: [{i - 1}]={scheduler.Timesteps[i - 1]}, [{i}]={scheduler.Timesteps[i]}");
    }

    [Fact]
    public void EulerDiscreteScheduler_Step_ProducesOutput()
    {
        var scheduler = new EulerDiscreteScheduler();
        scheduler.SetTimesteps(20);

        var modelOutput = new float[] { 0.1f, -0.2f, 0.3f, -0.4f };
        var sample = new float[] { 1.0f, 2.0f, 3.0f, 4.0f };

        var result = scheduler.Step(modelOutput, stepIndex: 0, sample);

        Assert.Equal(sample.Length, result.Length);
        // Result should differ from input (scheduler applies a denoising step)
        Assert.False(result.SequenceEqual(sample), "Step output should differ from input sample");
    }

    [Fact]
    public void EulerDiscreteScheduler_ScaleModelInput_ProducesOutput()
    {
        var scheduler = new EulerDiscreteScheduler();
        scheduler.SetTimesteps(20);

        var sample = new float[] { 1.0f, 2.0f, 3.0f, 4.0f };

        var result = scheduler.ScaleModelInput(sample, stepIndex: 0);

        Assert.Equal(sample.Length, result.Length);
        // Scaled values should be smaller in magnitude (divided by sqrt(sigma^2 + 1))
        for (int i = 0; i < sample.Length; i++)
            Assert.True(MathF.Abs(result[i]) <= MathF.Abs(sample[i]),
                $"Scaled value [{i}] ({result[i]}) should not exceed original ({sample[i]})");
    }

    // --- OnnxImageGenerationOptions tests ---

    [Fact]
    public void Options_DefaultValues()
    {
        var options = new OnnxImageGenerationOptions
        {
            ModelDirectory = "dummy"
        };

        Assert.Equal(20, options.NumInferenceSteps);
        Assert.Equal(7.5f, options.GuidanceScale);
        Assert.Equal(512, options.Width);
        Assert.Equal(512, options.Height);
        Assert.Null(options.Seed);
        Assert.Null(options.NegativePrompt);
    }

    [Fact]
    public void Options_CustomValues_ArePreserved()
    {
        var options = new OnnxImageGenerationOptions
        {
            ModelDirectory = "/path/to/model",
            NumInferenceSteps = 50,
            GuidanceScale = 12.0f,
            Width = 768,
            Height = 768,
            Seed = 42,
            NegativePrompt = "blurry"
        };

        Assert.Equal("/path/to/model", options.ModelDirectory);
        Assert.Equal(50, options.NumInferenceSteps);
        Assert.Equal(12.0f, options.GuidanceScale);
        Assert.Equal(768, options.Width);
        Assert.Equal(768, options.Height);
        Assert.Equal(42, options.Seed);
        Assert.Equal("blurry", options.NegativePrompt);
    }

    // --- Integration test (model required) ---

    private const string ModelDir = "models/stable-diffusion";
    private static bool ModelExists =>
        Directory.Exists(ModelDir) &&
        File.Exists(Path.Combine(ModelDir, "text_encoder", "model.onnx")) &&
        File.Exists(Path.Combine(ModelDir, "unet", "model.onnx")) &&
        File.Exists(Path.Combine(ModelDir, "vae_decoder", "model.onnx"));

    [SkippableFact]
    public void Generate_WithRealModel_ProducesImage()
    {
        Skip.Unless(ModelExists, "Stable Diffusion model not available at " + ModelDir);

        var options = new OnnxImageGenerationOptions
        {
            ModelDirectory = ModelDir,
            NumInferenceSteps = 1,
            Width = 256,
            Height = 256,
            Seed = 42
        };

        using var transformer = new OnnxImageGenerationTransformer(options);
        using var image = transformer.Generate("a white cat");

        Assert.NotNull(image);
        Assert.Equal(256, image.Width);
        Assert.Equal(256, image.Height);
    }

    private const string VocabPath = "models/clip/vocab.json";
    private const string MergesPath = "models/clip/merges.txt";
    private static bool TokenizerExists => File.Exists(VocabPath) && File.Exists(MergesPath);

    [SkippableFact]
    public void Generate_WithClipTokenizer_DifferentPromptsDifferentImages()
    {
        Skip.Unless(ModelExists && TokenizerExists, "SD model or CLIP tokenizer not available");

        var options = new OnnxImageGenerationOptions
        {
            ModelDirectory = ModelDir,
            VocabPath = VocabPath,
            MergesPath = MergesPath,
            NumInferenceSteps = 1,
            Width = 256,
            Height = 256,
            Seed = 42
        };

        using var transformer = new OnnxImageGenerationTransformer(options);

        using var image1 = transformer.Generate("a cat sitting on a beach");
        using var image2 = transformer.Generate("a mountain covered in snow");

        Assert.NotNull(image1);
        Assert.NotNull(image2);
        Assert.Equal(256, image1.Width);
        Assert.Equal(256, image2.Width);

        // With real tokenizer, different prompts should produce different pixel data
        // Extract a sample of pixels and verify they differ
        var pixels1 = GetPixelSample(image1);
        var pixels2 = GetPixelSample(image2);
        Assert.False(pixels1.SequenceEqual(pixels2),
            "Different prompts with real tokenizer should produce different images");
    }

    private static byte[] GetPixelSample(MLImage image)
    {
        // Sample center region bytes from the raw pixel data
        var allPixels = image.Pixels;
        int bytesPerPixel = 4; // RGBA32
        int stride = image.Width * bytesPerPixel;
        int cx = image.Width / 2;
        int cy = image.Height / 2;
        int startOffset = cy * stride + cx * bytesPerPixel;
        var sample = new byte[64];
        for (int i = 0; i < sample.Length && startOffset + i < allPixels.Length; i++)
            sample[i] = allPixels[startOffset + i];
        return sample;
    }
}
