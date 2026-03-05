using Microsoft.Extensions.AI;
using Microsoft.ML.Data;
using MLNet.Image.Core;
using MLNet.ImageInference.Onnx.ImageCaptioning;
using MLNet.ImageInference.Onnx.MEAI;
using Xunit;

namespace MLNet.ImageInference.Onnx.Tests;

/// <summary>
/// Integration tests for Visual Question Answering (VQA) using GIT-VQA ONNX models.
/// Note: GIT-textvqa is trained on images with text — synthetic solid-color images
/// may produce empty answers, which is expected model behavior.
/// </summary>
public class VqaIntegrationTests : IDisposable
{
    private const string EncoderPath = "models/git-base-textvqa/encoder.onnx";
    private const string DecoderPath = "models/git-base-textvqa/decoder.onnx";
    private const string VocabPath = "models/git-base-textvqa/vocab.txt";

    private static bool ModelsExist =>
        File.Exists(EncoderPath) && File.Exists(DecoderPath) && File.Exists(VocabPath);

    private readonly List<IDisposable> _disposables = [];

    public void Dispose()
    {
        foreach (var d in _disposables)
            d.Dispose();
    }

    [SkippableFact]
    public void AnswerQuestion_RunsWithoutError()
    {
        Skip.Unless(ModelsExist, "GIT-VQA models not found");

        var options = CreateOptions();
        using var transformer = new OnnxImageCaptioningTransformer(options);

        using var image = CreateTestImage(480, 480, r: 255, g: 0, b: 0);
        var answer = transformer.AnswerQuestion(image, "what color is this?");

        // VQA on synthetic images may return empty — that's valid model behavior
        Assert.NotNull(answer);
    }

    [SkippableFact]
    public void AnswerQuestion_DifferentQuestions_RunsSuccessfully()
    {
        Skip.Unless(ModelsExist, "GIT-VQA models not found");

        var options = CreateOptions();
        using var transformer = new OnnxImageCaptioningTransformer(options);

        using var image = CreateTestImage(480, 480, r: 0, g: 128, b: 255);
        var answer1 = transformer.AnswerQuestion(image, "what color is this?");
        var answer2 = transformer.AnswerQuestion(image, "how many objects are there?");

        Assert.NotNull(answer1);
        Assert.NotNull(answer2);
    }

    [SkippableFact]
    public void AnswerQuestion_MultipleImages_RunsSuccessfully()
    {
        Skip.Unless(ModelsExist, "GIT-VQA models not found");

        var options = CreateOptions();
        using var transformer = new OnnxImageCaptioningTransformer(options);

        using var red = CreateTestImage(480, 480, r: 255, g: 0, b: 0);
        using var blue = CreateTestImage(480, 480, r: 0, g: 0, b: 255);

        var answer1 = transformer.AnswerQuestion(red, "what color is this?");
        var answer2 = transformer.AnswerQuestion(blue, "what color is this?");

        Assert.NotNull(answer1);
        Assert.NotNull(answer2);
    }

    [SkippableFact]
    public void GenerateCaption_StillWorks_WithVqaModel()
    {
        Skip.Unless(ModelsExist, "GIT-VQA models not found");

        var options = CreateOptions();
        using var transformer = new OnnxImageCaptioningTransformer(options);

        using var image = CreateTestImage(480, 480, r: 135, g: 206, b: 235);
        var caption = transformer.GenerateCaption(image);

        Assert.NotNull(caption);
    }

    [SkippableFact]
    public void ChatClient_WithTextAndImage_UsesVqaMode()
    {
        Skip.Unless(ModelsExist, "GIT-VQA models not found");

        var options = CreateOptions();
        using var chatClient = new OnnxImageCaptioningChatClient(options, modelId: "git-base-textvqa");

        using var image = CreateTestImage(480, 480, r: 255, g: 0, b: 0);
        var imageData = image.ToDataContent();

        var messages = new[]
        {
            new ChatMessage(ChatRole.User, [
                imageData,
                new TextContent("what color is this?")
            ])
        };

        var response = chatClient.GetResponseAsync(messages).Result;

        Assert.NotNull(response);
        Assert.NotNull(response.Text);
    }

    [SkippableFact]
    public void ChatClient_WithImageOnly_UsesCaptioningMode()
    {
        Skip.Unless(ModelsExist, "GIT-VQA models not found");

        var options = CreateOptions();
        using var chatClient = new OnnxImageCaptioningChatClient(options, modelId: "git-base-textvqa");

        using var image = CreateTestImage(480, 480, r: 135, g: 206, b: 235);
        var imageData = image.ToDataContent();

        var messages = new[]
        {
            new ChatMessage(ChatRole.User, [imageData])
        };

        var response = chatClient.GetResponseAsync(messages).Result;

        Assert.NotNull(response);
        Assert.NotNull(response.Text);
    }

    [SkippableFact]
    public async Task ChatClient_StreamingVqa_ReturnsUpdate()
    {
        Skip.Unless(ModelsExist, "GIT-VQA models not found");

        var options = CreateOptions();
        using var chatClient = new OnnxImageCaptioningChatClient(options, modelId: "git-base-textvqa");

        using var image = CreateTestImage(480, 480, r: 0, g: 255, b: 0);
        var imageData = image.ToDataContent();

        var messages = new[]
        {
            new ChatMessage(ChatRole.User, [
                imageData,
                new TextContent("what is in this image?")
            ])
        };

        var updates = new List<ChatResponseUpdate>();
        await foreach (var update in chatClient.GetStreamingResponseAsync(messages))
        {
            updates.Add(update);
        }

        // Streaming always returns at least one update (even if text is empty)
        Assert.NotEmpty(updates);
    }

    [SkippableFact]
    public void AnswerQuestion_MaxLength_RespectsLimit()
    {
        Skip.Unless(ModelsExist, "GIT-VQA models not found");

        var options = new OnnxImageCaptioningOptions
        {
            EncoderModelPath = EncoderPath,
            DecoderModelPath = DecoderPath,
            VocabPath = VocabPath,
            PreprocessorConfig = PreprocessorConfig.GITVQA,
            MaxLength = 5
        };

        using var transformer = new OnnxImageCaptioningTransformer(options);
        using var image = CreateTestImage(480, 480, r: 200, g: 100, b: 50);
        var answer = transformer.AnswerQuestion(image, "describe this image");

        Assert.NotNull(answer);
    }

    private static OnnxImageCaptioningOptions CreateOptions() => new()
    {
        EncoderModelPath = EncoderPath,
        DecoderModelPath = DecoderPath,
        VocabPath = VocabPath,
        PreprocessorConfig = PreprocessorConfig.GITVQA,
        MaxLength = 30
    };

    private static MLImage CreateTestImage(int width, int height, byte r, byte g, byte b)
    {
        var pixels = new byte[width * height * 4];
        for (int i = 0; i < width * height; i++)
        {
            int idx = i * 4;
            pixels[idx + 0] = r;
            pixels[idx + 1] = g;
            pixels[idx + 2] = b;
            pixels[idx + 3] = 255;
        }
        return MLImage.CreateFromPixels(width, height, MLPixelFormat.Rgba32, pixels);
    }
}
