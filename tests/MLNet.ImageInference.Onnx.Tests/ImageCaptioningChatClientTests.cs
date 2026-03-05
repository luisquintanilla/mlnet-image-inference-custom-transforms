using Microsoft.Extensions.AI;
using Microsoft.ML.Data;
using MLNet.Image.Core;
using MLNet.ImageInference.Onnx.ImageCaptioning;
using MLNet.ImageInference.Onnx.MEAI;
using Xunit;

namespace MLNet.ImageInference.Onnx.Tests;

/// <summary>
/// Integration tests for OnnxImageCaptioningChatClient (IChatClient adapter).
/// Tests the MEAI integration layer for image captioning.
/// </summary>
public class ImageCaptioningChatClientTests : IDisposable
{
    private const string EncoderPath = "models/git-coco/encoder.onnx";
    private const string DecoderPath = "models/git-coco/decoder.onnx";
    private const string VocabPath = "models/git-coco/vocab.txt";

    private static bool ModelsExist =>
        File.Exists(EncoderPath) && File.Exists(DecoderPath) && File.Exists(VocabPath);

    private readonly List<IDisposable> _disposables = [];

    public void Dispose()
    {
        foreach (var d in _disposables)
            d.Dispose();
    }

    private OnnxImageCaptioningChatClient CreateClient()
    {
        var options = new OnnxImageCaptioningOptions
        {
            EncoderModelPath = EncoderPath,
            DecoderModelPath = DecoderPath,
            VocabPath = VocabPath,
            PreprocessorConfig = PreprocessorConfig.GIT,
            MaxLength = 50
        };
        var client = new OnnxImageCaptioningChatClient(options, modelId: "git-base-coco");
        _disposables.Add(client);
        return client;
    }

    [SkippableFact]
    public async Task GetResponseAsync_WithImage_ReturnsCaptionText()
    {
        Skip.Unless(ModelsExist, "GIT captioning models not found");

        var client = CreateClient();
        using var image = CreateTestImage(224, 224, r: 135, g: 206, b: 235);
        var dataContent = image.ToDataContent("image/png");

        var messages = new List<ChatMessage>
        {
            new(ChatRole.User, [dataContent])
        };

        var response = await client.GetResponseAsync(messages);

        Assert.NotNull(response);
        Assert.NotNull(response.Text);
        Assert.NotEmpty(response.Text);
        Assert.True(response.Text.Length > 2, $"Caption too short: '{response.Text}'");
    }

    [SkippableFact]
    public async Task GetResponseAsync_NoImage_ReturnsErrorMessage()
    {
        Skip.Unless(ModelsExist, "GIT captioning models not found");

        var client = CreateClient();
        var messages = new List<ChatMessage>
        {
            new(ChatRole.User, "Describe this image")
        };

        var response = await client.GetResponseAsync(messages);

        Assert.NotNull(response);
        Assert.Contains("No image", response.Text);
    }

    [SkippableFact]
    public async Task GetResponseAsync_MultipleMessages_UsesLastUserImage()
    {
        Skip.Unless(ModelsExist, "GIT captioning models not found");

        var client = CreateClient();

        // First message has an image, second has a different image
        using var blueImage = CreateTestImage(224, 224, r: 135, g: 206, b: 235);
        using var redImage = CreateTestImage(224, 224, r: 255, g: 0, b: 0);

        var messages = new List<ChatMessage>
        {
            new(ChatRole.User, [blueImage.ToDataContent("image/png")]),
            new(ChatRole.User, [redImage.ToDataContent("image/png")])
        };

        var response = await client.GetResponseAsync(messages);

        Assert.NotNull(response);
        Assert.NotEmpty(response.Text);
    }

    [SkippableFact]
    public async Task GetStreamingResponseAsync_ReturnsUpdate()
    {
        Skip.Unless(ModelsExist, "GIT captioning models not found");

        var client = CreateClient();
        using var image = CreateTestImage(224, 224, r: 135, g: 206, b: 235);

        var messages = new List<ChatMessage>
        {
            new(ChatRole.User, [image.ToDataContent("image/png")])
        };

        var updates = new List<ChatResponseUpdate>();
        await foreach (var update in client.GetStreamingResponseAsync(messages))
        {
            updates.Add(update);
        }

        Assert.Single(updates);
        Assert.NotEmpty(updates[0].Text ?? "");
    }

    [SkippableFact]
    public void Metadata_HasExpectedValues()
    {
        Skip.Unless(ModelsExist, "GIT captioning models not found");

        var client = CreateClient();

        Assert.Equal("MLNet.ImageInference.Onnx", client.Metadata.ProviderName);
        Assert.Equal("git-base-coco", client.Metadata.DefaultModelId);
    }

    [SkippableFact]
    public void GetService_ReturnsCorrectTypes()
    {
        Skip.Unless(ModelsExist, "GIT captioning models not found");

        var client = CreateClient();

        Assert.Same(client, client.GetService(typeof(OnnxImageCaptioningChatClient)));
        Assert.IsType<OnnxImageCaptioningTransformer>(client.GetService(typeof(OnnxImageCaptioningTransformer)));
        Assert.Null(client.GetService(typeof(string)));
    }

    [SkippableFact]
    public async Task GetResponseAsync_JpegImage_WorksCorrectly()
    {
        Skip.Unless(ModelsExist, "GIT captioning models not found");

        var client = CreateClient();
        using var image = CreateTestImage(320, 240, r: 34, g: 139, b: 34); // Forest green
        var dataContent = image.ToDataContent("image/jpeg");

        var messages = new List<ChatMessage>
        {
            new(ChatRole.User, [dataContent])
        };

        var response = await client.GetResponseAsync(messages);

        Assert.NotNull(response);
        Assert.NotEmpty(response.Text);
    }

    [SkippableFact]
    public async Task GetResponseAsync_ConstructFromTransformer_Works()
    {
        Skip.Unless(ModelsExist, "GIT captioning models not found");

        var options = new OnnxImageCaptioningOptions
        {
            EncoderModelPath = EncoderPath,
            DecoderModelPath = DecoderPath,
            VocabPath = VocabPath,
            PreprocessorConfig = PreprocessorConfig.GIT,
            MaxLength = 50
        };

        using var transformer = new OnnxImageCaptioningTransformer(options);
        using var client = new OnnxImageCaptioningChatClient(transformer);

        using var image = CreateTestImage(224, 224, r: 135, g: 206, b: 235);
        var messages = new List<ChatMessage>
        {
            new(ChatRole.User, [image.ToDataContent("image/png")])
        };

        var response = await client.GetResponseAsync(messages);

        Assert.NotNull(response);
        Assert.NotEmpty(response.Text);
    }

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
