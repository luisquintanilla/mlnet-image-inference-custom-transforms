using System.Runtime.CompilerServices;
using Microsoft.Extensions.AI;
using Microsoft.ML.Data;
using MLNet.Image.Core;
using MLNet.ImageInference.Onnx.ImageCaptioning;

namespace MLNet.ImageInference.Onnx.MEAI;

/// <summary>
/// IChatClient implementation for image captioning using local ONNX models.
/// Accepts ChatMessages containing image DataContent and returns captions as text.
///
/// Usage:
///   var client = new OnnxImageCaptioningChatClient(options);
///   var response = await client.GetResponseAsync([
///       new ChatMessage(ChatRole.User, [new DataContent(imageBytes, "image/jpeg")])
///   ]);
///   Console.WriteLine(response.Text); // "a blue sky with no clouds"
///
/// This makes local ONNX captioning interchangeable with cloud vision APIs
/// (OpenAI GPT-4o, Azure AI, etc.) through the MEAI IChatClient abstraction.
/// </summary>
public sealed class OnnxImageCaptioningChatClient : IChatClient
{
    private readonly OnnxImageCaptioningTransformer _transformer;
    private readonly bool _ownsTransformer;

    public ChatClientMetadata Metadata { get; }

    /// <summary>
    /// Create from an existing transformer instance.
    /// </summary>
    public OnnxImageCaptioningChatClient(
        OnnxImageCaptioningTransformer transformer,
        string? modelId = null)
    {
        ArgumentNullException.ThrowIfNull(transformer);
        _transformer = transformer;
        _ownsTransformer = false;

        Metadata = new ChatClientMetadata(
            providerName: "MLNet.ImageInference.Onnx",
            defaultModelId: modelId ?? "git-base-coco");
    }

    /// <summary>
    /// Create from options (constructs and owns the transformer).
    /// </summary>
    public OnnxImageCaptioningChatClient(OnnxImageCaptioningOptions options, string? modelId = null)
    {
        ArgumentNullException.ThrowIfNull(options);
        var estimator = new OnnxImageCaptioningEstimator(options);
        _transformer = estimator.Fit(null!);
        _ownsTransformer = true;

        Metadata = new ChatClientMetadata(
            providerName: "MLNet.ImageInference.Onnx",
            defaultModelId: modelId ?? "git-base-coco");
    }

    /// <summary>
    /// Generate a caption for images in the chat messages.
    /// Extracts the first image from the last user message and returns its caption.
    /// </summary>
    public Task<ChatResponse> GetResponseAsync(
        IEnumerable<ChatMessage> chatMessages,
        ChatOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();

        var image = ExtractImage(chatMessages);
        if (image is null)
        {
            return Task.FromResult(new ChatResponse(
                new ChatMessage(ChatRole.Assistant, "No image found in the message. Please provide an image.")));
        }

        try
        {
            var caption = _transformer.GenerateCaption(image);
            return Task.FromResult(new ChatResponse(
                new ChatMessage(ChatRole.Assistant, caption)));
        }
        finally
        {
            image.Dispose();
        }
    }

    /// <summary>
    /// Streaming is not supported for local captioning (greedy decode produces the full caption at once).
    /// Returns the complete caption as a single update.
    /// </summary>
    public async IAsyncEnumerable<ChatResponseUpdate> GetStreamingResponseAsync(
        IEnumerable<ChatMessage> chatMessages,
        ChatOptions? options = null,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        var response = await GetResponseAsync(chatMessages, options, cancellationToken);
        yield return new ChatResponseUpdate(ChatRole.Assistant, response.Text);
    }

    public object? GetService(Type serviceType, object? serviceKey = null)
    {
        if (serviceType == typeof(OnnxImageCaptioningChatClient))
            return this;
        if (serviceType == typeof(OnnxImageCaptioningTransformer))
            return _transformer;
        return null;
    }

    public void Dispose()
    {
        if (_ownsTransformer)
            _transformer?.Dispose();
    }

    /// <summary>
    /// Extract the first image from the last user message.
    /// Supports DataContent with image MIME types.
    /// </summary>
    private static MLImage? ExtractImage(IEnumerable<ChatMessage> messages)
    {
        // Find the last user message
        ChatMessage? userMessage = null;
        foreach (var msg in messages)
        {
            if (msg.Role == ChatRole.User)
                userMessage = msg;
        }

        if (userMessage?.Contents is null)
            return null;

        // Find the first image content
        foreach (var content in userMessage.Contents)
        {
            if (content is DataContent dataContent &&
                dataContent.MediaType?.StartsWith("image/", StringComparison.OrdinalIgnoreCase) == true)
            {
                return dataContent.ToMLImage();
            }
        }

        return null;
    }
}
