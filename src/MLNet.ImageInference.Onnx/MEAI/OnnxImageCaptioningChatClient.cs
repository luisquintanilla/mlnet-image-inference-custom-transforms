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
    /// Generate a caption or answer a question about images in the chat messages.
    /// If the message contains both image and text, uses VQA mode (answers the question).
    /// If the message contains only an image, uses captioning mode.
    /// </summary>
    public Task<ChatResponse> GetResponseAsync(
        IEnumerable<ChatMessage> chatMessages,
        ChatOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();

        var (image, question) = ExtractImageAndQuestion(chatMessages);
        if (image is null)
        {
            return Task.FromResult(new ChatResponse(
                new ChatMessage(ChatRole.Assistant, "No image found in the message. Please provide an image.")));
        }

        try
        {
            string result = question is not null
                ? _transformer.AnswerQuestion(image, question, cancellationToken)
                : _transformer.GenerateCaption(image, cancellationToken);

            return Task.FromResult(new ChatResponse(
                new ChatMessage(ChatRole.Assistant, result)));
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
    /// Extract the first image and optional question text from the last user message.
    /// When both image and text are present, the text is used as a VQA question.
    /// </summary>
    private static (MLImage? Image, string? Question) ExtractImageAndQuestion(IEnumerable<ChatMessage> messages)
    {
        // Find the last user message
        ChatMessage? userMessage = null;
        foreach (var msg in messages)
        {
            if (msg.Role == ChatRole.User)
                userMessage = msg;
        }

        if (userMessage?.Contents is null)
            return (null, null);

        MLImage? image = null;
        string? question = null;

        foreach (var content in userMessage.Contents)
        {
            if (image is null &&
                content is DataContent dataContent &&
                dataContent.MediaType?.StartsWith("image/", StringComparison.OrdinalIgnoreCase) == true)
            {
                image = dataContent.ToMLImage();
            }
            else if (question is null && content is TextContent textContent &&
                     !string.IsNullOrWhiteSpace(textContent.Text))
            {
                question = textContent.Text;
            }
        }

        return (image, question);
    }
}
