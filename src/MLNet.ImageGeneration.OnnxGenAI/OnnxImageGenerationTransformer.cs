using System.Diagnostics;
using Microsoft.Extensions.Logging;
using Microsoft.ML.Data;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using MLNet.Image.Tokenizers;

namespace MLNet.ImageGeneration.OnnxGenAI;

/// <summary>
/// Transformer for text-to-image generation using ONNX Stable Diffusion models.
/// Runs the full pipeline: CLIP text encoder → UNet denoising loop → VAE decoder → image.
/// Uses Microsoft.ML.OnnxRuntime InferenceSession for each pipeline stage.
/// </summary>
public sealed class OnnxImageGenerationTransformer : IDisposable
{
    private readonly OnnxImageGenerationOptions _options;
    private readonly InferenceSession _textEncoder;
    private readonly InferenceSession _unet;
    private readonly InferenceSession _vaeDecoder;
    private readonly EulerDiscreteScheduler _scheduler;
    private readonly ClipTokenizer? _tokenizer;
    private readonly ILogger _logger;
    private bool _disposed;

    public OnnxImageGenerationTransformer(OnnxImageGenerationOptions options)
    {
        _options = options ?? throw new ArgumentNullException(nameof(options));
        _logger = (ILogger?)options.Logger ?? Microsoft.Extensions.Logging.Abstractions.NullLogger.Instance;

        var sw = Stopwatch.StartNew();
        var sessionOptions = CreateSessionOptions(options);

        var textEncoderPath = Path.Combine(options.ModelDirectory, "text_encoder", "model.onnx");
        var unetPath = Path.Combine(options.ModelDirectory, "unet", "model.onnx");
        var vaeDecoderPath = Path.Combine(options.ModelDirectory, "vae_decoder", "model.onnx");

        ValidateModelFile(textEncoderPath, "text_encoder");
        ValidateModelFile(unetPath, "unet");
        ValidateModelFile(vaeDecoderPath, "vae_decoder");

        _textEncoder = new InferenceSession(textEncoderPath, sessionOptions);
        _unet = new InferenceSession(unetPath, sessionOptions);
        _vaeDecoder = new InferenceSession(vaeDecoderPath, sessionOptions);
        _scheduler = new EulerDiscreteScheduler();

        if (options.VocabPath is not null && options.MergesPath is not null)
            _tokenizer = ClipTokenizer.Create(options.VocabPath, options.MergesPath);

        _logger.LogInformation("Stable Diffusion models loaded from {ModelDirectory} in {ElapsedMs}ms",
            options.ModelDirectory, sw.ElapsedMilliseconds);
    }

    private static SessionOptions CreateSessionOptions(OnnxImageGenerationOptions options)
    {
        var sessionOptions = new SessionOptions();

        if (options.ExecutionProvider == OnnxExecutionProvider.CPU)
            return sessionOptions;

        try
        {
            switch (options.ExecutionProvider)
            {
                case OnnxExecutionProvider.CUDA:
                    sessionOptions.AppendExecutionProvider_CUDA(options.GpuDeviceId);
                    break;
                case OnnxExecutionProvider.DirectML:
                    sessionOptions.AppendExecutionProvider_DML(options.GpuDeviceId);
                    break;
                case OnnxExecutionProvider.TensorRT:
                    sessionOptions.AppendExecutionProvider_Tensorrt(options.GpuDeviceId);
                    break;
            }
        }
        catch (OnnxRuntimeException ex) when (options.FallbackToCpu)
        {
            options.Logger?.LogWarning(ex, "{Provider} initialization failed, falling back to CPU", options.ExecutionProvider);
            sessionOptions = new SessionOptions();
        }

        return sessionOptions;
    }

    private static void ValidateModelFile(string path, string componentName)
    {
        if (!File.Exists(path))
            throw new FileNotFoundException(
                $"Stable Diffusion {componentName} model not found: '{path}'. " +
                "Ensure the model has been exported and the ModelDirectory is correct.",
                path);
    }

    /// <summary>
    /// Generate an image from a text prompt using the Stable Diffusion pipeline.
    /// </summary>
    /// <param name="prompt">The text prompt describing the desired image.</param>
    /// <param name="seed">Optional random seed for reproducibility.</param>
    /// <returns>Generated image as an MLImage.</returns>
    public MLImage Generate(string prompt, int? seed = null, CancellationToken cancellationToken = default)
    {
        var sw = Stopwatch.StartNew();
        _logger.LogInformation("Generating {Width}x{Height} image with {Steps} steps, prompt: \"{Prompt}\"",
            _options.Width, _options.Height, _options.NumInferenceSteps, prompt);

        int height = _options.Height;
        int width = _options.Width;
        int steps = _options.NumInferenceSteps;
        float guidanceScale = _options.GuidanceScale;
        int latentH = height / 8;
        int latentW = width / 8;

        // 1. Tokenize with CLIP BPE tokenizer (or fallback to simple SOT+EOT)
        var tokenIds = Tokenize(prompt);
        var uncondTokenIds = Tokenize(_options.NegativePrompt ?? "");

        // 2. Text encode
        var condEmbeddings = TextEncode(tokenIds);
        var uncondEmbeddings = TextEncode(uncondTokenIds);

        // Concatenate for classifier-free guidance: [2, 77, embedding_dim]
        var textEmbeddings = ConcatEmbeddings(uncondEmbeddings, condEmbeddings);

        // 3. Init latents with random noise
        var random = seed.HasValue ? new Random(seed.Value) : new Random();
        var latents = new float[1 * 4 * latentH * latentW];
        for (int i = 0; i < latents.Length; i++)
        {
            // Box-Muller transform for Gaussian noise
            float u1 = 1.0f - (float)random.NextDouble();
            float u2 = (float)random.NextDouble();
            latents[i] = MathF.Sqrt(-2.0f * MathF.Log(u1)) * MathF.Cos(2.0f * MathF.PI * u2);
        }

        // 4. Setup scheduler and scale initial noise
        _scheduler.SetTimesteps(steps);
        for (int i = 0; i < latents.Length; i++)
            latents[i] *= _scheduler.InitNoiseSigma;

        // 5. Denoise loop
        for (int step = 0; step < steps; step++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            var scaledLatents = _scheduler.ScaleModelInput(latents, step);

            // Duplicate for CFG: [2, 4, H/8, W/8]
            var latentInput = new float[2 * 4 * latentH * latentW];
            Array.Copy(scaledLatents, 0, latentInput, 0, scaledLatents.Length);
            Array.Copy(scaledLatents, 0, latentInput, scaledLatents.Length, scaledLatents.Length);

            var noisePred = UNetPredict(latentInput, _scheduler.Timesteps[step], textEmbeddings, latentH, latentW);

            // Apply classifier-free guidance
            int halfLen = noisePred.Length / 2;
            var guidedNoise = new float[halfLen];
            for (int i = 0; i < halfLen; i++)
            {
                float uncondPred = noisePred[i];
                float condPred = noisePred[halfLen + i];
                guidedNoise[i] = uncondPred + guidanceScale * (condPred - uncondPred);
            }

            latents = _scheduler.Step(guidedNoise, step, latents);
            _logger.LogDebug("Denoising step {Step}/{TotalSteps} complete", step + 1, steps);
        }

        // 6. Scale latents for VAE (1/0.18215 scaling factor)
        for (int i = 0; i < latents.Length; i++)
            latents[i] *= (1.0f / 0.18215f);

        // 7. VAE decode
        var imageData = VaeDecode(latents, latentH, latentW);

        // 8. Convert to MLImage
        var result = ConvertToMLImage(imageData, height, width);
        _logger.LogInformation("Image generation completed in {ElapsedMs}ms", sw.ElapsedMilliseconds);
        return result;
    }

    /// <summary>
    /// Tokenize text using CLIP BPE tokenizer if available, otherwise fall back to simple SOT+EOT.
    /// Returns long[] of length 77 (CLIP context length).
    /// </summary>
    private long[] Tokenize(string text)
    {
        if (_tokenizer is not null)
        {
            var intTokens = _tokenizer.Encode(text);
            var longTokens = new long[intTokens.Length];
            for (int i = 0; i < intTokens.Length; i++)
                longTokens[i] = intTokens[i];
            return longTokens;
        }

        // Fallback: SOT + EOT padded to 77 (no real text encoding)
        var tokens = new long[77];
        tokens[0] = 49406; // SOT
        tokens[1] = 49407; // EOT
        for (int i = 2; i < 77; i++)
            tokens[i] = 49407;
        return tokens;
    }

    private float[] TextEncode(long[] tokenIds)
    {
        var inputTensor = new DenseTensor<long>(tokenIds, [1, 77]);
        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor(_textEncoder.InputMetadata.First().Key, inputTensor)
        };
        using var results = _textEncoder.Run(inputs);
        return results.First().AsTensor<float>().ToArray();
    }

    private static float[] ConcatEmbeddings(float[] uncond, float[] cond)
    {
        var result = new float[uncond.Length + cond.Length];
        Array.Copy(uncond, 0, result, 0, uncond.Length);
        Array.Copy(cond, 0, result, uncond.Length, cond.Length);
        return result;
    }

    private float[] UNetPredict(float[] latentInput, float timestep, float[] textEmbeddings, int latentH, int latentW)
    {
        var sampleTensor = new DenseTensor<float>(latentInput, [2, 4, latentH, latentW]);
        var timestepTensor = new DenseTensor<float>(new[] { timestep }, new[] { 1 });
        var embedTensor = new DenseTensor<float>(textEmbeddings, [2, 77, textEmbeddings.Length / (2 * 77)]);

        // Discover input names from model metadata
        var inputNames = _unet.InputMetadata.Keys.ToArray();
        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor(inputNames[0], sampleTensor),
            NamedOnnxValue.CreateFromTensor(inputNames[1], timestepTensor),
            NamedOnnxValue.CreateFromTensor(inputNames[2], embedTensor)
        };

        using var results = _unet.Run(inputs);
        return results.First().AsTensor<float>().ToArray();
    }

    private float[] VaeDecode(float[] latents, int latentH, int latentW)
    {
        var latentTensor = new DenseTensor<float>(latents, [1, 4, latentH, latentW]);
        var inputName = _vaeDecoder.InputMetadata.First().Key;
        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor(inputName, latentTensor)
        };
        using var results = _vaeDecoder.Run(inputs);
        return results.First().AsTensor<float>().ToArray();
    }

    /// <summary>
    /// Convert VAE output [1, 3, H, W] in [-1, 1] range to MLImage (RGBA32).
    /// </summary>
    private static MLImage ConvertToMLImage(float[] imageData, int height, int width)
    {
        var pixels = new byte[height * width * 4];
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                int pixelIdx = (y * width + x) * 4;
                for (int c = 0; c < 3; c++)
                {
                    float val = imageData[c * height * width + y * width + x];
                    val = (val / 2.0f + 0.5f); // [-1,1] → [0,1]
                    val = Math.Clamp(val, 0, 1);
                    pixels[pixelIdx + c] = (byte)(val * 255);
                }
                pixels[pixelIdx + 3] = 255; // Alpha
            }
        }
        return MLImage.CreateFromPixels(width, height, MLPixelFormat.Rgba32, pixels);
    }

    public void Dispose()
    {
        if (!_disposed)
        {
            _textEncoder?.Dispose();
            _unet?.Dispose();
            _vaeDecoder?.Dispose();
            _disposed = true;
        }
    }
}
