using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Microsoft.ML.Tokenizers;
using MLNet.Image.Core;
using MLNet.ImageInference.Onnx.Shared;

namespace MLNet.ImageInference.Onnx.ImageCaptioning;

/// <summary>
/// Transformer that performs image captioning using GIT (Generative Image-to-text Transformer):
/// MLImage → preprocess → vision encoder → autoregressive text decoder → caption string.
/// Composes ImagePreprocessingTransformer for vision preprocessing, plus separate ONNX sessions
/// for the vision encoder and text decoder.
/// </summary>
public sealed class OnnxImageCaptioningTransformer : ITransformer, IDisposable
{
    private readonly OnnxImageCaptioningOptions _options;
    private readonly ImagePreprocessingTransformer _preprocessor;
    private readonly OnnxSessionPool _encoderPool;
    private readonly OnnxSessionPool _decoderPool;
    private readonly ModelMetadataDiscovery.ModelMetadata _encoderMetadata;
    private readonly ModelMetadataDiscovery.ModelMetadata _decoderMetadata;
    private readonly BertTokenizer _tokenizer;

    public bool IsRowToRowMapper => true;

    public OnnxImageCaptioningTransformer(OnnxImageCaptioningOptions options)
        : this(options,
              new ImagePreprocessingTransformer(new ImagePreprocessingOptions
              {
                  InputColumnName = options.InputColumnName,
                  PreprocessorConfig = options.PreprocessorConfig
              }))
    {
    }

    internal OnnxImageCaptioningTransformer(
        OnnxImageCaptioningOptions options,
        ImagePreprocessingTransformer preprocessor)
    {
        _options = options;
        _preprocessor = preprocessor;

        _encoderPool = new OnnxSessionPool(options.EncoderModelPath);
        _encoderMetadata = ModelMetadataDiscovery.Discover(_encoderPool.Session);

        _decoderPool = new OnnxSessionPool(options.DecoderModelPath);
        _decoderMetadata = ModelMetadataDiscovery.Discover(_decoderPool.Session);

        _tokenizer = BertTokenizer.Create(options.VocabPath, new BertOptions
        {
            LowerCaseBeforeTokenization = true
        });
    }

    /// <summary>
    /// Generate a caption for a single image.
    /// </summary>
    public string GenerateCaption(MLImage image)
    {
        // Stage 1: Preprocess image
        var tensor = _preprocessor.Preprocess(image);

        // Stage 2: Encode image → visual features
        var visualFeatures = EncodeImage(tensor);

        // Stage 3: Autoregressive text generation
        var tokenIds = GenerateTokens(visualFeatures);

        // Stage 4: Decode tokens to text
        return _tokenizer.Decode(tokenIds);
    }

    /// <summary>
    /// Generate captions for a batch of images.
    /// </summary>
    public string[] GenerateCaptionBatch(IReadOnlyList<MLImage> images)
    {
        if (images == null || images.Count == 0)
            return [];

        // Captioning is inherently sequential per image (autoregressive decoding),
        // so we process each image individually even if the encoder supports batching.
        var captions = new string[images.Count];
        for (int i = 0; i < images.Count; i++)
            captions[i] = GenerateCaption(images[i]);
        return captions;
    }

    /// <summary>
    /// Run the vision encoder on a preprocessed tensor to get visual features.
    /// </summary>
    private DenseTensor<float> EncodeImage(float[] preprocessedTensor)
    {
        int h = _options.PreprocessorConfig.ImageSize.Height;
        int w = _options.PreprocessorConfig.ImageSize.Width;

        var inputTensor = new DenseTensor<float>(preprocessedTensor, [1, 3, h, w]);
        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor(_encoderMetadata.InputNames[0], inputTensor)
        };

        using var results = _encoderPool.Session.Run(inputs);
        var output = results.First().AsTensor<float>();

        // Copy to a new tensor (results will be disposed)
        var dims = output.Dimensions.ToArray();
        var copy = new DenseTensor<float>(dims);
        var outputArray = output.ToArray();
        outputArray.CopyTo(copy.Buffer.Span);
        return copy;
    }

    /// <summary>
    /// Autoregressive greedy decode: generate tokens one at a time until EOS or max length.
    /// </summary>
    private List<int> GenerateTokens(DenseTensor<float> visualFeatures)
    {
        var generatedIds = new List<int>();
        int currentToken = _options.BosTokenId;

        for (int step = 0; step < _options.MaxLength; step++)
        {
            // Build input_ids tensor: [1, seq_len]
            int seqLen = step + 1;
            var inputIdsTensor = new DenseTensor<long>([1, seqLen]);
            inputIdsTensor[0, 0] = _options.BosTokenId;
            for (int i = 0; i < generatedIds.Count; i++)
                inputIdsTensor[0, i + 1] = generatedIds[i];

            // Run decoder: input_ids + visual_features → logits
            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor(_decoderMetadata.InputNames[0], inputIdsTensor),
                NamedOnnxValue.CreateFromTensor(_decoderMetadata.InputNames[1], visualFeatures)
            };

            using var results = _decoderPool.Session.Run(inputs);
            var logits = results.First().AsTensor<float>();

            // Get logits at the last position → argmax for greedy decoding
            int vocabSize = (int)logits.Dimensions[^1];
            int lastPos = seqLen - 1;

            int bestToken = 0;
            float bestScore = float.MinValue;
            for (int v = 0; v < vocabSize; v++)
            {
                float score = logits[0, lastPos, v];
                if (score > bestScore)
                {
                    bestScore = score;
                    bestToken = v;
                }
            }

            if (bestToken == _options.EosTokenId)
                break;

            generatedIds.Add(bestToken);
        }

        return generatedIds;
    }

    internal OnnxImageCaptioningOptions Options => _options;
    internal ImagePreprocessingTransformer Preprocessor => _preprocessor;

    public IDataView Transform(IDataView input)
        => new ImageCaptioningDataView(input, this);

    public DataViewSchema GetOutputSchema(DataViewSchema inputSchema)
    {
        var builder = new DataViewSchema.Builder();
        for (int i = 0; i < inputSchema.Count; i++)
            builder.AddColumn(inputSchema[i].Name, inputSchema[i].Type, inputSchema[i].Annotations);
        builder.AddColumn(_options.OutputColumnName, TextDataViewType.Instance);
        return builder.ToSchema();
    }

    public IRowToRowMapper GetRowToRowMapper(DataViewSchema inputSchema)
        => throw new InvalidOperationException("This transformer does not support row-to-row mapping.");

    void ICanSaveModel.Save(ModelSaveContext ctx)
        => throw new NotSupportedException("Use transformer-specific save/load instead of mlContext.Model.Save().");

    public void Dispose()
    {
        _encoderPool?.Dispose();
        _decoderPool?.Dispose();
    }
}
