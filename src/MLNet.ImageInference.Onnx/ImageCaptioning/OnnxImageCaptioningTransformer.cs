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
    public string GenerateCaption(MLImage image, CancellationToken cancellationToken = default)
    {
        // Stage 1: Preprocess image
        var tensor = _preprocessor.Preprocess(image);

        // Stage 2: Encode image → visual features
        var visualFeatures = EncodeImage(tensor);

        // Stage 3: Autoregressive text generation
        var tokenIds = GenerateTokens(visualFeatures, initialIds: [_options.BosTokenId], cancellationToken);

        // Stage 4: Decode tokens to text
        return _tokenizer.Decode(tokenIds);
    }

    /// <summary>
    /// Answer a question about an image (Visual Question Answering).
    /// The question is tokenized and used as initial decoder input before autoregressive generation.
    /// Requires a GIT-VQA model (e.g., microsoft/git-base-textvqa).
    /// </summary>
    public string AnswerQuestion(MLImage image, string question, CancellationToken cancellationToken = default)
    {
        // Stage 1: Preprocess image
        var tensor = _preprocessor.Preprocess(image);

        // Stage 2: Encode image → visual features
        var visualFeatures = EncodeImage(tensor);

        // Stage 3: Tokenize question → [CLS] question_tokens [SEP]
        var questionTokenIds = TokenizeQuestion(question);

        // Stage 4: Autoregressive generation starting after the question
        var answerTokenIds = GenerateTokens(visualFeatures, initialIds: questionTokenIds, cancellationToken);

        // Stage 5: Decode answer tokens to text
        return _tokenizer.Decode(answerTokenIds);
    }

    /// <summary>
    /// Generate captions for a batch of images.
    /// </summary>
    public string[] GenerateCaptionBatch(IReadOnlyList<MLImage> images, CancellationToken cancellationToken = default)
    {
        if (images == null || images.Count == 0)
            return [];

        var captions = new string[images.Count];
        for (int i = 0; i < images.Count; i++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            captions[i] = GenerateCaption(images[i], cancellationToken);
        }
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
    /// <param name="visualFeatures">Encoded visual features from the image.</param>
    /// <param name="initialIds">Initial token IDs (e.g., [CLS] for captioning, or [CLS]+question+[SEP] for VQA).</param>
    private List<int> GenerateTokens(DenseTensor<float> visualFeatures, int[] initialIds, CancellationToken cancellationToken)
    {
        var generatedIds = new List<int>();

        for (int step = 0; step < _options.MaxLength; step++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            // Build input_ids tensor: [1, initial_len + generated_len]
            int seqLen = initialIds.Length + generatedIds.Count;
            var inputIdsTensor = new DenseTensor<long>([1, seqLen]);
            for (int i = 0; i < initialIds.Length; i++)
                inputIdsTensor[0, i] = initialIds[i];
            for (int i = 0; i < generatedIds.Count; i++)
                inputIdsTensor[0, initialIds.Length + i] = generatedIds[i];

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

    /// <summary>
    /// Tokenize a question into [CLS] question_tokens [SEP] format for VQA.
    /// </summary>
    private int[] TokenizeQuestion(string question)
    {
        var encoded = _tokenizer.EncodeToIds(question, _options.MaxLength - 2, out _, out _);
        var ids = new int[encoded.Count + 2];
        ids[0] = _options.BosTokenId; // [CLS]
        for (int i = 0; i < encoded.Count; i++)
            ids[i + 1] = encoded[i];
        ids[encoded.Count + 1] = _options.EosTokenId; // [SEP]
        return ids;
    }

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
