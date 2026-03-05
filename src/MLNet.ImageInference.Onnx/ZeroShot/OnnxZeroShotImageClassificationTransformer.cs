using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using MLNet.Image.Core;
using MLNet.Image.Tokenizers;
using MLNet.ImageInference.Onnx.Shared;
using System.Numerics.Tensors;

namespace MLNet.ImageInference.Onnx.ZeroShot;

/// <summary>
/// Transformer that performs zero-shot image classification using CLIP:
/// MLImage → vision encoder → image embedding, compared against pre-encoded text embeddings via cosine similarity.
/// Composes ImagePreprocessingTransformer + OnnxImageScoringTransformer for the vision side;
/// text encoding remains internal (CLIP-specific).
/// </summary>
public sealed class OnnxZeroShotImageClassificationTransformer : ITransformer, IDisposable
{
    private readonly OnnxZeroShotImageClassificationOptions _options;
    private readonly ImagePreprocessingTransformer _preprocessor;
    private readonly OnnxImageScoringTransformer _visionScorer;
    private readonly OnnxSessionPool _textSessionPool;
    private readonly float[][] _textEmbeddings;

    public bool IsRowToRowMapper => true;

    public OnnxZeroShotImageClassificationTransformer(OnnxZeroShotImageClassificationOptions options)
        : this(options,
              new ImagePreprocessingTransformer(new ImagePreprocessingOptions
              {
                  InputColumnName = options.InputColumnName,
                  PreprocessorConfig = options.PreprocessorConfig
              }),
              new OnnxImageScoringTransformer(new OnnxImageScoringOptions
              {
                  ModelPath = options.ImageModelPath,
                  ImageHeight = options.PreprocessorConfig.ImageSize.Height,
                  ImageWidth = options.PreprocessorConfig.ImageSize.Width,
                  BatchSize = options.BatchSize
              }))
    {
    }

    internal OnnxZeroShotImageClassificationTransformer(
        OnnxZeroShotImageClassificationOptions options,
        ImagePreprocessingTransformer preprocessor,
        OnnxImageScoringTransformer visionScorer)
    {
        _options = options;
        _preprocessor = preprocessor;
        _visionScorer = visionScorer;

        // Text side remains internal (CLIP-specific)
        _textSessionPool = new OnnxSessionPool(options.TextModelPath);
        var textMetadata = ModelMetadataDiscovery.Discover(_textSessionPool.Session);

        var tokenizer = ClipTokenizer.Create(options.VocabPath, options.MergesPath);
        _textEmbeddings = EncodeTexts(tokenizer, textMetadata, options.CandidateLabels);
    }

    private float[][] EncodeTexts(
        ClipTokenizer tokenizer,
        ModelMetadataDiscovery.ModelMetadata textMetadata,
        string[] labels)
    {
        var embeddings = new float[labels.Length][];
        for (int i = 0; i < labels.Length; i++)
        {
            embeddings[i] = EncodeText(tokenizer, textMetadata, labels[i]);
        }
        return embeddings;
    }

    private float[] EncodeText(
        ClipTokenizer tokenizer,
        ModelMetadataDiscovery.ModelMetadata textMetadata,
        string text)
    {
        int[] tokenIds = tokenizer.Encode(text);
        int[] attentionMask = tokenizer.CreateAttentionMask(tokenIds);
        int contextLength = tokenIds.Length; // 77 for CLIP

        // Create int64 tensors as required by CLIP text encoder
        var inputIdsTensor = new DenseTensor<long>(new[] { 1, contextLength });
        var attentionMaskTensor = new DenseTensor<long>(new[] { 1, contextLength });

        for (int i = 0; i < contextLength; i++)
        {
            inputIdsTensor[0, i] = tokenIds[i];
            attentionMaskTensor[0, i] = attentionMask[i];
        }

        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor(textMetadata.InputNames[0], inputIdsTensor),
            NamedOnnxValue.CreateFromTensor(textMetadata.InputNames[1], attentionMaskTensor)
        };

        using var results = _textSessionPool.Session.Run(inputs);
        var output = results.First().AsTensor<float>();

        // Extract the EOS token embedding (last non-padding token position)
        float[] embedding;
        if (output.Dimensions.Length == 3)
        {
            // Shape [1, 77, hidden_dim] — find EOS position
            int eosPos = FindEosPosition(attentionMask);
            int hiddenDim = (int)output.Dimensions[2];
            embedding = new float[hiddenDim];
            for (int i = 0; i < hiddenDim; i++)
            {
                embedding[i] = output[0, eosPos, i];
            }
        }
        else
        {
            // Shape [1, hidden_dim] — already pooled
            int hiddenDim = (int)output.Dimensions[^1];
            embedding = new float[hiddenDim];
            for (int i = 0; i < hiddenDim; i++)
            {
                embedding[i] = output[0, i];
            }
        }

        // L2-normalize the text embedding
        float norm = TensorPrimitives.Norm(embedding);
        if (norm > 0)
        {
            TensorPrimitives.Divide(embedding, norm, embedding);
        }

        return embedding;
    }

    /// <summary>
    /// Find the position of the last non-padding token (EOS token).
    /// </summary>
    private static int FindEosPosition(int[] attentionMask)
    {
        int lastNonZero = 0;
        for (int i = 0; i < attentionMask.Length; i++)
        {
            if (attentionMask[i] != 0)
            {
                lastNonZero = i;
            }
        }
        return lastNonZero;
    }

    /// <summary>
    /// Classify a single image using zero-shot CLIP classification.
    /// </summary>
    public (string Label, float Probability)[] Classify(MLImage image)
    {
        // Stage 1: Preprocess
        var tensor = _preprocessor.Preprocess(image);

        // Stage 2: Score (vision encoder)
        var output = _visionScorer.Score(tensor);

        // Stage 3: Post-process (extract embedding + cosine similarity with text embeddings)
        return PostProcess(output);
    }

    /// <summary>
    /// Classifies a batch of images using zero-shot classification with pre-encoded labels.
    /// Uses true tensor batching if the model supports dynamic batch, otherwise loops.
    /// </summary>
    public (string Label, float Probability)[][] ClassifyBatch(IReadOnlyList<MLImage> images)
    {
        if (images == null || images.Count == 0)
            return Array.Empty<(string, float)[]>();

        if (_visionScorer.IsBatchDynamic)
        {
            return ClassifyBatchDynamic(images);
        }
        else
        {
            var results = new (string Label, float Probability)[images.Count][];
            for (int i = 0; i < images.Count; i++)
                results[i] = Classify(images[i]);
            return results;
        }
    }

    private (string Label, float Probability)[][] ClassifyBatchDynamic(IReadOnlyList<MLImage> images)
    {
        int n = images.Count;

        // Stage 1: Batch preprocess
        var batchTensor = _preprocessor.PreprocessBatch(images);

        // Stage 2: Batch score (vision encoder)
        var (output, _) = _visionScorer.ScoreBatch(batchTensor, n);

        int hiddenDim = (int)_visionScorer.Metadata.OutputShapes[0][^1];
        int outputPerImage = output.Length / n;
        var batchResults = new (string Label, float Probability)[n][];

        for (int i = 0; i < n; i++)
        {
            // Stage 3: Post-process each image's output
            var imageOutput = output.AsSpan(i * outputPerImage, outputPerImage).ToArray();
            batchResults[i] = PostProcess(imageOutput);
        }

        return batchResults;
    }

    private (string Label, float Probability)[] PostProcess(float[] visionOutput)
    {
        // Extract image embedding (first hiddenDim elements = CLS token or pooled output)
        int hiddenDim = (int)_visionScorer.Metadata.OutputShapes[0][^1];
        float[] imageEmbedding = visionOutput[..hiddenDim];

        // L2-normalize image embedding
        float norm = TensorPrimitives.Norm(imageEmbedding);
        if (norm > 0)
        {
            TensorPrimitives.Divide(imageEmbedding, norm, imageEmbedding);
        }

        // Compute scores via cosine similarity + softmax
        float[] probabilities = ClipScorePostProcessor.ComputeScores(imageEmbedding, _textEmbeddings);

        // Build results
        var predictions = new (string Label, float Probability)[_options.CandidateLabels.Length];
        for (int i = 0; i < _options.CandidateLabels.Length; i++)
        {
            predictions[i] = (_options.CandidateLabels[i], probabilities[i]);
        }

        // Sort by probability descending
        Array.Sort(predictions, (a, b) => b.Probability.CompareTo(a.Probability));

        // Apply TopK if specified
        if (_options.TopK.HasValue && _options.TopK.Value < predictions.Length)
        {
            predictions = predictions[.._options.TopK.Value];
        }

        return predictions;
    }

    internal OnnxZeroShotImageClassificationOptions Options => _options;
    internal ImagePreprocessingTransformer Preprocessor => _preprocessor;
    internal OnnxImageScoringTransformer VisionScorer => _visionScorer;

    public IDataView Transform(IDataView input)
    {
        return new ZeroShotDataView(input, this);
    }

    public DataViewSchema GetOutputSchema(DataViewSchema inputSchema)
    {
        var builder = new DataViewSchema.Builder();
        for (int i = 0; i < inputSchema.Count; i++)
            builder.AddColumn(inputSchema[i].Name, inputSchema[i].Type, inputSchema[i].Annotations);
        builder.AddColumn(_options.PredictedLabelColumnName, TextDataViewType.Instance);
        builder.AddColumn(_options.ProbabilityColumnName, new VectorDataViewType(NumberDataViewType.Single));
        return builder.ToSchema();
    }

    public IRowToRowMapper GetRowToRowMapper(DataViewSchema inputSchema)
        => throw new InvalidOperationException("This transformer does not support row-to-row mapping.");

    void ICanSaveModel.Save(ModelSaveContext ctx)
        => throw new NotSupportedException("Use transformer-specific save/load instead of mlContext.Model.Save().");

    public void Dispose()
    {
        _visionScorer?.Dispose();
        _textSessionPool?.Dispose();
    }
}
