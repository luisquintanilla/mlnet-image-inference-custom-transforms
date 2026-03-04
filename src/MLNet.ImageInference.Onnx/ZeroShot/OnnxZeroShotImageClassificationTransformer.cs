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
/// </summary>
public sealed class OnnxZeroShotImageClassificationTransformer : ITransformer, IDisposable
{
    private readonly OnnxZeroShotImageClassificationOptions _options;
    private readonly OnnxSessionPool _imageSessionPool;
    private readonly OnnxSessionPool _textSessionPool;
    private readonly ModelMetadataDiscovery.ModelMetadata _imageMetadata;
    private readonly float[][] _textEmbeddings;

    public bool IsRowToRowMapper => true;

    public OnnxZeroShotImageClassificationTransformer(OnnxZeroShotImageClassificationOptions options)
    {
        _options = options;

        // Create session pools for both encoders
        _imageSessionPool = new OnnxSessionPool(options.ImageModelPath);
        _textSessionPool = new OnnxSessionPool(options.TextModelPath);

        // Discover model metadata
        _imageMetadata = ModelMetadataDiscovery.Discover(_imageSessionPool.Session);
        var textMetadata = ModelMetadataDiscovery.Discover(_textSessionPool.Session);

        // Create tokenizer and pre-encode all candidate labels
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
        // Preprocess image
        var tensor = HuggingFaceImagePreprocessor.Preprocess(image, _options.PreprocessorConfig);
        int height = _options.PreprocessorConfig.ImageSize.Height;
        int width = _options.PreprocessorConfig.ImageSize.Width;

        // Create ONNX input tensor [1, 3, H, W]
        var inputTensor = new DenseTensor<float>(tensor, [1, 3, height, width]);
        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor(_imageMetadata.InputNames[0], inputTensor)
        };

        // Run vision encoder
        using var results = _imageSessionPool.Session.Run(inputs);
        var output = results.First().AsTensor<float>();

        // Extract image embedding (CLS token or pooled output)
        float[] imageEmbedding;
        if (output.Dimensions.Length == 3)
        {
            // [1, seq_len, hidden_dim] — take CLS token (first token)
            int hiddenDim = (int)output.Dimensions[2];
            imageEmbedding = new float[hiddenDim];
            for (int i = 0; i < hiddenDim; i++)
            {
                imageEmbedding[i] = output[0, 0, i];
            }
        }
        else
        {
            // [1, hidden_dim] — already pooled
            int hiddenDim = (int)output.Dimensions[^1];
            imageEmbedding = new float[hiddenDim];
            for (int i = 0; i < hiddenDim; i++)
            {
                imageEmbedding[i] = output[0, i];
            }
        }

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
        _imageSessionPool?.Dispose();
        _textSessionPool?.Dispose();
    }
}
