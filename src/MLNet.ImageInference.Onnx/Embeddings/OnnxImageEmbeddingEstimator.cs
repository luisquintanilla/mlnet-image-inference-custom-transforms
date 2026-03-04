using Microsoft.ML;
using Microsoft.ML.Data;
using MLNet.ImageInference.Onnx.Shared;

namespace MLNet.ImageInference.Onnx.Embeddings;

/// <summary>
/// Facade estimator that chains: image preprocessing → ONNX scoring → pooling → L2 normalize.
/// </summary>
public sealed class OnnxImageEmbeddingEstimator : IEstimator<OnnxImageEmbeddingTransformer>
{
    private readonly OnnxImageEmbeddingOptions _options;

    public OnnxImageEmbeddingEstimator(OnnxImageEmbeddingOptions options)
    {
        ArgumentNullException.ThrowIfNull(options);
        _options = options;
    }

    public OnnxImageEmbeddingTransformer Fit(IDataView input)
    {
        return new OnnxImageEmbeddingTransformer(_options);
    }

    public SchemaShape GetOutputSchema(SchemaShape inputSchema)
    {
        var columns = inputSchema.ToDictionary(c => c.Name);

        columns[_options.EmbeddingColumnName] = SchemaShapeHelper.CreateColumn(
            _options.EmbeddingColumnName,
            SchemaShape.Column.VectorKind.Vector,
            NumberDataViewType.Single,
            isKey: false);

        return new SchemaShape(columns.Values);
    }
}
