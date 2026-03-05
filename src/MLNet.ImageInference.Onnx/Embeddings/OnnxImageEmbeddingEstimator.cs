using Microsoft.ML;
using Microsoft.ML.Data;
using MLNet.ImageInference.Onnx.Shared;

namespace MLNet.ImageInference.Onnx.Embeddings;

/// <summary>
/// Facade estimator that chains: image preprocessing → ONNX scoring → pooling → L2 normalize.
/// </summary>
public sealed class OnnxImageEmbeddingEstimator
    : OnnxImageEstimatorBase<OnnxImageEmbeddingTransformer, OnnxImageEmbeddingOptions>
{
    public OnnxImageEmbeddingEstimator(OnnxImageEmbeddingOptions options)
        : base(options) { }

    protected override OnnxImageEmbeddingTransformer CreateTransformer()
        => new(Options);

    protected override void ConfigureOutputSchema(IDictionary<string, SchemaShape.Column> columns)
    {
        columns[Options.EmbeddingColumnName] = SchemaShapeHelper.CreateColumn(
            Options.EmbeddingColumnName,
            SchemaShape.Column.VectorKind.Vector,
            NumberDataViewType.Single,
            isKey: false);
    }
}
