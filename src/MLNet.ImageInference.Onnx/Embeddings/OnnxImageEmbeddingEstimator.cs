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
    {
        // Stage 1: Preprocessing
        var preprocessOptions = new ImagePreprocessingOptions
        {
            InputColumnName = Options.InputColumnName,
            PreprocessorConfig = Options.PreprocessorConfig
        };
        var preprocessor = new ImagePreprocessingEstimator(preprocessOptions).Fit(null!);

        // Stage 2: ONNX Scoring
        var scorerOptions = new OnnxImageScoringOptions
        {
            ModelPath = Options.ModelPath,
            ImageHeight = Options.PreprocessorConfig.ImageSize.Height,
            ImageWidth = Options.PreprocessorConfig.ImageSize.Width,
            BatchSize = Options.BatchSize
        };
        var scorer = new OnnxImageScoringEstimator(scorerOptions).Fit(null!);

        // Compose all stages
        return new OnnxImageEmbeddingTransformer(Options, preprocessor, scorer);
    }

    protected override void ConfigureOutputSchema(IDictionary<string, SchemaShape.Column> columns)
    {
        columns[Options.EmbeddingColumnName] = SchemaShapeHelper.CreateColumn(
            Options.EmbeddingColumnName,
            SchemaShape.Column.VectorKind.Vector,
            NumberDataViewType.Single,
            isKey: false);
    }
}
