using Microsoft.ML;
using Microsoft.ML.Data;
using MLNet.ImageInference.Onnx.Shared;

namespace MLNet.ImageInference.Onnx.Segmentation;

/// <summary>
/// Facade estimator that chains: image preprocessing → ONNX scoring → argmax segmentation.
/// </summary>
public sealed class OnnxImageSegmentationEstimator
    : OnnxImageEstimatorBase<OnnxImageSegmentationTransformer, OnnxImageSegmentationOptions>
{
    public OnnxImageSegmentationEstimator(OnnxImageSegmentationOptions options)
        : base(options) { }

    protected override OnnxImageSegmentationTransformer CreateTransformer()
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
        return new OnnxImageSegmentationTransformer(Options, preprocessor, scorer);
    }

    protected override void ConfigureOutputSchema(IDictionary<string, SchemaShape.Column> columns)
    {
        columns[Options.OutputColumnName] = SchemaShapeHelper.CreateColumn(
            Options.OutputColumnName,
            SchemaShape.Column.VectorKind.Vector,
            NumberDataViewType.Int32,
            isKey: false);
    }
}
