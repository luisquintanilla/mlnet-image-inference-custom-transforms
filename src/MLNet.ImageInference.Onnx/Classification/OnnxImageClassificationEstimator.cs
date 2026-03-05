using Microsoft.ML;
using Microsoft.ML.Data;
using MLNet.ImageInference.Onnx.Shared;

namespace MLNet.ImageInference.Onnx.Classification;

/// <summary>
/// Facade estimator that composes: ImagePreprocessing → OnnxImageScoring → classification post-processing.
/// </summary>
public sealed class OnnxImageClassificationEstimator
    : OnnxImageEstimatorBase<OnnxImageClassificationTransformer, OnnxImageClassificationOptions>
{
    public OnnxImageClassificationEstimator(OnnxImageClassificationOptions options)
        : base(options) { }

    protected override OnnxImageClassificationTransformer CreateTransformer()
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
        return new OnnxImageClassificationTransformer(Options, preprocessor, scorer);
    }

    protected override void ConfigureOutputSchema(IDictionary<string, SchemaShape.Column> columns)
    {
        columns[Options.PredictedLabelColumnName] = SchemaShapeHelper.CreateColumn(
            Options.PredictedLabelColumnName,
            SchemaShape.Column.VectorKind.Scalar,
            TextDataViewType.Instance,
            isKey: false);

        columns[Options.ProbabilityColumnName] = SchemaShapeHelper.CreateColumn(
            Options.ProbabilityColumnName,
            SchemaShape.Column.VectorKind.Vector,
            NumberDataViewType.Single,
            isKey: false);
    }
}
