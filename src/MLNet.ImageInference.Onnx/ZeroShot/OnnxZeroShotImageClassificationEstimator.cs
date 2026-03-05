using Microsoft.ML;
using Microsoft.ML.Data;
using MLNet.ImageInference.Onnx.Shared;

namespace MLNet.ImageInference.Onnx.ZeroShot;

/// <summary>
/// Facade estimator for zero-shot image classification using CLIP.
/// </summary>
public sealed class OnnxZeroShotImageClassificationEstimator
    : OnnxImageEstimatorBase<OnnxZeroShotImageClassificationTransformer, OnnxZeroShotImageClassificationOptions>
{
    public OnnxZeroShotImageClassificationEstimator(OnnxZeroShotImageClassificationOptions options)
        : base(options) { }

    protected override OnnxZeroShotImageClassificationTransformer CreateTransformer()
    {
        // Stage 1: Preprocessing
        var preprocessOptions = new ImagePreprocessingOptions
        {
            InputColumnName = Options.InputColumnName,
            PreprocessorConfig = Options.PreprocessorConfig
        };
        var preprocessor = new ImagePreprocessingEstimator(preprocessOptions).Fit(null!);

        // Stage 2: ONNX Scoring (vision encoder)
        var visionScorerOptions = new OnnxImageScoringOptions
        {
            ModelPath = Options.ImageModelPath,
            ImageHeight = Options.PreprocessorConfig.ImageSize.Height,
            ImageWidth = Options.PreprocessorConfig.ImageSize.Width,
            BatchSize = Options.BatchSize
        };
        var visionScorer = new OnnxImageScoringEstimator(visionScorerOptions).Fit(null!);

        // Compose vision stages (text encoding handled internally by transformer)
        return new OnnxZeroShotImageClassificationTransformer(Options, preprocessor, visionScorer);
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
