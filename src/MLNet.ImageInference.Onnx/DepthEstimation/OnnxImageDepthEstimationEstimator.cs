using Microsoft.ML;
using Microsoft.ML.Data;
using MLNet.ImageInference.Onnx.Shared;

namespace MLNet.ImageInference.Onnx.DepthEstimation;

/// <summary>
/// Facade estimator that chains: image preprocessing → ONNX scoring → depth normalization.
/// </summary>
public sealed class OnnxImageDepthEstimationEstimator
    : OnnxImageEstimatorBase<OnnxImageDepthEstimationTransformer, OnnxImageDepthEstimationOptions>
{
    public OnnxImageDepthEstimationEstimator(OnnxImageDepthEstimationOptions options)
        : base(options) { }

    protected override OnnxImageDepthEstimationTransformer CreateTransformer()
    {
        var preprocessOptions = new ImagePreprocessingOptions
        {
            InputColumnName = Options.InputColumnName,
            PreprocessorConfig = Options.PreprocessorConfig
        };
        var preprocessor = new ImagePreprocessingEstimator(preprocessOptions).Fit(null!);

        var scorerOptions = new OnnxImageScoringOptions
        {
            ModelPath = Options.ModelPath,
            ImageHeight = Options.PreprocessorConfig.ImageSize.Height,
            ImageWidth = Options.PreprocessorConfig.ImageSize.Width,
            BatchSize = Options.BatchSize
        };
        var scorer = new OnnxImageScoringEstimator(scorerOptions).Fit(null!);

        return new OnnxImageDepthEstimationTransformer(Options, preprocessor, scorer);
    }

    protected override void ConfigureOutputSchema(IDictionary<string, SchemaShape.Column> columns)
    {
        columns[Options.OutputColumnName] = SchemaShapeHelper.CreateColumn(
            Options.OutputColumnName,
            SchemaShape.Column.VectorKind.Vector,
            NumberDataViewType.Single,
            isKey: false);
    }
}
