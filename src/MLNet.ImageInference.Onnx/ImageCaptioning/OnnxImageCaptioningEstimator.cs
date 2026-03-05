using Microsoft.ML;
using Microsoft.ML.Data;
using MLNet.ImageInference.Onnx.Shared;

namespace MLNet.ImageInference.Onnx.ImageCaptioning;

/// <summary>
/// Facade estimator for image captioning: image preprocessing → vision encoder → text decoder → caption.
/// </summary>
public sealed class OnnxImageCaptioningEstimator
    : OnnxImageEstimatorBase<OnnxImageCaptioningTransformer, OnnxImageCaptioningOptions>
{
    public OnnxImageCaptioningEstimator(OnnxImageCaptioningOptions options)
        : base(options) { }

    protected override OnnxImageCaptioningTransformer CreateTransformer()
    {
        var preprocessOptions = new ImagePreprocessingOptions
        {
            InputColumnName = Options.InputColumnName,
            PreprocessorConfig = Options.PreprocessorConfig
        };
        var preprocessor = new ImagePreprocessingEstimator(preprocessOptions).Fit(null!);

        return new OnnxImageCaptioningTransformer(Options, preprocessor);
    }

    protected override void ConfigureOutputSchema(IDictionary<string, SchemaShape.Column> columns)
    {
        columns[Options.OutputColumnName] = SchemaShapeHelper.CreateColumn(
            Options.OutputColumnName,
            SchemaShape.Column.VectorKind.Scalar,
            TextDataViewType.Instance,
            isKey: false);
    }
}
