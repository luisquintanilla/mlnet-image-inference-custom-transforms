using Microsoft.ML;
using Microsoft.ML.Data;
using MLNet.ImageInference.Onnx.Shared;

namespace MLNet.ImageInference.Onnx.ZeroShot;

/// <summary>
/// Custom IDataView that wraps a source IDataView and appends zero-shot classification output columns
/// (PredictedLabel, Probability) by running ONNX inference lazily per row.
/// </summary>
internal sealed class ZeroShotDataView : OnnxImageDataViewBase
{
    private readonly OnnxZeroShotImageClassificationTransformer _transformer;

    public ZeroShotDataView(IDataView source, OnnxZeroShotImageClassificationTransformer transformer)
        : base(source, transformer.Options.InputColumnName, (builder, nextIndex) =>
        {
            builder.AddColumn(transformer.Options.PredictedLabelColumnName, TextDataViewType.Instance);
            builder.AddColumn(transformer.Options.ProbabilityColumnName, new VectorDataViewType(NumberDataViewType.Single));
        })
    {
        _transformer = transformer;
        PredictedLabelIndex = source.Schema.Count;
        ProbabilityIndex = source.Schema.Count + 1;
    }

    protected override DataViewRowCursor CreateCursor(
        DataViewRowCursor sourceCursor,
        DataViewSchema.Column? inputCol)
    {
        return new ZeroShotCursor(this, sourceCursor, _transformer, inputCol);
    }

    internal int PredictedLabelIndex { get; private set; }
    internal int ProbabilityIndex { get; private set; }
}
