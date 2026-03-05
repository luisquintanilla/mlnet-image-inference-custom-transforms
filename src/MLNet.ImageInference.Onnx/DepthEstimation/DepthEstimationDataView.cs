using Microsoft.ML;
using Microsoft.ML.Data;
using MLNet.ImageInference.Onnx.Shared;

namespace MLNet.ImageInference.Onnx.DepthEstimation;

/// <summary>
/// Custom IDataView that wraps a source and appends depth map output columns
/// (DepthMap, DepthMap_Width, DepthMap_Height) by running ONNX inference lazily per row.
/// </summary>
internal sealed class DepthEstimationDataView : OnnxImageDataViewBase
{
    private readonly OnnxImageDepthEstimationTransformer _transformer;

    public DepthEstimationDataView(IDataView source, OnnxImageDepthEstimationTransformer transformer)
        : base(source, transformer.Options.InputColumnName, (builder, nextIndex) =>
        {
            builder.AddColumn(transformer.Options.OutputColumnName, new VectorDataViewType(NumberDataViewType.Single));
            builder.AddColumn(transformer.Options.OutputColumnName + "_Width", NumberDataViewType.Int32);
            builder.AddColumn(transformer.Options.OutputColumnName + "_Height", NumberDataViewType.Int32);
        })
    {
        _transformer = transformer;
        DepthIndex = source.Schema.Count;
        WidthIndex = source.Schema.Count + 1;
        HeightIndex = source.Schema.Count + 2;
    }

    protected override DataViewRowCursor CreateCursor(
        DataViewRowCursor sourceCursor,
        DataViewSchema.Column? inputCol)
    {
        return new DepthEstimationCursor(this, sourceCursor, _transformer, inputCol);
    }

    internal int DepthIndex { get; private set; }
    internal int WidthIndex { get; private set; }
    internal int HeightIndex { get; private set; }
}
