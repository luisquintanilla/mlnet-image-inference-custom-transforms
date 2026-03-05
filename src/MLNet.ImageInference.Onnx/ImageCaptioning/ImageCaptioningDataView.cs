using Microsoft.ML;
using Microsoft.ML.Data;
using MLNet.ImageInference.Onnx.Shared;

namespace MLNet.ImageInference.Onnx.ImageCaptioning;

/// <summary>
/// Custom IDataView that wraps a source and appends a caption string output column
/// by running image captioning inference lazily per row.
/// </summary>
internal sealed class ImageCaptioningDataView : OnnxImageDataViewBase
{
    private readonly OnnxImageCaptioningTransformer _transformer;

    public ImageCaptioningDataView(IDataView source, OnnxImageCaptioningTransformer transformer)
        : base(source, transformer.Options.InputColumnName, (builder, nextIndex) =>
        {
            builder.AddColumn(transformer.Options.OutputColumnName, TextDataViewType.Instance);
        })
    {
        _transformer = transformer;
        CaptionIndex = source.Schema.Count;
    }

    protected override DataViewRowCursor CreateCursor(
        DataViewRowCursor sourceCursor,
        DataViewSchema.Column? inputCol)
    {
        return new ImageCaptioningCursor(this, sourceCursor, _transformer, inputCol);
    }

    internal int CaptionIndex { get; private set; }
}
