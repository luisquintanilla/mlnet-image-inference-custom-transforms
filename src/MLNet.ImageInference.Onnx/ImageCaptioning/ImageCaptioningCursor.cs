using Microsoft.ML;
using Microsoft.ML.Data;
using MLNet.Image.Core;
using MLNet.ImageInference.Onnx.Shared;

namespace MLNet.ImageInference.Onnx.ImageCaptioning;

/// <summary>
/// Cursor that lazily performs image captioning inference per row,
/// reading MLImage from the source and producing a caption string column.
/// </summary>
internal sealed class ImageCaptioningCursor : OnnxImageCursorBase<string>
{
    private readonly ImageCaptioningDataView _parent;
    private readonly OnnxImageCaptioningTransformer _transformer;

    private ReadOnlyMemory<char> _caption;

    public ImageCaptioningCursor(
        ImageCaptioningDataView parent,
        DataViewRowCursor sourceCursor,
        OnnxImageCaptioningTransformer transformer,
        DataViewSchema.Column? inputCol)
        : base(parent, sourceCursor, inputCol, transformer.Options.BatchSize)
    {
        _parent = parent;
        _transformer = transformer;
    }

    protected override bool IsOutputColumn(int columnIndex)
        => columnIndex == _parent.CaptionIndex;

    protected override string CreateEmptyResult()
        => string.Empty;

    protected override void ExtractCurrentResult(string result)
    {
        _caption = result.AsMemory();
    }

    protected override void RunBatchInference(List<MLImage> images)
    {
        var batchResults = _transformer.GenerateCaptionBatch(images);
        for (int i = 0; i < batchResults.Length; i++)
            BatchResults.Add(batchResults[i]);
    }

    protected override ValueGetter<TValue> CreateOutputGetter<TValue>(DataViewSchema.Column column)
    {
        ValueGetter<ReadOnlyMemory<char>> getter = (ref ReadOnlyMemory<char> value) => value = _caption;
        return (ValueGetter<TValue>)(Delegate)getter;
    }

    public override ValueGetter<DataViewRowId> GetIdGetter()
        => SourceCursor.GetIdGetter();
}
