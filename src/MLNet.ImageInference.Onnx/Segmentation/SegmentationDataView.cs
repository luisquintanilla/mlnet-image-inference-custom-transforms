using Microsoft.ML;
using Microsoft.ML.Data;

namespace MLNet.ImageInference.Onnx.Segmentation;

/// <summary>
/// Custom IDataView that wraps a source IDataView and appends segmentation output columns
/// (SegmentationMask, SegmentationMask_Width, SegmentationMask_Height) by running ONNX inference lazily per row.
/// </summary>
internal sealed class SegmentationDataView : IDataView
{
    private readonly IDataView _source;
    private readonly OnnxImageSegmentationTransformer _transformer;
    private readonly DataViewSchema _schema;
    private readonly int _maskIndex;
    private readonly int _widthIndex;
    private readonly int _heightIndex;

    public SegmentationDataView(IDataView source, OnnxImageSegmentationTransformer transformer)
    {
        _source = source;
        _transformer = transformer;

        var builder = new DataViewSchema.Builder();
        for (int i = 0; i < source.Schema.Count; i++)
            builder.AddColumn(source.Schema[i].Name, source.Schema[i].Type, source.Schema[i].Annotations);

        _maskIndex = source.Schema.Count;
        builder.AddColumn(transformer.Options.OutputColumnName, new VectorDataViewType(NumberDataViewType.Int32));

        _widthIndex = source.Schema.Count + 1;
        builder.AddColumn(transformer.Options.OutputColumnName + "_Width", NumberDataViewType.Int32);

        _heightIndex = source.Schema.Count + 2;
        builder.AddColumn(transformer.Options.OutputColumnName + "_Height", NumberDataViewType.Int32);

        _schema = builder.ToSchema();
    }

    public DataViewSchema Schema => _schema;

    public bool CanShuffle => false;

    public long? GetRowCount() => null;

    public DataViewRowCursor GetRowCursor(IEnumerable<DataViewSchema.Column> columnsNeeded, Random? rand = null)
    {
        var sourceColumnsNeeded = columnsNeeded
            .Where(c => c.Index < _source.Schema.Count)
            .Select(c => _source.Schema[c.Index]);

        var inputCol = _source.Schema.GetColumnOrNull(_transformer.Options.InputColumnName);
        if (inputCol.HasValue)
            sourceColumnsNeeded = sourceColumnsNeeded.Append(inputCol.Value);

        var sourceCursor = _source.GetRowCursor(sourceColumnsNeeded.Distinct(), rand);
        return new SegmentationCursor(this, sourceCursor, _transformer, inputCol);
    }

    public DataViewRowCursor[] GetRowCursorSet(IEnumerable<DataViewSchema.Column> columnsNeeded, int n, Random? rand = null)
    {
        return [GetRowCursor(columnsNeeded, rand)];
    }

    internal int MaskIndex => _maskIndex;
    internal int WidthIndex => _widthIndex;
    internal int HeightIndex => _heightIndex;
}
