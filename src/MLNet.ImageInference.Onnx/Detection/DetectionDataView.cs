using Microsoft.ML;
using Microsoft.ML.Data;

namespace MLNet.ImageInference.Onnx.Detection;

/// <summary>
/// Custom IDataView that wraps a source IDataView and appends detection output columns
/// (DetectedObjects_Boxes, DetectedObjects_Count) by running ONNX inference lazily per row.
/// </summary>
internal sealed class DetectionDataView : IDataView
{
    private readonly IDataView _source;
    private readonly OnnxObjectDetectionTransformer _transformer;
    private readonly DataViewSchema _schema;
    private readonly int _boxesIndex;
    private readonly int _countIndex;

    public DetectionDataView(IDataView source, OnnxObjectDetectionTransformer transformer)
    {
        _source = source;
        _transformer = transformer;

        var builder = new DataViewSchema.Builder();
        for (int i = 0; i < source.Schema.Count; i++)
            builder.AddColumn(source.Schema[i].Name, source.Schema[i].Type, source.Schema[i].Annotations);

        _boxesIndex = source.Schema.Count;
        builder.AddColumn(transformer.Options.OutputColumnName + "_Boxes", new VectorDataViewType(NumberDataViewType.Single));

        _countIndex = source.Schema.Count + 1;
        builder.AddColumn(transformer.Options.OutputColumnName + "_Count", NumberDataViewType.Int32);

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
        return new DetectionCursor(this, sourceCursor, _transformer, inputCol);
    }

    public DataViewRowCursor[] GetRowCursorSet(IEnumerable<DataViewSchema.Column> columnsNeeded, int n, Random? rand = null)
    {
        return [GetRowCursor(columnsNeeded, rand)];
    }

    internal int BoxesIndex => _boxesIndex;
    internal int CountIndex => _countIndex;
}
