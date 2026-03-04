using Microsoft.ML;
using Microsoft.ML.Data;

namespace MLNet.ImageInference.Onnx.ZeroShot;

/// <summary>
/// Custom IDataView that wraps a source IDataView and appends zero-shot classification output columns
/// (PredictedLabel, Probability) by running ONNX inference lazily per row.
/// </summary>
internal sealed class ZeroShotDataView : IDataView
{
    private readonly IDataView _source;
    private readonly OnnxZeroShotImageClassificationTransformer _transformer;
    private readonly DataViewSchema _schema;
    private readonly int _predictedLabelIndex;
    private readonly int _probabilityIndex;

    public ZeroShotDataView(IDataView source, OnnxZeroShotImageClassificationTransformer transformer)
    {
        _source = source;
        _transformer = transformer;

        var builder = new DataViewSchema.Builder();
        for (int i = 0; i < source.Schema.Count; i++)
            builder.AddColumn(source.Schema[i].Name, source.Schema[i].Type, source.Schema[i].Annotations);

        _predictedLabelIndex = source.Schema.Count;
        builder.AddColumn(transformer.Options.PredictedLabelColumnName, TextDataViewType.Instance);

        _probabilityIndex = source.Schema.Count + 1;
        builder.AddColumn(transformer.Options.ProbabilityColumnName, new VectorDataViewType(NumberDataViewType.Single));

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
        return new ZeroShotCursor(this, sourceCursor, _transformer, inputCol);
    }

    public DataViewRowCursor[] GetRowCursorSet(IEnumerable<DataViewSchema.Column> columnsNeeded, int n, Random? rand = null)
    {
        return [GetRowCursor(columnsNeeded, rand)];
    }

    internal int PredictedLabelIndex => _predictedLabelIndex;
    internal int ProbabilityIndex => _probabilityIndex;
}
