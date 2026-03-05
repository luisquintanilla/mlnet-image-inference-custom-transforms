using Microsoft.ML;
using Microsoft.ML.Data;

namespace MLNet.ImageInference.Onnx.Shared;

/// <summary>
/// Base class for all ONNX image inference IDataView wrappers.
/// Copies source schema columns and delegates cursor creation to derived types.
/// </summary>
internal abstract class OnnxImageDataViewBase : IDataView
{
    private readonly IDataView _source;
    private readonly DataViewSchema _schema;
    private readonly string _inputColumnName;

    /// <summary>
    /// Initialize the DataView. Derived classes must pass an action that adds output columns
    /// to the schema builder (called during construction with the next available column index).
    /// </summary>
    protected OnnxImageDataViewBase(
        IDataView source,
        string inputColumnName,
        Action<DataViewSchema.Builder, int> addOutputColumns)
    {
        _source = source;
        _inputColumnName = inputColumnName;

        var builder = new DataViewSchema.Builder();
        for (int i = 0; i < source.Schema.Count; i++)
            builder.AddColumn(source.Schema[i].Name, source.Schema[i].Type, source.Schema[i].Annotations);

        addOutputColumns(builder, source.Schema.Count);

        _schema = builder.ToSchema();
    }

    public DataViewSchema Schema => _schema;

    public bool CanShuffle => false;

    public long? GetRowCount() => null;

    /// <summary>Create the task-specific cursor.</summary>
    protected abstract DataViewRowCursor CreateCursor(
        DataViewRowCursor sourceCursor,
        DataViewSchema.Column? inputCol);

    public DataViewRowCursor GetRowCursor(IEnumerable<DataViewSchema.Column> columnsNeeded, Random? rand = null)
    {
        var sourceColumnsNeeded = columnsNeeded
            .Where(c => c.Index < _source.Schema.Count)
            .Select(c => _source.Schema[c.Index]);

        var inputCol = _source.Schema.GetColumnOrNull(_inputColumnName);
        if (inputCol.HasValue)
            sourceColumnsNeeded = sourceColumnsNeeded.Append(inputCol.Value);

        var sourceCursor = _source.GetRowCursor(sourceColumnsNeeded.Distinct(), rand);
        return CreateCursor(sourceCursor, inputCol);
    }

    public DataViewRowCursor[] GetRowCursorSet(IEnumerable<DataViewSchema.Column> columnsNeeded, int n, Random? rand = null)
    {
        return [GetRowCursor(columnsNeeded, rand)];
    }

    /// <summary>Number of columns in the original source schema (before appended output columns).</summary>
    internal int SourceSchemaCount => _source.Schema.Count;
}
