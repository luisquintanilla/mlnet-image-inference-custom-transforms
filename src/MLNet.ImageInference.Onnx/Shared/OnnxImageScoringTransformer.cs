using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace MLNet.ImageInference.Onnx.Shared;

/// <summary>
/// Options for the ONNX image scoring sub-transform.
/// </summary>
public class OnnxImageScoringOptions
{
    /// <summary>Path to the ONNX model file.</summary>
    public required string ModelPath { get; init; }

    /// <summary>Name of the input column containing the preprocessed tensor (VBuffer&lt;float&gt;).</summary>
    public string InputColumnName { get; init; } = "PreprocessedTensor";

    /// <summary>Name of the output column for raw ONNX scores.</summary>
    public string OutputColumnName { get; init; } = "RawScores";

    /// <summary>Image height (for reshaping the flat tensor to [N,3,H,W]).</summary>
    public required int ImageHeight { get; init; }

    /// <summary>Image width (for reshaping the flat tensor to [N,3,H,W]).</summary>
    public required int ImageWidth { get; init; }

    /// <summary>Maximum batch size for lookahead batching. Default: 32.</summary>
    public int BatchSize { get; set; } = 32;
}

/// <summary>
/// Sub-transform that runs ONNX inference on preprocessed tensors.
/// Task-agnostic — takes a flat tensor column and produces raw model output.
/// </summary>
public sealed class OnnxImageScoringTransformer : ITransformer, IDisposable
{
    private readonly OnnxImageScoringOptions _options;
    private readonly OnnxSessionPool _sessionPool;
    private readonly ModelMetadataDiscovery.ModelMetadata _metadata;

    public bool IsRowToRowMapper => true;

    public OnnxImageScoringTransformer(OnnxImageScoringOptions options)
    {
        _options = options;
        _sessionPool = new OnnxSessionPool(options.ModelPath);
        _metadata = ModelMetadataDiscovery.Discover(_sessionPool.Session);
    }

    internal OnnxImageScoringOptions Options => _options;
    internal ModelMetadataDiscovery.ModelMetadata Metadata => _metadata;
    internal OnnxSessionPool SessionPool => _sessionPool;

    /// <summary>Score a single preprocessed tensor. Returns raw ONNX output.</summary>
    public float[] Score(float[] preprocessedTensor)
    {
        var inputTensor = new DenseTensor<float>(preprocessedTensor,
            [1, 3, _options.ImageHeight, _options.ImageWidth]);
        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor(_metadata.InputNames[0], inputTensor)
        };

        using var results = _sessionPool.Session.Run(inputs);
        return results.First().AsEnumerable<float>().ToArray();
    }

    /// <summary>Score a single preprocessed tensor. Returns raw output with actual tensor dimensions.</summary>
    public (float[] Output, int[] Dimensions) ScoreWithDimensions(float[] preprocessedTensor)
    {
        var inputTensor = new DenseTensor<float>(preprocessedTensor,
            [1, 3, _options.ImageHeight, _options.ImageWidth]);
        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor(_metadata.InputNames[0], inputTensor)
        };

        using var results = _sessionPool.Session.Run(inputs);
        var resultTensor = results.First().AsTensor<float>();
        return (resultTensor.ToArray(), resultTensor.Dimensions.ToArray());
    }

    /// <summary>
    /// Score a batch of preprocessed tensors. Uses true tensor batching for dynamic-batch models.
    /// Returns (rawOutput, numImages) — caller slices per-image output from the flat array.
    /// </summary>
    public (float[] Output, int N) ScoreBatch(float[] batchTensor, int numImages)
    {
        var inputTensor = new DenseTensor<float>(batchTensor,
            [numImages, 3, _options.ImageHeight, _options.ImageWidth]);
        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor(_metadata.InputNames[0], inputTensor)
        };

        using var results = _sessionPool.Session.Run(inputs);
        return (results.First().AsEnumerable<float>().ToArray(), numImages);
    }

    /// <summary>Score a batch with actual output tensor dimensions. Returns (rawOutput, numImages, dimensions).</summary>
    public (float[] Output, int N, int[] Dimensions) ScoreBatchWithDimensions(float[] batchTensor, int numImages)
    {
        var inputTensor = new DenseTensor<float>(batchTensor,
            [numImages, 3, _options.ImageHeight, _options.ImageWidth]);
        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor(_metadata.InputNames[0], inputTensor)
        };

        using var results = _sessionPool.Session.Run(inputs);
        var resultTensor = results.First().AsTensor<float>();
        return (resultTensor.ToArray(), numImages, resultTensor.Dimensions.ToArray());
    }

    /// <summary>Whether the model supports dynamic batch (dim[0] == -1).</summary>
    public bool IsBatchDynamic => _metadata.IsBatchDynamic;

    public IDataView Transform(IDataView input)
        => new ScoringDataView(input, this);

    public DataViewSchema GetOutputSchema(DataViewSchema inputSchema)
    {
        var builder = new DataViewSchema.Builder();
        for (int i = 0; i < inputSchema.Count; i++)
            builder.AddColumn(inputSchema[i].Name, inputSchema[i].Type, inputSchema[i].Annotations);
        builder.AddColumn(_options.OutputColumnName, new VectorDataViewType(NumberDataViewType.Single));
        return builder.ToSchema();
    }

    public IRowToRowMapper GetRowToRowMapper(DataViewSchema inputSchema)
        => throw new InvalidOperationException("Use Transform() to get an IDataView.");

    void ICanSaveModel.Save(ModelSaveContext ctx)
        => throw new NotSupportedException();

    public void Dispose() => _sessionPool?.Dispose();
}

/// <summary>
/// Estimator for ONNX image scoring.
/// </summary>
public sealed class OnnxImageScoringEstimator : IEstimator<OnnxImageScoringTransformer>
{
    private readonly OnnxImageScoringOptions _options;

    public OnnxImageScoringEstimator(OnnxImageScoringOptions options)
    {
        ArgumentNullException.ThrowIfNull(options);
        _options = options;
    }

    public OnnxImageScoringTransformer Fit(IDataView input) => new(_options);

    public SchemaShape GetOutputSchema(SchemaShape inputSchema)
    {
        var columns = inputSchema.ToDictionary(c => c.Name);
        columns[_options.OutputColumnName] = SchemaShapeHelper.CreateColumn(
            _options.OutputColumnName,
            SchemaShape.Column.VectorKind.Vector,
            NumberDataViewType.Single,
            isKey: false);
        return new SchemaShape(columns.Values);
    }
}

/// <summary>
/// IDataView wrapper that adds a raw ONNX scores column.
/// </summary>
internal sealed class ScoringDataView : IDataView
{
    private readonly IDataView _source;
    private readonly OnnxImageScoringTransformer _transformer;
    private readonly DataViewSchema _schema;

    public ScoringDataView(IDataView source, OnnxImageScoringTransformer transformer)
    {
        _source = source;
        _transformer = transformer;

        var builder = new DataViewSchema.Builder();
        for (int i = 0; i < source.Schema.Count; i++)
            builder.AddColumn(source.Schema[i].Name, source.Schema[i].Type, source.Schema[i].Annotations);
        builder.AddColumn(transformer.Options.OutputColumnName, new VectorDataViewType(NumberDataViewType.Single));
        _schema = builder.ToSchema();
    }

    public DataViewSchema Schema => _schema;
    public bool CanShuffle => false;
    public long? GetRowCount() => null;

    internal int ScoresIndex => _source.Schema.Count;

    public DataViewRowCursor GetRowCursor(IEnumerable<DataViewSchema.Column> columnsNeeded, Random? rand = null)
    {
        var sourceColumnsNeeded = columnsNeeded
            .Where(c => c.Index < _source.Schema.Count)
            .Select(c => _source.Schema[c.Index]);

        var inputCol = _source.Schema.GetColumnOrNull(_transformer.Options.InputColumnName);
        if (inputCol.HasValue)
            sourceColumnsNeeded = sourceColumnsNeeded.Append(inputCol.Value);

        var sourceCursor = _source.GetRowCursor(sourceColumnsNeeded.Distinct(), rand);
        return new ScoringCursor(this, sourceCursor, _transformer, inputCol);
    }

    public DataViewRowCursor[] GetRowCursorSet(IEnumerable<DataViewSchema.Column> columnsNeeded, int n, Random? rand = null)
        => [GetRowCursor(columnsNeeded, rand)];
}

/// <summary>
/// Cursor with lookahead batching that reads preprocessed tensors and produces raw ONNX scores.
/// </summary>
internal sealed class ScoringCursor : DataViewRowCursor
{
    private readonly ScoringDataView _parent;
    private readonly DataViewRowCursor _sourceCursor;
    private readonly OnnxImageScoringTransformer _transformer;
    private readonly DataViewSchema.Column? _inputCol;
    private readonly int _batchSize;

    private List<float[]> _batchResults = new();
    private int _batchIndex = -1;
    private bool _inputExhausted;
    private long _position = -1;
    private bool _disposed;
    private float[] _currentScores = [];

    public ScoringCursor(
        ScoringDataView parent,
        DataViewRowCursor sourceCursor,
        OnnxImageScoringTransformer transformer,
        DataViewSchema.Column? inputCol)
    {
        _parent = parent;
        _sourceCursor = sourceCursor;
        _transformer = transformer;
        _inputCol = inputCol;
        _batchSize = transformer.Options.BatchSize;
    }

    public override DataViewSchema Schema => _parent.Schema;
    public override long Position => _position;
    public override long Batch => 0;

    public override bool IsColumnActive(DataViewSchema.Column column)
    {
        if (column.Index == _parent.ScoresIndex)
            return true;
        if (column.Index < _sourceCursor.Schema.Count)
            return _sourceCursor.IsColumnActive(_sourceCursor.Schema[column.Index]);
        return false;
    }

    public override bool MoveNext()
    {
        _batchIndex++;
        if (_batchResults.Count == 0 || _batchIndex >= _batchResults.Count)
        {
            if (_inputExhausted) return false;
            if (!FillNextBatch()) return false;
        }
        _currentScores = _batchResults[_batchIndex];
        _position++;
        return true;
    }

    private bool FillNextBatch()
    {
        _batchResults.Clear();
        _batchIndex = 0;

        var tensors = new List<float[]>();
        int tensorSize = 3 * _transformer.Options.ImageHeight * _transformer.Options.ImageWidth;

        for (int i = 0; i < _batchSize; i++)
        {
            if (!_sourceCursor.MoveNext())
            {
                _inputExhausted = true;
                break;
            }

            if (_inputCol.HasValue)
            {
                VBuffer<float> buffer = default;
                var getter = _sourceCursor.GetGetter<VBuffer<float>>(_inputCol.Value);
                getter(ref buffer);
                tensors.Add(buffer.DenseValues().ToArray());
            }
            else
            {
                _batchResults.Add(Array.Empty<float>());
            }
        }

        if (tensors.Count > 0)
        {
            if (_transformer.IsBatchDynamic && tensors.Count > 1)
            {
                // True batch inference
                var batchTensor = new float[tensors.Count * tensorSize];
                for (int i = 0; i < tensors.Count; i++)
                    Array.Copy(tensors[i], 0, batchTensor, i * tensorSize, tensorSize);

                var (output, n) = _transformer.ScoreBatch(batchTensor, tensors.Count);
                int outputPerImage = output.Length / n;

                for (int i = 0; i < n; i++)
                {
                    var imageOutput = new float[outputPerImage];
                    Array.Copy(output, i * outputPerImage, imageOutput, 0, outputPerImage);
                    _batchResults.Add(imageOutput);
                }
            }
            else
            {
                // Per-image scoring
                foreach (var tensor in tensors)
                    _batchResults.Add(_transformer.Score(tensor));
            }
        }

        return _batchResults.Count > 0;
    }

    public override ValueGetter<TValue> GetGetter<TValue>(DataViewSchema.Column column)
    {
        if (column.Index == _parent.ScoresIndex)
        {
            ValueGetter<VBuffer<float>> getter = (ref VBuffer<float> value) =>
                value = new VBuffer<float>(_currentScores.Length, _currentScores);
            return (ValueGetter<TValue>)(Delegate)getter;
        }

        if (column.Index < _sourceCursor.Schema.Count)
            return _sourceCursor.GetGetter<TValue>(_sourceCursor.Schema[column.Index]);

        throw new ArgumentOutOfRangeException(nameof(column));
    }

    public override ValueGetter<DataViewRowId> GetIdGetter()
        => (ref DataViewRowId id) => id = new DataViewRowId((ulong)_position, 0);

    protected override void Dispose(bool disposing)
    {
        if (!_disposed)
        {
            if (disposing) _sourceCursor.Dispose();
            _disposed = true;
        }
        base.Dispose(disposing);
    }
}
