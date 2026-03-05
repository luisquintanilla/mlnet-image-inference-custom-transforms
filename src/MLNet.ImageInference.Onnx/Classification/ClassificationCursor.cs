using Microsoft.ML;
using Microsoft.ML.Data;
using MLNet.ImageInference.Onnx.Shared;

namespace MLNet.ImageInference.Onnx.Classification;

/// <summary>
/// Cursor that lazily performs image classification inference per row,
/// reading MLImage from the source and producing PredictedLabel + Probability columns.
/// </summary>
internal sealed class ClassificationCursor : OnnxImageCursorBase<(string Label, float[] Probabilities)>
{
    private readonly ClassificationDataView _parent;
    private readonly OnnxImageClassificationTransformer _transformer;

    private string _predictedLabel = string.Empty;
    private float[] _probabilities = [];

    // Passthrough caching for image column
    private List<MLImage> _batchImages = new();
    private readonly Dictionary<int, IColumnCache> _passthroughCaches = new();

    public ClassificationCursor(
        ClassificationDataView parent,
        DataViewRowCursor sourceCursor,
        OnnxImageClassificationTransformer transformer,
        DataViewSchema.Column? inputCol)
        : base(parent, sourceCursor, inputCol, transformer.Options.BatchSize)
    {
        _parent = parent;
        _transformer = transformer;
    }

    protected override bool IsOutputColumn(int columnIndex)
        => columnIndex == _parent.PredictedLabelIndex || columnIndex == _parent.ProbabilityIndex;

    protected override (string Label, float[] Probabilities) CreateEmptyResult()
        => (string.Empty, Array.Empty<float>());

    protected override void ExtractCurrentResult((string Label, float[] Probabilities) result)
    {
        _predictedLabel = result.Label;
        _probabilities = result.Probabilities;
    }

    protected override void RunBatchInference(List<MLImage> images)
    {
        var batchResults = _transformer.ClassifyBatch(images);

        for (int i = 0; i < batchResults.Length; i++)
        {
            var results = batchResults[i];
            var label = results.Length > 0 ? results[0].Label : string.Empty;
            var probs = new float[results.Length];
            for (int j = 0; j < results.Length; j++)
                probs[j] = results[j].Probability;
            BatchResults.Add((label, probs));
        }
    }

    protected override void OnBatchClear()
    {
        _batchImages.Clear();
        foreach (var cache in _passthroughCaches.Values)
            cache.Clear();
    }

    protected override void OnImageRead(MLImage image) => _batchImages.Add(image);

    protected override void OnSourceRowRead()
    {
        foreach (var cache in _passthroughCaches.Values)
            cache.ReadCurrentRow();
    }

    protected override ValueGetter<TValue> CreateOutputGetter<TValue>(DataViewSchema.Column column)
    {
        if (column.Index == _parent.PredictedLabelIndex)
        {
            ValueGetter<ReadOnlyMemory<char>> getter = (ref ReadOnlyMemory<char> value) =>
                value = _predictedLabel.AsMemory();
            return (ValueGetter<TValue>)(Delegate)getter;
        }

        ValueGetter<VBuffer<float>> probGetter = (ref VBuffer<float> value) =>
            value = new VBuffer<float>(_probabilities.Length, _probabilities);
        return (ValueGetter<TValue>)(Delegate)probGetter;
    }

    public override ValueGetter<TValue> GetGetter<TValue>(DataViewSchema.Column column)
    {
        if (IsOutputColumn(column.Index))
            return CreateOutputGetter<TValue>(column);

        // Passthrough: serve cached image for the input column
        if (column.Index < SourceCursor.Schema.Count)
        {
            // Check if this is the image input column — return cached batch image
            var inputCol = _parent.Schema.GetColumnOrNull(_transformer.Options.InputColumnName);
            if (inputCol.HasValue && column.Index == inputCol.Value.Index)
            {
                ValueGetter<MLImage> imgGetter = (ref MLImage value) => value = _batchImages[BatchIndex];
                return (ValueGetter<TValue>)(Delegate)imgGetter;
            }

            // Lazy-register passthrough cache for other source columns
            if (!_passthroughCaches.TryGetValue(column.Index, out _))
            {
                var sourceGetter = SourceCursor.GetGetter<TValue>(SourceCursor.Schema[column.Index]);
                _passthroughCaches[column.Index] = new PassthroughCache<TValue>(sourceGetter);
            }

            var cache = (PassthroughCache<TValue>)_passthroughCaches[column.Index];
            return (ref TValue value) =>
            {
                if (BatchIndex >= 0 && BatchIndex < cache.Values.Count)
                    value = cache.Values[BatchIndex];
            };
        }

        throw new ArgumentOutOfRangeException(nameof(column), $"Column index {column.Index} is out of range.");
    }

    /// <summary>Type-erased interface for passthrough column value caching.</summary>
    private interface IColumnCache
    {
        void ReadCurrentRow();
        void Clear();
    }

    /// <summary>Typed cache that reads and stores values from a source cursor getter.</summary>
    private sealed class PassthroughCache<T> : IColumnCache
    {
        private readonly ValueGetter<T> _sourceGetter;
        public readonly List<T> Values = new();

        public PassthroughCache(ValueGetter<T> sourceGetter)
        {
            _sourceGetter = sourceGetter;
        }

        public void ReadCurrentRow()
        {
            T val = default!;
            _sourceGetter(ref val);
            Values.Add(val);
        }

        public void Clear() => Values.Clear();
    }
}
