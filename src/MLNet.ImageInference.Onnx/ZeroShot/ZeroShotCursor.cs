using Microsoft.ML;
using Microsoft.ML.Data;
using MLNet.ImageInference.Onnx.Shared;

namespace MLNet.ImageInference.Onnx.ZeroShot;

/// <summary>
/// Cursor that lazily performs zero-shot image classification inference per row,
/// reading MLImage from the source and producing PredictedLabel + Probability columns.
/// </summary>
internal sealed class ZeroShotCursor : OnnxImageCursorBase<(string Label, float[] Probabilities)>
{
    private readonly ZeroShotDataView _parent;
    private readonly OnnxZeroShotImageClassificationTransformer _transformer;

    private string _predictedLabel = string.Empty;
    private float[] _probabilities = [];

    public ZeroShotCursor(
        ZeroShotDataView parent,
        DataViewRowCursor sourceCursor,
        OnnxZeroShotImageClassificationTransformer transformer,
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

    public override ValueGetter<DataViewRowId> GetIdGetter()
        => SourceCursor.GetIdGetter();
}
