using Microsoft.ML;
using Microsoft.ML.Data;
using MLNet.ImageInference.Onnx.Shared;

namespace MLNet.ImageInference.Onnx.Embeddings;

/// <summary>
/// Cursor that lazily performs image embedding inference per row,
/// reading MLImage from the source and producing an Embedding vector column.
/// Uses lookahead batching to amortize ONNX inference overhead.
/// </summary>
internal sealed class EmbeddingCursor : OnnxImageCursorBase<float[]>
{
    private readonly EmbeddingDataView _parent;
    private readonly OnnxImageEmbeddingTransformer _transformer;

    private float[] _embedding = [];

    public EmbeddingCursor(
        EmbeddingDataView parent,
        DataViewRowCursor sourceCursor,
        OnnxImageEmbeddingTransformer transformer,
        DataViewSchema.Column? inputCol)
        : base(parent, sourceCursor, inputCol, transformer.Options.BatchSize)
    {
        _parent = parent;
        _transformer = transformer;
    }

    protected override bool IsOutputColumn(int columnIndex)
        => columnIndex == _parent.EmbeddingIndex;

    protected override float[] CreateEmptyResult()
        => new float[_transformer.EmbeddingDimension];

    protected override void ExtractCurrentResult(float[] result)
    {
        _embedding = result;
    }

    protected override void RunBatchInference(List<MLImage> images)
    {
        var embeddings = _transformer.GenerateEmbeddingBatch(images);
        BatchResults.AddRange(embeddings);
    }

    protected override ValueGetter<TValue> CreateOutputGetter<TValue>(DataViewSchema.Column column)
    {
        ValueGetter<VBuffer<float>> getter = (ref VBuffer<float> value) =>
            value = new VBuffer<float>(_embedding.Length, _embedding);
        return (ValueGetter<TValue>)(Delegate)getter;
    }
}
