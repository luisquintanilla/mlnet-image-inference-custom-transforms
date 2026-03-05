using Microsoft.ML;
using Microsoft.ML.Data;
using MLNet.ImageInference.Onnx.Shared;

namespace MLNet.ImageInference.Onnx.Detection;

/// <summary>
/// Cursor that lazily performs object detection inference per row,
/// reading MLImage from the source and producing DetectedObjects_Boxes + DetectedObjects_Count columns.
/// Boxes are flattened as [x, y, w, h, classId, score, ...] with 6 values per detection.
/// </summary>
internal sealed class DetectionCursor : OnnxImageCursorBase<(float[] Boxes, int Count)>
{
    private readonly DetectionDataView _parent;
    private readonly OnnxObjectDetectionTransformer _transformer;

    private float[] _boxes = [];
    private int _count;

    public DetectionCursor(
        DetectionDataView parent,
        DataViewRowCursor sourceCursor,
        OnnxObjectDetectionTransformer transformer,
        DataViewSchema.Column? inputCol)
        : base(parent, sourceCursor, inputCol, transformer.Options.BatchSize)
    {
        _parent = parent;
        _transformer = transformer;
    }

    protected override bool IsOutputColumn(int columnIndex)
        => columnIndex == _parent.BoxesIndex || columnIndex == _parent.CountIndex;

    protected override (float[] Boxes, int Count) CreateEmptyResult()
        => (Array.Empty<float>(), 0);

    protected override void ExtractCurrentResult((float[] Boxes, int Count) result)
    {
        _boxes = result.Boxes;
        _count = result.Count;
    }

    protected override void RunBatchInference(List<MLImage> images)
    {
        var batchResults = _transformer.DetectBatch(images);

        for (int i = 0; i < batchResults.Length; i++)
        {
            var detections = batchResults[i];
            var count = detections.Length;

            // Flatten: 6 values per detection [x, y, w, h, classId, score]
            var boxes = new float[count * 6];
            for (int j = 0; j < count; j++)
            {
                int offset = j * 6;
                boxes[offset] = detections[j].X;
                boxes[offset + 1] = detections[j].Y;
                boxes[offset + 2] = detections[j].Width;
                boxes[offset + 3] = detections[j].Height;
                boxes[offset + 4] = detections[j].ClassId;
                boxes[offset + 5] = detections[j].Score;
            }

            BatchResults.Add((boxes, count));
        }
    }

    protected override ValueGetter<TValue> CreateOutputGetter<TValue>(DataViewSchema.Column column)
    {
        if (column.Index == _parent.BoxesIndex)
        {
            ValueGetter<VBuffer<float>> getter = (ref VBuffer<float> value) =>
                value = new VBuffer<float>(_boxes.Length, _boxes);
            return (ValueGetter<TValue>)(Delegate)getter;
        }

        ValueGetter<int> countGetter = (ref int value) => value = _count;
        return (ValueGetter<TValue>)(Delegate)countGetter;
    }

    public override ValueGetter<DataViewRowId> GetIdGetter()
        => SourceCursor.GetIdGetter();
}
