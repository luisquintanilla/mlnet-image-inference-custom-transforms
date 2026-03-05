using Microsoft.ML;
using Microsoft.ML.Data;
using MLNet.Image.Core;

namespace MLNet.ImageInference.Onnx.Shared;

/// <summary>
/// Options for the image preprocessing sub-transform.
/// </summary>
public class ImagePreprocessingOptions
{
    /// <summary>Name of the input column containing MLImage values.</summary>
    public string InputColumnName { get; init; } = "Image";

    /// <summary>Name of the output column for the preprocessed tensor.</summary>
    public string OutputColumnName { get; init; } = "PreprocessedTensor";

    /// <summary>Preprocessing configuration (mean, std, image size, etc.).</summary>
    public required PreprocessorConfig PreprocessorConfig { get; init; }
}

/// <summary>
/// Sub-transform that converts MLImage → preprocessed float tensor [3,H,W].
/// Reusable across all image inference tasks.
/// </summary>
public sealed class ImagePreprocessingTransformer : ITransformer
{
    private readonly ImagePreprocessingOptions _options;

    public bool IsRowToRowMapper => true;

    public ImagePreprocessingTransformer(ImagePreprocessingOptions options)
    {
        _options = options;
    }

    internal ImagePreprocessingOptions Options => _options;

    /// <summary>Preprocess a single image to a flat float tensor [3*H*W].</summary>
    public float[] Preprocess(MLImage image)
        => HuggingFaceImagePreprocessor.Preprocess(image, _options.PreprocessorConfig);

    /// <summary>Preprocess a batch of images to a flat float tensor [N*3*H*W].</summary>
    public float[] PreprocessBatch(IReadOnlyList<MLImage> images)
        => HuggingFaceImagePreprocessor.PreprocessBatch(images, _options.PreprocessorConfig);

    public IDataView Transform(IDataView input)
        => new PreprocessingDataView(input, this);

    public DataViewSchema GetOutputSchema(DataViewSchema inputSchema)
    {
        var builder = new DataViewSchema.Builder();
        for (int i = 0; i < inputSchema.Count; i++)
            builder.AddColumn(inputSchema[i].Name, inputSchema[i].Type, inputSchema[i].Annotations);

        int tensorSize = 3 * _options.PreprocessorConfig.ImageSize.Height * _options.PreprocessorConfig.ImageSize.Width;
        builder.AddColumn(_options.OutputColumnName, new VectorDataViewType(NumberDataViewType.Single, tensorSize));
        return builder.ToSchema();
    }

    public IRowToRowMapper GetRowToRowMapper(DataViewSchema inputSchema)
        => throw new InvalidOperationException("Use Transform() to get an IDataView.");

    void ICanSaveModel.Save(ModelSaveContext ctx)
        => throw new NotSupportedException();
}

/// <summary>
/// Estimator for image preprocessing.
/// </summary>
public sealed class ImagePreprocessingEstimator : IEstimator<ImagePreprocessingTransformer>
{
    private readonly ImagePreprocessingOptions _options;

    public ImagePreprocessingEstimator(ImagePreprocessingOptions options)
    {
        ArgumentNullException.ThrowIfNull(options);
        _options = options;
    }

    public ImagePreprocessingTransformer Fit(IDataView input) => new(_options);

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
/// IDataView wrapper that adds a preprocessed tensor column.
/// </summary>
internal sealed class PreprocessingDataView : OnnxImageDataViewBase
{
    private readonly ImagePreprocessingTransformer _transformer;

    public PreprocessingDataView(IDataView source, ImagePreprocessingTransformer transformer)
        : base(source, transformer.Options.InputColumnName, (builder, nextIndex) =>
        {
            int tensorSize = 3 * transformer.Options.PreprocessorConfig.ImageSize.Height
                               * transformer.Options.PreprocessorConfig.ImageSize.Width;
            builder.AddColumn(transformer.Options.OutputColumnName,
                new VectorDataViewType(NumberDataViewType.Single, tensorSize));
        })
    {
        _transformer = transformer;
        TensorIndex = source.Schema.Count;
    }

    internal int TensorIndex { get; }

    protected override DataViewRowCursor CreateCursor(
        DataViewRowCursor sourceCursor, DataViewSchema.Column? inputCol)
        => new PreprocessingCursor(this, sourceCursor, _transformer, inputCol);
}

/// <summary>
/// Cursor that reads MLImage and produces preprocessed tensor per row.
/// </summary>
internal sealed class PreprocessingCursor : OnnxImageCursorBase<float[]>
{
    private readonly PreprocessingDataView _parent;
    private readonly ImagePreprocessingTransformer _transformer;
    private float[] _currentTensor = [];

    public PreprocessingCursor(
        PreprocessingDataView parent,
        DataViewRowCursor sourceCursor,
        ImagePreprocessingTransformer transformer,
        DataViewSchema.Column? inputCol)
        : base(parent, sourceCursor, inputCol, 1) // Row-by-row (scoring stage handles batching)
    {
        _parent = parent;
        _transformer = transformer;
    }

    protected override bool IsOutputColumn(int columnIndex)
        => columnIndex == _parent.TensorIndex;

    protected override float[] CreateEmptyResult()
    {
        int size = 3 * _transformer.Options.PreprocessorConfig.ImageSize.Height
                     * _transformer.Options.PreprocessorConfig.ImageSize.Width;
        return new float[size];
    }

    protected override void ExtractCurrentResult(float[] result) => _currentTensor = result;

    protected override void RunBatchInference(List<MLImage> images)
    {
        foreach (var image in images)
            BatchResults.Add(_transformer.Preprocess(image));
    }

    protected override ValueGetter<TValue> CreateOutputGetter<TValue>(DataViewSchema.Column column)
    {
        ValueGetter<VBuffer<float>> getter = (ref VBuffer<float> value) =>
            value = new VBuffer<float>(_currentTensor.Length, _currentTensor);
        return (ValueGetter<TValue>)(Delegate)getter;
    }
}
