using Microsoft.ML;
using Microsoft.ML.Data;
using MLNet.Image.Core;
using MLNet.ImageInference.Onnx;
using MLNet.ImageInference.Onnx.Classification;

// =====================================================
// Image Classification Sample using ViT (Vision Transformer)
// =====================================================
//
// Before running:
//   1. Download a ViT ONNX model from HuggingFace:
//      pip install optimum[onnxruntime]
//      optimum-cli export onnx --model google/vit-base-patch16-224 models/vit/
//   2. Place the model at: models/vit/model.onnx
//   3. Place a test image at: test-image.jpg
//
// The pipeline: Image → HuggingFace Preprocess → ONNX Score → Softmax → Labels

Console.WriteLine("=== ML.NET Image Classification with ONNX ===");
Console.WriteLine();

var modelPath = args.Length > 0 ? args[0] : "models/vit/model.onnx";
var imagePath = args.Length > 1 ? args[1] : "test-image.jpg";

if (!File.Exists(modelPath))
{
    Console.WriteLine($"Model not found at: {modelPath}");
    Console.WriteLine();
    Console.WriteLine("Download a ViT ONNX model:");
    Console.WriteLine("  pip install optimum[onnxruntime]");
    Console.WriteLine("  optimum-cli export onnx --model google/vit-base-patch16-224 models/vit/");
    return;
}

if (!File.Exists(imagePath))
{
    Console.WriteLine($"Image not found at: {imagePath}");
    Console.WriteLine("Place a test image (JPEG/PNG) at the path above.");
    return;
}

// --- Style 1: Convenience API (single call) ---
Console.WriteLine("--- Style 1: Direct Classification API ---");
Console.WriteLine();

var options = new OnnxImageClassificationOptions
{
    ModelPath = modelPath,
    PreprocessorConfig = PreprocessorConfig.ImageNet,
    TopK = 5
};

var estimator = new OnnxImageClassificationEstimator(options);
using var transformer = estimator.Fit(null!);

// Load and classify an image
using var image = MLImage.CreateFromFile(imagePath);
var predictions = transformer.Classify(image);

Console.WriteLine($"Image: {imagePath}");
Console.WriteLine($"Top {predictions.Length} predictions:");
foreach (var (label, probability) in predictions)
{
    Console.WriteLine($"  {label}: {probability:P2}");
}

Console.WriteLine();

// --- Style 2: ML.NET IDataView Pipeline (composable) ---
Console.WriteLine("--- Style 2: ML.NET IDataView Pipeline ---");
Console.WriteLine();

var mlContext = new MLContext();

// Load images into an IDataView
using var pipelineImage = MLImage.CreateFromFile(imagePath);
var sourceDataView = new ImageDataView([pipelineImage]);

// Build pipeline using MLContext extension method
var pipeline = mlContext.Transforms.OnnxImageClassification(new OnnxImageClassificationOptions
{
    ModelPath = modelPath,
    PreprocessorConfig = PreprocessorConfig.ImageNet,
    TopK = 5
});

// Fit and transform
using var model = pipeline.Fit(sourceDataView);
var transformed = model.Transform(sourceDataView);

// Read results from the cursor
var labelCol = transformed.Schema["PredictedLabel"];
var probCol = transformed.Schema["Probability"];

using var cursor = transformed.GetRowCursor(transformed.Schema);
var labelGetter = cursor.GetGetter<ReadOnlyMemory<char>>(labelCol);
var probGetter = cursor.GetGetter<VBuffer<float>>(probCol);

while (cursor.MoveNext())
{
    ReadOnlyMemory<char> label = default;
    VBuffer<float> probs = default;
    labelGetter(ref label);
    probGetter(ref probs);

    Console.WriteLine($"  PredictedLabel: {label}");
    var topProbs = probs.GetValues().ToArray().Take(5).Select(v => v.ToString("P2"));
    Console.WriteLine($"  Top probabilities: [{string.Join(", ", topProbs)}]");
}

Console.WriteLine();
Console.WriteLine("Done!");

/// <summary>
/// Minimal IDataView that serves MLImage rows through the "Image" column.
/// </summary>
sealed class ImageDataView : IDataView
{
    private readonly MLImage[] _images;
    private readonly DataViewSchema _schema;

    public ImageDataView(MLImage[] images)
    {
        _images = images;
        var builder = new DataViewSchema.Builder();
        builder.AddColumn("Image", NumberDataViewType.Single);
        _schema = builder.ToSchema();
    }

    public DataViewSchema Schema => _schema;
    public bool CanShuffle => false;
    public long? GetRowCount() => _images.Length;

    public DataViewRowCursor GetRowCursor(IEnumerable<DataViewSchema.Column> columnsNeeded, Random? rand = null)
        => new ImageCursor(this);

    public DataViewRowCursor[] GetRowCursorSet(IEnumerable<DataViewSchema.Column> columnsNeeded, int n, Random? rand = null)
        => [GetRowCursor(columnsNeeded, rand)];

    private sealed class ImageCursor(ImageDataView parent) : DataViewRowCursor
    {
        private long _position = -1;

        public override DataViewSchema Schema => parent._schema;
        public override long Position => _position;
        public override long Batch => 0;
        public override bool IsColumnActive(DataViewSchema.Column column) => true;
        public override bool MoveNext() => ++_position < parent._images.Length;
        public override ValueGetter<DataViewRowId> GetIdGetter() =>
            (ref DataViewRowId id) => id = new DataViewRowId((ulong)_position, 0);

        public override ValueGetter<TValue> GetGetter<TValue>(DataViewSchema.Column column)
        {
            ValueGetter<MLImage> getter = (ref MLImage value) => value = parent._images[_position];
            return (ValueGetter<TValue>)(Delegate)getter;
        }
    }
}
