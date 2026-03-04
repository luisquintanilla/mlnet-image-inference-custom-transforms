using Microsoft.Extensions.AI;
using Microsoft.ML;
using Microsoft.ML.Data;
using MLNet.Image.Core;
using MLNet.ImageInference.Onnx;
using MLNet.ImageInference.Onnx.Embeddings;
using MLNet.ImageInference.Onnx.MEAI;

// =====================================================
// Image Embeddings Sample using CLIP / DINOv2
// =====================================================
//
// Before running:
//   1. Download a CLIP ONNX model from HuggingFace:
//      pip install optimum[onnxruntime]
//      optimum-cli export onnx --model openai/clip-vit-base-patch32 models/clip/
//   2. Place the vision model at: models/clip/model.onnx
//   3. Place test images in the current directory
//
// The pipeline: Image → HuggingFace Preprocess → ONNX Score → Pooling → L2 Normalize → float[] embedding

Console.WriteLine("=== ML.NET Image Embeddings with ONNX ===");
Console.WriteLine();

var modelPath = args.Length > 0 ? args[0] : "models/clip/model.onnx";

if (!File.Exists(modelPath))
{
    Console.WriteLine($"Model not found at: {modelPath}");
    Console.WriteLine();
    Console.WriteLine("Download a CLIP ONNX model:");
    Console.WriteLine("  pip install optimum[onnxruntime]");
    Console.WriteLine("  optimum-cli export onnx --model openai/clip-vit-base-patch32 models/clip/");
    return;
}

// --- Style 1: Direct Embedding API ---
Console.WriteLine("--- Style 1: Direct Embedding API ---");
Console.WriteLine();

var embeddingOptions = new OnnxImageEmbeddingOptions
{
    ModelPath = modelPath,
    PreprocessorConfig = PreprocessorConfig.CLIP,
    Pooling = PoolingStrategy.ClsToken,
    Normalize = true
};

using var transformer = new OnnxImageEmbeddingTransformer(embeddingOptions);
Console.WriteLine($"Embedding dimension: {transformer.EmbeddingDimension}");
Console.WriteLine();

// Find test images
var imageFiles = Directory.GetFiles(".", "*.jpg")
    .Concat(Directory.GetFiles(".", "*.png"))
    .Take(3)
    .ToArray();

if (imageFiles.Length == 0)
{
    Console.WriteLine("No test images found. Place .jpg or .png files in the current directory.");
    Console.WriteLine("Skipping embedding generation demo.");
}
else
{
    var embeddings = new List<(string File, float[] Embedding)>();

    foreach (var file in imageFiles)
    {
        using var image = MLImage.CreateFromFile(file);
        var embedding = transformer.GenerateEmbedding(image);
        embeddings.Add((Path.GetFileName(file), embedding));
        Console.WriteLine($"  {Path.GetFileName(file)}: [{string.Join(", ", embedding.Take(5).Select(v => v.ToString("F4")))}...]");
    }

    // Compute cosine similarity between all pairs
    if (embeddings.Count >= 2)
    {
        Console.WriteLine();
        Console.WriteLine("Cosine similarities:");
        for (int i = 0; i < embeddings.Count; i++)
        {
            for (int j = i + 1; j < embeddings.Count; j++)
            {
                float similarity = CosineSimilarity(embeddings[i].Embedding, embeddings[j].Embedding);
                Console.WriteLine($"  {embeddings[i].File} ↔ {embeddings[j].File}: {similarity:F4}");
            }
        }
    }
}

Console.WriteLine();

// --- Style 2: MEAI IEmbeddingGenerator<MLImage, Embedding<float>> ---
Console.WriteLine("--- Style 2: MEAI IEmbeddingGenerator ---");
Console.WriteLine();

var generator = new OnnxImageEmbeddingGenerator(modelPath);
Console.WriteLine($"Provider: {generator.Metadata.ProviderName}");
Console.WriteLine($"Model: {generator.Metadata.DefaultModelId}");
Console.WriteLine($"Dimensions: {generator.Metadata.DefaultModelDimensions}");

// The generator implements IEmbeddingGenerator<MLImage, Embedding<float>>
IEmbeddingGenerator<MLImage, Embedding<float>> embeddingGenerator = generator;

if (imageFiles.Length > 0)
{
    var images = imageFiles.Select(f => MLImage.CreateFromFile(f)).ToList();
    var meaiEmbeddings = await embeddingGenerator.GenerateAsync(images);

    Console.WriteLine($"Generated {meaiEmbeddings.Count} embeddings via MEAI interface.");

    foreach (var img in images) img.Dispose();
}

Console.WriteLine();

// --- MLImage ↔ DataContent conversion helpers ---
Console.WriteLine("--- MLImage ↔ DataContent Conversion ---");
Console.WriteLine();

if (imageFiles.Length > 0)
{
    using var sampleImage = MLImage.CreateFromFile(imageFiles[0]);
    Console.WriteLine($"Original MLImage: {sampleImage.Width}x{sampleImage.Height}");

    // Convert to MEAI DataContent
    var dataContent = sampleImage.ToDataContent();
    Console.WriteLine($"DataContent: {dataContent.MediaType}, {dataContent.Data.Length} bytes");

    // Convert back to MLImage
    using var roundTripped = dataContent.ToMLImage();
    Console.WriteLine($"Round-tripped MLImage: {roundTripped.Width}x{roundTripped.Height}");
}

Console.WriteLine();

// --- Style 3: ML.NET IDataView Pipeline ---
Console.WriteLine("--- Style 3: ML.NET IDataView Pipeline ---");
Console.WriteLine();

if (imageFiles.Length > 0)
{
    var mlContext = new MLContext();

    // Load images into an IDataView
    var images = imageFiles.Select(f => MLImage.CreateFromFile(f)).ToArray();
    var sourceDataView = new ImageDataView(images);

    // Build embedding pipeline
    var embeddingPipeline = mlContext.Transforms.OnnxImageEmbedding(new OnnxImageEmbeddingOptions
    {
        ModelPath = modelPath,
        PreprocessorConfig = PreprocessorConfig.CLIP,
        Pooling = PoolingStrategy.ClsToken,
        Normalize = true
    });

    // Fit and transform
    using var embeddingModel = embeddingPipeline.Fit(sourceDataView);
    var transformed = embeddingModel.Transform(sourceDataView);

    // Read embedding results from the cursor
    var embeddingCol = transformed.Schema["Embedding"];
    using var cursor = transformed.GetRowCursor(transformed.Schema);
    var embeddingGetter = cursor.GetGetter<VBuffer<float>>(embeddingCol);

    int rowIndex = 0;
    while (cursor.MoveNext())
    {
        VBuffer<float> embedding = default;
        embeddingGetter(ref embedding);

        var values = embedding.GetValues().ToArray();
        Console.WriteLine($"  {Path.GetFileName(imageFiles[rowIndex])}: dim={values.Length}, [{string.Join(", ", values.Take(5).Select(v => v.ToString("F4")))}...]");
        rowIndex++;
    }

    foreach (var img in images) img.Dispose();
}
else
{
    Console.WriteLine("No test images found. Skipping IDataView pipeline demo.");
}

Console.WriteLine();
Console.WriteLine("Done!");

static float CosineSimilarity(float[] a, float[] b)
{
    float dot = 0, normA = 0, normB = 0;
    for (int i = 0; i < a.Length; i++)
    {
        dot += a[i] * b[i];
        normA += a[i] * a[i];
        normB += b[i] * b[i];
    }
    return dot / (MathF.Sqrt(normA) * MathF.Sqrt(normB));
}

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
