# Extending the Library

> How to add new image tasks following the established six-component pattern.

## Overview

Every image inference task in this library follows the same six-component pattern. This guide walks through adding a hypothetical new task — **image captioning** — to illustrate each step.

The six components are:

| # | Component | File | Purpose |
|---|---|---|---|
| 1 | **Options** | `OnnxImage{Task}Options.cs` | Configuration: model path, column names, preprocessor config, task-specific settings |
| 2 | **PostProcessor** | (inline in transformer or separate class) | Convert raw ONNX output to task-specific result type |
| 3 | **Estimator** | `OnnxImage{Task}Estimator.cs` | `IEstimator<T>` that creates the transformer via `Fit()` |
| 4 | **Transformer** | `OnnxImage{Task}Transformer.cs` | `ITransformer` that owns the ONNX session and runs the full pipeline |
| 5 | **MLContext extension** | `MLContextExtensions.cs` | Extension method on `TransformsCatalog` for pipeline composition |
| 6 | **MEAI adapter** | `OnnxImage{Task}Generator.cs` | (Optional) Implements an MEAI interface like `IEmbeddingGenerator` |

---

## Step 1: Create the Options Class

Create a new folder under `src/MLNet.ImageInference.Onnx/` for your task, then add the options class.

```
src/MLNet.ImageInference.Onnx/
  Captioning/
    OnnxImageCaptioningOptions.cs    ← new
```

```csharp
using MLNet.Image.Core;

namespace MLNet.ImageInference.Onnx.Captioning;

public class OnnxImageCaptioningOptions
{
    /// <summary>Path to the ONNX model file.</summary>
    public required string ModelPath { get; init; }

    /// <summary>Name of the input column containing MLImage values.</summary>
    public string InputColumnName { get; init; } = "Image";

    /// <summary>Name of the output column for the generated caption.</summary>
    public string CaptionColumnName { get; init; } = "Caption";

    /// <summary>Preprocessing configuration.</summary>
    public PreprocessorConfig PreprocessorConfig { get; init; } = PreprocessorConfig.ImageNet;

    /// <summary>Maximum caption length in tokens.</summary>
    public int MaxLength { get; init; } = 50;
}
```

**Conventions:**
- Use `required` for `ModelPath` — it's always mandatory.
- Default `InputColumnName` to `"Image"` — matches ML.NET's `LoadImages` output.
- Default `PreprocessorConfig` to the most common preset for the task.
- Use `init`-only properties to ensure immutability after construction.

---

## Step 2: Define the Result Type (if needed)

If your task produces a new output shape, add a result type to `MLNet.Image.Core`:

```csharp
// In MLNet.Image.Core (only if no existing type fits)
namespace MLNet.Image.Core;

public record CaptionResult
{
    public string Caption { get; init; } = string.Empty;
    public float Confidence { get; init; }
}
```

Existing result types:
- `(string Label, float Probability)[]` — classification
- `float[]` — embeddings
- `BoundingBox` — detection
- `SegmentationMask` — segmentation

---

## Step 3: Create the Transformer

The transformer is the core component. It owns the `OnnxSessionPool` and implements the full three-stage pipeline.

```csharp
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using MLNet.Image.Core;
using MLNet.ImageInference.Onnx.Shared;

namespace MLNet.ImageInference.Onnx.Captioning;

public sealed class OnnxImageCaptioningTransformer : ITransformer, IDisposable
{
    private readonly OnnxImageCaptioningOptions _options;
    private readonly OnnxSessionPool _sessionPool;
    private readonly ModelMetadataDiscovery.ModelMetadata _metadata;

    public bool IsRowToRowMapper => false;

    public OnnxImageCaptioningTransformer(OnnxImageCaptioningOptions options)
    {
        _options = options;
        _sessionPool = new OnnxSessionPool(options.ModelPath);
        _metadata = ModelMetadataDiscovery.Discover(_sessionPool.Session);
    }

    /// <summary>
    /// Convenience API: generate a caption for a single image.
    /// </summary>
    public string Caption(MLImage image)
    {
        // Stage 1: Preprocess
        var tensor = HuggingFaceImagePreprocessor.Preprocess(image, _options.PreprocessorConfig);
        int height = _options.PreprocessorConfig.ImageSize.Height;
        int width = _options.PreprocessorConfig.ImageSize.Width;

        // Stage 2: ONNX Score
        var inputTensor = new DenseTensor<float>(tensor, [1, 3, height, width]);
        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor(_metadata.InputNames[0], inputTensor)
        };
        using var results = _sessionPool.Session.Run(inputs);

        // Stage 3: PostProcess (task-specific)
        var output = results.First().AsEnumerable<float>().ToArray();
        return DecodeCaption(output); // Your post-processing logic
    }

    private string DecodeCaption(float[] output)
    {
        // TODO: Implement token decoding (greedy/beam search)
        throw new NotImplementedException();
    }

    // --- Required ITransformer members ---

    public IDataView Transform(IDataView input)
    {
        throw new NotImplementedException(
            "Full IDataView Transform is under development. "
            + "Use the Caption() method for single-image captioning.");
    }

    public DataViewSchema GetOutputSchema(DataViewSchema inputSchema)
    {
        var builder = new DataViewSchema.Builder();
        builder.AddColumn(_options.CaptionColumnName, TextDataViewType.Instance);
        return builder.ToSchema();
    }

    public IRowToRowMapper GetRowToRowMapper(DataViewSchema inputSchema)
        => throw new InvalidOperationException(
            "This transformer does not support row-to-row mapping.");

    void ICanSaveModel.Save(ModelSaveContext ctx)
        => throw new NotSupportedException(
            "Use transformer-specific save/load instead of mlContext.Model.Save().");

    public void Dispose() => _sessionPool?.Dispose();
}
```

**Key patterns to follow:**
- Always use `OnnxSessionPool` — never create `InferenceSession` directly.
- Use `ModelMetadataDiscovery.Discover()` to auto-detect input/output names.
- Provide a typed convenience method (`Caption()`, `Classify()`, `GenerateEmbedding()`) alongside `Transform(IDataView)`.
- Implement `IDisposable` to clean up the session pool.

---

## Step 4: Create the Estimator

The estimator is thin — it just creates the transformer.

```csharp
using Microsoft.ML;
using Microsoft.ML.Data;
using MLNet.ImageInference.Onnx.Shared;

namespace MLNet.ImageInference.Onnx.Captioning;

public sealed class OnnxImageCaptioningEstimator
    : IEstimator<OnnxImageCaptioningTransformer>
{
    private readonly OnnxImageCaptioningOptions _options;

    public OnnxImageCaptioningEstimator(OnnxImageCaptioningOptions options)
    {
        ArgumentNullException.ThrowIfNull(options);
        _options = options;
    }

    public OnnxImageCaptioningTransformer Fit(IDataView input)
    {
        return new OnnxImageCaptioningTransformer(_options);
    }

    public SchemaShape GetOutputSchema(SchemaShape inputSchema)
    {
        var columns = inputSchema.ToDictionary(c => c.Name);

        columns[_options.CaptionColumnName] = SchemaShapeHelper.CreateColumn(
            _options.CaptionColumnName,
            SchemaShape.Column.VectorKind.Scalar,
            TextDataViewType.Instance,
            isKey: false);

        return new SchemaShape(columns.Values);
    }
}
```

**Notes:**
- `Fit()` doesn't train anything — it just constructs the transformer with the provided options.
- `GetOutputSchema()` declares what columns the transformer will produce, using `SchemaShapeHelper` to work around ML.NET's internal constructor.

---

## Step 5: Add the MLContext Extension Method

Add a new method to `MLContextExtensions.cs`:

```csharp
// In MLContextExtensions.cs — add this method:

/// <summary>
/// Create an image captioning pipeline: preprocess → ONNX score → decode → caption.
/// </summary>
public static OnnxImageCaptioningEstimator OnnxImageCaptioning(
    this TransformsCatalog catalog,
    OnnxImageCaptioningOptions options)
{
    ArgumentNullException.ThrowIfNull(options);
    return new OnnxImageCaptioningEstimator(options);
}
```

Add the necessary `using` for the new namespace at the top of the file:

```csharp
using MLNet.ImageInference.Onnx.Captioning;
```

This gives users the natural ML.NET entry point:

```csharp
var pipeline = mlContext.Transforms.OnnxImageCaptioning(new OnnxImageCaptioningOptions
{
    ModelPath = "models/blip/model.onnx"
});
```

---

## Step 6: Add MEAI Adapter (Optional)

If your task maps to an MEAI abstraction, create an adapter in `MEAI/`:

```csharp
// Only if there's a matching MEAI interface for your task.
// For captioning, there isn't a standard one yet, so you'd skip this step.
// For embeddings, the adapter implements IEmbeddingGenerator<MLImage, Embedding<float>>.
```

Current MEAI adapters:

| Task | MEAI Interface | Adapter |
|---|---|---|
| Embeddings | `IEmbeddingGenerator<MLImage, Embedding<float>>` | `OnnxImageEmbeddingGenerator` |
| Generation | `IImageGenerator` (future) | (planned) |

---

## Checklist for Adding a New Task

- [ ] Create `src/MLNet.ImageInference.Onnx/{TaskName}/` folder
- [ ] Add `OnnxImage{Task}Options.cs` with `required string ModelPath` and sensible defaults
- [ ] Add result type to `MLNet.Image.Core` if no existing type fits
- [ ] Add `OnnxImage{Task}Transformer.cs` implementing `ITransformer, IDisposable`
  - [ ] Use `OnnxSessionPool` for session management
  - [ ] Use `ModelMetadataDiscovery.Discover()` for input/output names
  - [ ] Use `HuggingFaceImagePreprocessor.Preprocess()` for stage 1
  - [ ] Implement task-specific post-processing for stage 3
  - [ ] Add typed convenience method (e.g., `Caption()`)
- [ ] Add `OnnxImage{Task}Estimator.cs` implementing `IEstimator<T>`
- [ ] Add extension method to `MLContextExtensions.cs`
- [ ] (Optional) Add MEAI adapter in `MEAI/` folder
- [ ] Add a sample app under `samples/{TaskName}/`
- [ ] Update `README.md` task status table
