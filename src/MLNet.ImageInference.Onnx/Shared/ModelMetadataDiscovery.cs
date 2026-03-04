using Microsoft.ML.OnnxRuntime;

namespace MLNet.ImageInference.Onnx.Shared;

/// <summary>
/// Discovers ONNX model metadata: input/output names, shapes, and data types.
/// Used to auto-configure preprocessing and post-processing pipelines.
/// </summary>
public static class ModelMetadataDiscovery
{
    /// <summary>
    /// Discovered metadata about an ONNX model's inputs and outputs.
    /// </summary>
    public record ModelMetadata
    {
        public required string[] InputNames { get; init; }
        public required long[][] InputShapes { get; init; }
        public required string[] OutputNames { get; init; }
        public required long[][] OutputShapes { get; init; }

        /// <summary>
        /// Gets whether the model's first input dimension is dynamic (supports batch > 1).
        /// A dimension value of -1 indicates a dynamic axis.
        /// </summary>
        public bool IsBatchDynamic => InputShapes.Length > 0
            && InputShapes[0].Length > 0
            && InputShapes[0][0] == -1;
    }

    /// <summary>
    /// Discover input/output metadata from an ONNX model file.
    /// </summary>
    public static ModelMetadata Discover(string modelPath)
    {
        ArgumentException.ThrowIfNullOrEmpty(modelPath);

        using var session = new InferenceSession(modelPath);
        return Discover(session);
    }

    /// <summary>
    /// Discover input/output metadata from an existing InferenceSession.
    /// </summary>
    public static ModelMetadata Discover(InferenceSession session)
    {
        ArgumentNullException.ThrowIfNull(session);

        var inputNames = session.InputNames.ToArray();
        var inputShapes = session.InputMetadata.Values
            .Select(m => m.Dimensions.Select(d => (long)d).ToArray())
            .ToArray();

        var outputNames = session.OutputNames.ToArray();
        var outputShapes = session.OutputMetadata.Values
            .Select(m => m.Dimensions.Select(d => (long)d).ToArray())
            .ToArray();

        return new ModelMetadata
        {
            InputNames = inputNames,
            InputShapes = inputShapes,
            OutputNames = outputNames,
            OutputShapes = outputShapes
        };
    }
}
