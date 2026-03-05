namespace MLNet.ImageInference.Onnx.Shared;

/// <summary>
/// Common properties shared by all ONNX image inference options classes.
/// </summary>
public interface IOnnxImageOptions
{
    /// <summary>Name of the IDataView column containing the input MLImage.</summary>
    string InputColumnName { get; }

    /// <summary>Maximum number of images to batch in a single ONNX inference call.</summary>
    int BatchSize { get; }
}
