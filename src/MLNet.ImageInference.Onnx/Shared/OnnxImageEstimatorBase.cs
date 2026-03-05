using Microsoft.ML;
using Microsoft.ML.Data;

namespace MLNet.ImageInference.Onnx.Shared;

/// <summary>
/// Base class for all ONNX image inference estimators.
/// Derived types only need to implement transformer creation and output schema configuration.
/// </summary>
/// <typeparam name="TTransformer">The transformer type produced by this estimator.</typeparam>
/// <typeparam name="TOptions">The options type for this estimator.</typeparam>
public abstract class OnnxImageEstimatorBase<TTransformer, TOptions> : IEstimator<TTransformer>
    where TTransformer : class, ITransformer
{
    protected readonly TOptions Options;

    protected OnnxImageEstimatorBase(TOptions options)
    {
        ArgumentNullException.ThrowIfNull(options);
        Options = options;
    }

    /// <summary>Create the transformer for this task.</summary>
    protected abstract TTransformer CreateTransformer();

    /// <summary>Add task-specific output columns to the schema shape dictionary.</summary>
    protected abstract void ConfigureOutputSchema(IDictionary<string, SchemaShape.Column> columns);

    public TTransformer Fit(IDataView input) => CreateTransformer();

    public SchemaShape GetOutputSchema(SchemaShape inputSchema)
    {
        var columns = inputSchema.ToDictionary(c => c.Name);
        ConfigureOutputSchema(columns);
        return new SchemaShape(columns.Values);
    }
}
