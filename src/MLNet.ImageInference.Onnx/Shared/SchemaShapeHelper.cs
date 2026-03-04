using System.Reflection;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace MLNet.ImageInference.Onnx.Shared;

/// <summary>
/// Helper to create SchemaShape.Column instances via reflection,
/// since the constructor is internal in Microsoft.ML.
/// </summary>
internal static class SchemaShapeHelper
{
    private static readonly ConstructorInfo s_columnCtor =
        typeof(SchemaShape.Column)
            .GetConstructors(BindingFlags.NonPublic | BindingFlags.Instance)[0];

    internal static SchemaShape.Column CreateColumn(
        string name,
        SchemaShape.Column.VectorKind kind,
        DataViewType itemType,
        bool isKey)
    {
        return (SchemaShape.Column)s_columnCtor.Invoke([name, kind, itemType, isKey, null!]);
    }
}
