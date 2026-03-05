namespace MLNet.ImageInference.Onnx.Tests;

/// <summary>
/// Provides test-skip functionality for xUnit v2 tests running under the v3 runner.
/// When the condition is not met, throws an exception with the <c>$XunitDynamicSkip$</c>
/// marker that the xUnit v3 runner interprets as a skipped test.
/// </summary>
internal static class Skip
{
    public static void Unless(bool condition, string reason)
    {
        if (!condition)
            throw new SkipException(reason);
    }
}

internal sealed class SkipException(string reason)
    : Exception($"$XunitDynamicSkip${reason}");
