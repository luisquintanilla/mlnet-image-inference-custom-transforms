namespace MLNet.ImageInference.Onnx.Tests;

/// <summary>
/// Wraps Xunit.SkippableFact's Skip.IfNot so existing tests can keep calling Skip.Unless.
/// Tests using this must be decorated with [SkippableFact] instead of [Fact].
/// </summary>
internal static class Skip
{
    public static void Unless(bool condition, string reason)
    {
        Xunit.Skip.IfNot(condition, reason);
    }
}
