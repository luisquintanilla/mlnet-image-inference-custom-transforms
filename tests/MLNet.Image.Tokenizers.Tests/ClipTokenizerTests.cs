using MLNet.Image.Tokenizers;
using Xunit;

namespace MLNet.Image.Tokenizers.Tests;

public class ClipTokenizerTests : IDisposable
{
    private readonly string _vocabPath;
    private readonly string _mergesPath;
    private readonly ClipTokenizer _tokenizer;

    public ClipTokenizerTests()
    {
        _vocabPath = Path.Combine("TestData", "vocab.json");
        _mergesPath = Path.Combine("TestData", "merges.txt");
        _tokenizer = ClipTokenizer.Create(_vocabPath, _mergesPath);
    }

    public void Dispose()
    {
        // No unmanaged resources to dispose
    }

    [Fact]
    public void Create_FromFiles_Succeeds()
    {
        var tokenizer = ClipTokenizer.Create(_vocabPath, _mergesPath);
        Assert.NotNull(tokenizer);
    }

    [Fact]
    public void Create_FromStreams_Succeeds()
    {
        using var vocabStream = File.OpenRead(_vocabPath);
        using var mergesStream = File.OpenRead(_mergesPath);
        var tokenizer = ClipTokenizer.Create(vocabStream, mergesStream);
        Assert.NotNull(tokenizer);
    }

    [Fact]
    public void MaxContextLength_Is77()
    {
        Assert.Equal(77, _tokenizer.MaxContextLength);
    }

    [Fact]
    public void SotTokenId_Is49406()
    {
        Assert.Equal(49406, _tokenizer.SotTokenId);
    }

    [Fact]
    public void EotTokenId_Is49407()
    {
        Assert.Equal(49407, _tokenizer.EotTokenId);
    }

    [Fact]
    public void Encode_ReturnsArrayOfLength77()
    {
        var result = _tokenizer.Encode("a cat");
        Assert.Equal(77, result.Length);
    }

    [Fact]
    public void Encode_StartsWithSot()
    {
        var result = _tokenizer.Encode("a cat");
        Assert.Equal(49406, result[0]);
    }

    [Fact]
    public void Encode_HasEotAfterTokens()
    {
        var result = _tokenizer.Encode("a");
        // Find the first zero after index 0 — the position before it should be EOT
        // SOT at [0], at least one token, then EOT
        int eotIndex = Array.IndexOf(result, 49407);
        Assert.True(eotIndex > 0, "EOT token should be present after SOT and content tokens");
        Assert.True(eotIndex < 77, "EOT token should be within max context length");
    }

    [Fact]
    public void Encode_PadsWithZeros()
    {
        var result = _tokenizer.Encode("a");
        // Find EOT position, everything after should be 0
        int eotIndex = Array.IndexOf(result, 49407);
        for (int i = eotIndex + 1; i < result.Length; i++)
        {
            Assert.Equal(0, result[i]);
        }
    }

    [Fact]
    public void Encode_LowercasesInput()
    {
        var upper = _tokenizer.Encode("A CAT");
        var lower = _tokenizer.Encode("a cat");
        Assert.Equal(lower, upper);
    }

    [Fact]
    public void CreateAttentionMask_OnesForTokensZerosForPadding()
    {
        var encoded = _tokenizer.Encode("a");
        var mask = _tokenizer.CreateAttentionMask(encoded);

        Assert.Equal(encoded.Length, mask.Length);

        for (int i = 0; i < encoded.Length; i++)
        {
            int expected = encoded[i] != 0 ? 1 : 0;
            Assert.Equal(expected, mask[i]);
        }
    }

    [Fact]
    public void EncodeBatch_ReturnsCorrectCount()
    {
        var texts = new[] { "a", "a cat" };
        var results = _tokenizer.EncodeBatch(texts);

        Assert.Equal(2, results.Length);
        Assert.Equal(77, results[0].Length);
        Assert.Equal(77, results[1].Length);
    }

    [Fact]
    public void Decode_StripsSpecialTokensAndPadding()
    {
        var encoded = _tokenizer.Encode("a cat");
        var decoded = _tokenizer.Decode(encoded);

        Assert.NotNull(decoded);
        Assert.NotEmpty(decoded);
        // The decoded text should not contain SOT/EOT markers
        Assert.DoesNotContain("<|startoftext|>", decoded);
        Assert.DoesNotContain("<|endoftext|>", decoded);
    }

    [Fact]
    public void Encode_NullText_ThrowsArgumentNullException()
    {
        Assert.Throws<ArgumentNullException>(() => _tokenizer.Encode(null!));
    }
}
