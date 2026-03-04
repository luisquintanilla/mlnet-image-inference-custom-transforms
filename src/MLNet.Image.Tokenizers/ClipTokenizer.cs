using Microsoft.ML.Tokenizers;

namespace MLNet.Image.Tokenizers;

/// <summary>
/// CLIP-specific tokenizer that wraps Microsoft.ML.Tokenizers.BpeTokenizer
/// with CLIP's required preprocessing behavior.
///
/// CLIP's tokenizer differs from standard BPE in:
/// - Text is lowercased before tokenization
/// - SOT (start of text) token: &lt;|startoftext|&gt; (ID 49406)
/// - EOT (end of text) token: &lt;|endoftext|&gt; (ID 49407)  
/// - Max context length: 77 tokens (including SOT/EOT)
/// - Padding to max length with zeros
///
/// DESIGN NOTE: This is a convenience wrapper until Microsoft.ML.Tokenizers
/// adds native CLIP tokenizer support. When that happens, this class can be deprecated.
/// </summary>
public sealed class ClipTokenizer
{
    private readonly Tokenizer _inner;
    private readonly int _sotTokenId;
    private readonly int _eotTokenId;

    /// <summary>
    /// Maximum context length for CLIP models (including SOT and EOT tokens).
    /// </summary>
    public int MaxContextLength { get; }

    /// <summary>
    /// The SOT (start of text) token ID.
    /// </summary>
    public int SotTokenId => _sotTokenId;

    /// <summary>
    /// The EOT (end of text) token ID.
    /// </summary>
    public int EotTokenId => _eotTokenId;

    private ClipTokenizer(Tokenizer inner, int sotTokenId, int eotTokenId, int maxContextLength)
    {
        _inner = inner;
        _sotTokenId = sotTokenId;
        _eotTokenId = eotTokenId;
        MaxContextLength = maxContextLength;
    }

    /// <summary>
    /// Create a ClipTokenizer from CLIP's vocab.json and merges.txt files.
    /// These files can be downloaded from HuggingFace, e.g.:
    ///   openai/clip-vit-base-patch32/vocab.json
    ///   openai/clip-vit-base-patch32/merges.txt
    /// </summary>
    /// <param name="vocabPath">Path to vocab.json file.</param>
    /// <param name="mergesPath">Path to merges.txt file.</param>
    /// <param name="sotTokenId">SOT token ID (default: 49406 for standard CLIP).</param>
    /// <param name="eotTokenId">EOT token ID (default: 49407 for standard CLIP).</param>
    /// <param name="maxContextLength">Maximum context length (default: 77 for CLIP).</param>
    public static ClipTokenizer Create(
        string vocabPath,
        string mergesPath,
        int sotTokenId = 49406,
        int eotTokenId = 49407,
        int maxContextLength = 77)
    {
        ArgumentException.ThrowIfNullOrEmpty(vocabPath);
        ArgumentException.ThrowIfNullOrEmpty(mergesPath);

        using var vocabStream = File.OpenRead(vocabPath);
        using var mergesStream = File.OpenRead(mergesPath);

        return Create(vocabStream, mergesStream, sotTokenId, eotTokenId, maxContextLength);
    }

    /// <summary>
    /// Create a ClipTokenizer from vocab and merges streams.
    /// </summary>
    public static ClipTokenizer Create(
        Stream vocabStream,
        Stream mergesStream,
        int sotTokenId = 49406,
        int eotTokenId = 49407,
        int maxContextLength = 77)
    {
        ArgumentNullException.ThrowIfNull(vocabStream);
        ArgumentNullException.ThrowIfNull(mergesStream);

        var inner = BpeTokenizer.Create(vocabStream, mergesStream);
        return new ClipTokenizer(inner, sotTokenId, eotTokenId, maxContextLength);
    }

    /// <summary>
    /// Encode text to token IDs with CLIP-specific preprocessing:
    /// 1. Lowercase the text
    /// 2. Tokenize with BPE
    /// 3. Prepend SOT token
    /// 4. Append EOT token
    /// 5. Truncate to MaxContextLength
    /// 6. Pad to MaxContextLength with zeros
    /// </summary>
    /// <param name="text">The input text to encode.</param>
    /// <returns>Token IDs array of length MaxContextLength (padded with zeros).</returns>
    public int[] Encode(string text)
    {
        ArgumentNullException.ThrowIfNull(text);

        // CLIP lowercases all text
        string lowered = text.ToLowerInvariant();

        // Tokenize with the underlying BPE tokenizer
        // MaxContextLength - 2 accounts for SOT and EOT tokens
        var tokenIds = _inner.EncodeToIds(lowered, MaxContextLength - 2, out _, out _);

        // Build the full token sequence: [SOT] + tokens + [EOT] + padding
        var result = new int[MaxContextLength];
        result[0] = _sotTokenId;

        int tokenCount = Math.Min(tokenIds.Count, MaxContextLength - 2);
        for (int i = 0; i < tokenCount; i++)
        {
            result[i + 1] = tokenIds[i];
        }

        result[tokenCount + 1] = _eotTokenId;
        // Remaining positions are already 0 (padding)

        return result;
    }

    /// <summary>
    /// Encode a batch of texts to token ID arrays.
    /// </summary>
    /// <param name="texts">The input texts to encode.</param>
    /// <returns>Array of token ID arrays, each of length MaxContextLength.</returns>
    public int[][] EncodeBatch(IReadOnlyList<string> texts)
    {
        ArgumentNullException.ThrowIfNull(texts);

        var results = new int[texts.Count][];
        for (int i = 0; i < texts.Count; i++)
        {
            results[i] = Encode(texts[i]);
        }
        return results;
    }

    /// <summary>
    /// Decode token IDs back to text.
    /// Strips SOT/EOT tokens and padding zeros.
    /// </summary>
    public string Decode(IEnumerable<int> ids)
    {
        // Filter out SOT, EOT, and padding (0) tokens
        var filtered = ids.Where(id => id != _sotTokenId && id != _eotTokenId && id != 0);
        return _inner.Decode(filtered) ?? string.Empty;
    }

    /// <summary>
    /// Create an attention mask for encoded token IDs.
    /// Returns 1 for real tokens, 0 for padding.
    /// </summary>
    public int[] CreateAttentionMask(int[] tokenIds)
    {
        var mask = new int[tokenIds.Length];
        for (int i = 0; i < tokenIds.Length; i++)
        {
            mask[i] = tokenIds[i] != 0 ? 1 : 0;
        }
        return mask;
    }
}
