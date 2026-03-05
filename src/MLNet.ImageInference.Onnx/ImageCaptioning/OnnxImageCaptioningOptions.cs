using MLNet.Image.Core;
using MLNet.ImageInference.Onnx.Shared;

namespace MLNet.ImageInference.Onnx.ImageCaptioning;

/// <summary>
/// Options for ONNX-based image captioning using GIT (Generative Image-to-text Transformer).
/// Requires two ONNX models: a vision encoder and a text decoder.
/// </summary>
public class OnnxImageCaptioningOptions : IOnnxImageOptions
{
    /// <summary>Path to the ONNX vision encoder model (pixel_values → visual_features).</summary>
    public required string EncoderModelPath { get; init; }

    /// <summary>Path to the ONNX text decoder model (input_ids + visual_features → logits).</summary>
    public required string DecoderModelPath { get; init; }

    /// <summary>Path to the BERT vocab.txt file for the WordPiece tokenizer.</summary>
    public required string VocabPath { get; init; }

    /// <summary>Name of the input column containing MLImage values.</summary>
    public string InputColumnName { get; init; } = "Image";

    /// <summary>Name of the output column for the generated caption text.</summary>
    public string OutputColumnName { get; init; } = "Caption";

    /// <summary>Preprocessing configuration. Defaults to GIT (CLIP normalization, 224x224).</summary>
    public PreprocessorConfig PreprocessorConfig { get; init; } = PreprocessorConfig.GIT;

    /// <summary>
    /// Maximum number of tokens to generate (excluding the BOS token).
    /// Default: 50. Typical captions are 5-20 tokens.
    /// </summary>
    public int MaxLength { get; init; } = 50;

    /// <summary>
    /// BOS (beginning of sequence) token ID. Default: 101 ([CLS] for BERT-based models).
    /// </summary>
    public int BosTokenId { get; init; } = 101;

    /// <summary>
    /// EOS (end of sequence) token ID. Default: 102 ([SEP] for BERT-based models).
    /// </summary>
    public int EosTokenId { get; init; } = 102;

    /// <summary>Batch size for lookahead batching in IDataView cursors.</summary>
    public int BatchSize { get; init; } = 32;
}
