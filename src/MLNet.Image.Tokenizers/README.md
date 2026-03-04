# MLNet.Image.Tokenizers

CLIP-specific tokenizer extensions for [Microsoft.ML.Tokenizers](https://www.nuget.org/packages/Microsoft.ML.Tokenizers).

## Purpose

Fills gaps in `Microsoft.ML.Tokenizers` for CLIP model tokenization. While `BpeTokenizer` can load CLIP's vocab.json/merges.txt, CLIP requires specific preprocessing:

- **Lowercase** all input text before tokenization
- **SOT/EOT** special tokens (`<|startoftext|>` = 49406, `<|endoftext|>` = 49407)
- **77-token context** length with zero-padding
- **Attention mask** generation

## Usage

```csharp
// Create from CLIP's vocab and merges files
var tokenizer = ClipTokenizer.Create("vocab.json", "merges.txt");

// Encode text — returns padded int[77] with SOT/EOT
int[] tokenIds = tokenizer.Encode("a photo of a cat");

// Create attention mask (1 for real tokens, 0 for padding)
int[] attentionMask = tokenizer.CreateAttentionMask(tokenIds);

// Batch encode
int[][] batch = tokenizer.EncodeBatch(["a cat", "a dog", "a bird"]);
```

## Dependencies

- `Microsoft.ML.Tokenizers` 2.0.0 — base `BpeTokenizer` for byte-level BPE
