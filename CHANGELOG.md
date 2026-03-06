# Changelog

All notable changes to this project will be documented in this file.

## [v0.4.0-preview.1] - 2026-03-05

### Added
- **GPU/ExecutionProvider support** — `OnnxExecutionProvider` enum (CPU, CUDA, DirectML, TensorRT) on all options
- **CancellationToken** on all public transformer methods (11 transformers, MEAI adapters)
- **Microsoft.Extensions.Logging** — `ILogger` property on options, model load timing, denoising step progress, GPU fallback warnings
- **Eager model validation** — `File.Exists` check in `OnnxSessionPool` and SD constructor with descriptive `FileNotFoundException`
- **FallbackToCpu** — graceful GPU → CPU degradation on provider initialization failure
- `OnnxSessionPool.CreateSessionOptions(IOnnxImageOptions)` factory for GPU-configured sessions

### Fixed
- `OnnxImageEmbeddingGenerator` now properly declares `: IDisposable` interface

### Changed
- `OnnxSessionPool` constructor eagerly validates model file existence (was lazy)
- All batch methods check `CancellationToken` between items
- SD denoising loop and captioning decode loop check cancellation per step

## [v0.3.0-preview.1] - 2026-03-05

### Added
- **Visual Question Answering (VQA)** — `AnswerQuestion(image, question)` on `OnnxImageCaptioningTransformer` using GIT-VQA model (`microsoft/git-base-textvqa`)
- **IChatClient VQA mode** — `OnnxImageCaptioningChatClient` auto-detects image+text → VQA, image-only → captioning
- **CLIP BPE tokenizer for Stable Diffusion** — `OnnxImageGenerationTransformer` now uses real CLIP tokenization (different prompts produce different images)
- `PreprocessorConfig.GITVQA` preset (CLIP normalization, 480×480)
- `VocabPath`/`MergesPath` properties on `OnnxImageGenerationOptions`
- VQA sample project (`samples/VisualQuestionAnswering/`)
- 8 VQA integration tests, 1 SD tokenizer integration test

### Changed
- `OnnxImageCaptioningTransformer.GenerateTokens()` now accepts initial token IDs (enables VQA question prefix)
- `OnnxImageCaptioningChatClient.GetResponseAsync()` extracts both image and text from messages
- TextToImage sample updated with CLIP tokenizer configuration

### Models
- GIT-base-TextVQA (encoder 332 MB + decoder 344 MB)

## [v0.2.0-preview.1] - 2026-03-05

### Added
- **SAM2 (Segment Anything Model v2)** — prompt-based segmentation with point, bounding box, and multi-point prompts
- **Image Captioning** — GIT (Generative Image-to-text Transformer) with autoregressive text generation
- **IChatClient adapter** — `OnnxImageCaptioningChatClient` implements `IChatClient` for vision-language tasks
- **Depth Estimation** — DPT-Hybrid (MiDaS) monocular depth estimation
- **Segment Anything embeddings** — `EncodeImage()` for encode-once-decode-many workflows
- `PreprocessorConfig.SAM2`, `PreprocessorConfig.GIT`, `PreprocessorConfig.DPT` presets
- `SegmentAnythingPrompt` with static factory methods (`FromPoint`, `FromPoints`, `FromBoundingBox`)
- Sample projects: ImageCaptioning, SegmentAnything, DepthEstimation, TextToImage

### Models
- GIT-base-COCO (encoder + decoder, ~675 MB total)
- SAM2 Hiera-Tiny (encoder 104 MB + decoder 16 MB)
- DPT-Hybrid MiDaS (508 MB)
- Stable Diffusion v1.4 (4.4 GB, exported via torch.onnx.export)

## [v0.1.0-preview.1] - 2026-03-05

### Added
- **Image Classification** — SqueezeNet, MobileNetV2, ResNet-50 with softmax + top-K
- **Object Detection** — YOLOv8 with NMS post-processing
- **Image Segmentation** — SegFormer, DeepLabV3 semantic segmentation
- **Image Embeddings** — CLIP, DINOv2 with MEAI `IEmbeddingGenerator<MLImage, Embedding<float>>`
- **Zero-Shot Classification** — CLIP vision+text cosine similarity
- Core library: `HuggingFaceImagePreprocessor`, `PreprocessorConfig`, `MLImage↔DataContent` extensions
- CLIP BPE tokenizer extending `Microsoft.ML.Tokenizers`
- Shared base classes: `OnnxImageCursorBase`, `OnnxImageDataViewBase`, `OnnxImageEstimatorBase`
- Lookahead batching for all tasks via `IDataView` cursors
- `MLContext.Transforms` extension methods for all tasks
- CI/CD: GitHub Actions build + release workflows
- 10 sample projects demonstrating all tasks

### Models
- SqueezeNet 1.0 (4.7 MB), MobileNetV2 (13.6 MB)
- YOLOv8n (12.2 MB)
- SegFormer-b0 (14.6 MB), DeepLabV3-ResNet50 (160 MB)
- CLIP ViT-B/32 (1.2 GB), DINOv2 ViT-S/14 (83 MB)
- ResNet-50 v1.7 (98 MB)
