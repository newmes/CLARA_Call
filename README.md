# CLARA

**Clinical Longitudinal AI Research Assistant** — an iOS app that runs on-device medical AI models for real-time visual assessment and speech recognition during live patient consultations.

## Features

- **On-device medical image embeddings** via [MedSigLIP](https://huggingface.co/google/medsiglip-448) vision encoder (CoreML)
- **On-device speech recognition** via [MedASR](https://huggingface.co/google/medasr) (CoreML + CTC decoding)
- **Live camera streaming** with WebRTC
- **AI nurse consultations** powered by a Care AI backend — interprets visual findings and patient speech in real time

## Requirements

- Xcode 16.2+
- [XcodeGen](https://github.com/yonaskolb/XcodeGen) (`brew install xcodegen`)
- iOS 18.0+ deployment target
- Python 3 with `huggingface_hub` installed (`pip install huggingface_hub`)
- ~1.3 GB disk space for CoreML models

## Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/JacobNewmes/CLARA.git
cd CLARA
```

### 2. Fetch CoreML models

The models are hosted on [Hugging Face Hub](https://huggingface.co/JacobNewmes/CLARA-CoreML) and are not checked into git. Run the fetch script to download them:

```bash
./scripts/fetch-models.sh
```

This downloads into `CLARA/Resources/`:
| Model | Size | Purpose |
|-------|------|---------|
| `MedASR.mlpackage` | 403 MB | Medical speech recognition |
| `MedSigLIP_VisionEncoder.mlpackage` | 815 MB | Medical image embedding |

### 3. Generate the Xcode project

The `.xcodeproj` is not checked in — it's generated from `project.yml`:

```bash
xcodegen generate
open CLARA.xcodeproj
```

### 4. Build and run

Select a physical iOS device (CoreML models require Neural Engine — Simulator won't work well) and hit **Cmd+R**.

On first launch, CoreML compiles the models for your device's Neural Engine. This takes 30-40 seconds and is cached for subsequent runs.

## Project Structure

```
CLARA/
  App.swift                    # Entry point
  HomeView.swift               # Home screen, model loading
  LiveStreamView.swift         # Live camera + chat UI
  WebRTCManager.swift          # Camera capture, Care AI integration
  CareAIClient.swift           # Care AI backend HTTP client
  AudioTranscriber.swift       # Real-time speech-to-text pipeline
  MedSigLIPClassifier.swift    # Vision encoder loading + inference
  ImagePreprocessor.swift      # CGImage -> MLMultiArray (448x448)
  MelSpectrogramExtractor.swift # Audio -> mel spectrogram for ASR
  CTCDecoder.swift             # CTC beam search for ASR output
  ChatMessage.swift            # Chat message model
  Resources/
    MedASR.mlpackage/          # (fetched via script)
    MedSigLIP_VisionEncoder.mlpackage/  # (fetched via script)
    medasr_tokenizer/          # ASR vocabulary (committed)
scripts/
  fetch-models.sh              # Downloads models from HF Hub
```

## How It Works

1. **Startup** — The app loads MedASR and MedSigLIP VisionEncoder onto the Neural Engine in parallel
2. **Live session** — Camera feed streams via WebRTC; every 5 seconds a frame is grabbed and run through the vision encoder to produce a 1152-dim medical image embedding
3. **Speech** — Audio is captured, converted to mel spectrograms, and decoded on-device via MedASR with CTC decoding
4. **Consultation** — The image embedding + transcribed patient speech are sent to the Care AI backend, which returns a structured nurse response with visual assessment findings and follow-up questions
5. **Playback** — The backend response includes TTS audio that plays back through the app

## License

All rights reserved.
