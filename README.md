# CLARA_Call
An iOS app that demonstrates AI-powered clinical consultations using on-device medical vision models and a Data Collection Agent backend.

## Features

- **On-device medical image embeddings** via [MedSigLIP](https://huggingface.co/google/medsiglip-448) vision encoder converted to [CoreML](https://huggingface.co/JacobNewmes/coreml-medsiglip-448) — 448x448 input, 1152-dim embedding output, runs on CPU + Neural Engine
- **AI nurse consultations** powered by a Data Collection Agent backend — interprets visual findings and responds with text + TTS audio

## Requirements

- Xcode 16.2+
- [XcodeGen](https://github.com/yonaskolb/XcodeGen) (`brew install xcodegen`)
- iOS 18.0+ deployment target
- Python 3 with `huggingface_hub` installed (`pip install huggingface_hub`)
- ~815 MB disk space for CoreML model

## Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/JacobNewmes/CLARA_Call.git
cd CLARA_Call
```

### 2. Fetch CoreML models

The models are hosted on [Hugging Face Hub](https://huggingface.co/JacobNewmes/coreml-medsiglip-448) and are not checked into git. Run the fetch script to download them:

```bash
./scripts/fetch-models.sh
```

This downloads into `CLARA/Resources/`:
| Model | Size | Purpose |
|-------|------|---------|
| `MedSigLIP_VisionEncoder.mlpackage` | 815 MB | Medical image embedding |

### 3. Generate the Xcode project

The `.xcodeproj` is not checked in — it's generated from `project.yml`:

```bash
xcodegen generate
open CLARA.xcodeproj
```

### 4. Configure environment variables

In Xcode, go to **Product > Scheme > Edit Sceme > Run > Environment Variables** and add:

| Variable | Description |
|----------|-------------|
| `GEMINI_API_KEY` | Required for generating TTS(text-to-speech) audio file on Data Collection Agent backend server |

> **Note:** If the API key has reached its daily quota, the demo will not be functional.

### 5. Build and run

In Xcode, go to **Signing & Capabilities** and select your development team to sign the provisioning profile. Then select a physical iOS device (CoreML models require Neural Engine — Simulator won't work well) and hit **Cmd+R**.

On first launch, CoreML compiles the models for your device's Neural Engine. This takes about 20 seconds (on iPhone 15 Pro) and is cached for subsequent runs.

## Project Structure

```
CLARA/
  App.swift                    # Entry point
  HomeView.swift               # Incoming call UI, model loading
  LiveStreamView.swift         # Media panel + chat UI
  DemoOrchestrator.swift       # 3-tap demo state machine + audio/video playback
  WebRTCManager.swift          # Messaging, Data Collection Agent integration
  CareAIClient.swift           # Data Collection Agent backend HTTP client
  MedSigLIPClassifier.swift    # Vision encoder loading + inference
  ImagePreprocessor.swift      # CGImage -> MLMultiArray (448x448)
  ChatMessage.swift            # Chat message model
  Resources/
    MedSigLIP_VisionEncoder.mlpackage/  # (fetched via script)
  demo_files/
  clara_question.wav           # CLARA's opening question audio
  clara_answer.wav             # CLARA's follow-up audio
  p_reply_1.mp4                # Patient reply video 1
  p_reply_2.mp4                # Patient reply video 2
  p_reply_3.mp4                # Patient reply video 3
scripts/
  fetch-models.sh              # Downloads models from HF Hub
```

## How It Works

1. **Startup** — The app loads MedSigLIP VisionEncoder onto the Neural Engine. An incoming call screen is shown while loading; once ready, CLARA "calls" the user.
2. **Accept call** — User taps Accept to enter the consultation. CLARA asks an opening question via audio.
3. **3-tap demo** — Each tap plays a pre-recorded patient video reply. After the first reply, audio and a mid-frame embedding are extracted and sent to the Data Collection Agent backend, which returns a nurse response with TTS audio.

