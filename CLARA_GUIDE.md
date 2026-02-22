# CLARA - iOS MedSigLIP Classifier

Guide for building an iOS app that runs MedSigLIP-448 on-device for zero-shot medical image classification with custom text labels.

## Prerequisites

- Xcode 16.2+
- iOS 17.0 deployment target
- Two CoreML models (already converted):
  - `MedSigLIP_VisionEncoder.mlpackage` (~200MB, float16)
  - `MedSigLIP_TextEncoder.mlpackage` (~200MB, float16)
- Tokenizer files from HuggingFace:
  - `tokenizer.json`
  - `tokenizer_config.json`

### Downloading tokenizer files

```bash
pip install huggingface-hub
huggingface-cli download google/medsiglip-448 \
  tokenizer.json tokenizer_config.json \
  --local-dir tokenizer_files
```

## Project Setup

### Using XcodeGen (project.yml)

```yaml
name: CLARA
options:
  bundleIdPrefix: com.clara
  deploymentTarget:
    iOS: "17.0"
  xcodeVersion: "16.2"

packages:
  swift-transformers:
    url: https://github.com/huggingface/swift-transformers
    from: "0.1.12"

targets:
  CLARA:
    type: application
    platform: iOS
    sources:
      - path: CLARA
        excludes:
          - "Resources/**"
      - path: CLARA/Resources/MedSigLIP_VisionEncoder.mlpackage
      - path: CLARA/Resources/MedSigLIP_TextEncoder.mlpackage
      - path: CLARA/Resources/tokenizer
        type: folder
        buildPhase: resources
    dependencies:
      - package: swift-transformers
        product: Transformers
    info:
      path: CLARA/Info.plist
      properties:
        UILaunchScreen: {}
        NSCameraUsageDescription: "Take photos for classification"
        NSAppTransportSecurity:
          NSAllowsLocalNetworking: true
    settings:
      base:
        SWIFT_VERSION: "5.9"
```

Generate the Xcode project:

```bash
xcodegen generate
```

### Manual Xcode setup (alternative)

1. File > New > Project > iOS App (SwiftUI)
2. Add Swift Package: `https://github.com/huggingface/swift-transformers` (v0.1.12+), import `Transformers` product
3. Drag both `.mlpackage` files into the project (Xcode auto-generates Swift classes)
4. Create a `tokenizer/` folder in Resources containing `tokenizer.json` and `tokenizer_config.json`, add as a folder reference
5. Add `NSCameraUsageDescription` and `NSAllowsLocalNetworking` to Info.plist

## Project Structure

```
CLARA/
  CLARA/
    App.swift                       # @main entry point
    ContentView.swift               # UI: image picker, label input, results
    MedSigLIPClassifier.swift       # Model loading, tokenization, inference
    ImagePreprocessor.swift         # CGImage -> MLMultiArray conversion
    EmbeddingAPIClient.swift        # Optional: server-side classification
    Info.plist
    Resources/
      MedSigLIP_VisionEncoder.mlpackage
      MedSigLIP_TextEncoder.mlpackage
      tokenizer/
        tokenizer.json
        tokenizer_config.json
  project.yml
```

## Source Files

### App.swift

Standard SwiftUI entry point.

```swift
import SwiftUI

@main
struct CLARA: App {
    var body: some Scene {
        WindowGroup {
            ContentView()
        }
    }
}
```

### ImagePreprocessor.swift

Converts a `CGImage` to the `MLMultiArray` format the vision encoder expects: `(1, 3, 448, 448)` float32, pixels normalized to [-1, 1].

```swift
import CoreML
import CoreGraphics

enum ImagePreprocessor {

    static let imageSize = 448

    static func preprocess(_ image: CGImage) throws -> MLMultiArray {
        let width = imageSize
        let height = imageSize

        guard let context = CGContext(
            data: nil,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: width * 4,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue
        ) else {
            throw PreprocessorError.contextCreationFailed
        }

        context.interpolationQuality = .high
        context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))

        guard let data = context.data else {
            throw PreprocessorError.noPixelData
        }

        let pixelCount = width * height
        let ptr = data.bindMemory(to: UInt8.self, capacity: pixelCount * 4)

        let array = try MLMultiArray(
            shape: [1, 3, NSNumber(value: height), NSNumber(value: width)],
            dataType: .float32
        )
        let outPtr = UnsafeMutablePointer<Float>(OpaquePointer(array.dataPointer))
        let channelStride = pixelCount

        for i in 0..<pixelCount {
            let baseIn = i * 4
            let r = Float(ptr[baseIn])     / 255.0 * 2.0 - 1.0
            let g = Float(ptr[baseIn + 1]) / 255.0 * 2.0 - 1.0
            let b = Float(ptr[baseIn + 2]) / 255.0 * 2.0 - 1.0

            outPtr[0 * channelStride + i] = r
            outPtr[1 * channelStride + i] = g
            outPtr[2 * channelStride + i] = b
        }

        return array
    }

    enum PreprocessorError: LocalizedError {
        case contextCreationFailed
        case noPixelData

        var errorDescription: String? {
            switch self {
            case .contextCreationFailed: return "Failed to create CGContext for image resize"
            case .noPixelData: return "No pixel data in CGContext"
            }
        }
    }
}
```

### MedSigLIPClassifier.swift

Core inference class. Loads both CoreML models + tokenizer, handles tokenization with attention masks, computes softmax-scaled similarity scores.

```swift
import CoreML
import Accelerate
import Tokenizers

@MainActor
final class MedSigLIPClassifier: ObservableObject {

    @Published var isLoading = true
    @Published var error: String?
    @Published var loadTimes: LoadTimes?

    private var visionModel: MLModel?
    private var textModel: MLModel?
    private var tokenizer: Tokenizer?

    private let maxLength = 64
    private let padTokenId: Int32 = 1  // </s>
    private let logitScale: Float = 10.0

    struct LoadTimes {
        let visionEncoder: TimeInterval
        let textEncoder: TimeInterval
        let tokenizer: TimeInterval
        var total: TimeInterval { visionEncoder + textEncoder + tokenizer }
    }

    func load() async {
        do {
            let config = MLModelConfiguration()
            config.computeUnits = .cpuAndNeuralEngine

            var start = CFAbsoluteTimeGetCurrent()
            let vision = try await MedSigLIP_VisionEncoder.load(configuration: config)
            let visionTime = CFAbsoluteTimeGetCurrent() - start
            visionModel = vision.model

            start = CFAbsoluteTimeGetCurrent()
            let text = try await MedSigLIP_TextEncoder.load(configuration: config)
            let textTime = CFAbsoluteTimeGetCurrent() - start
            textModel = text.model

            start = CFAbsoluteTimeGetCurrent()
            guard let tokenizerDir = Bundle.main.url(
                forResource: "tokenizer", withExtension: nil
            ) else {
                throw ClassifierError.tokenizerNotFound
            }
            tokenizer = try await AutoTokenizer.from(modelFolder: tokenizerDir)
            let tokenizerTime = CFAbsoluteTimeGetCurrent() - start

            loadTimes = LoadTimes(
                visionEncoder: visionTime,
                textEncoder: textTime,
                tokenizer: tokenizerTime
            )
            isLoading = false
        } catch {
            self.error = error.localizedDescription
            isLoading = false
        }
    }

    func imageEmbedding(for image: CGImage) async throws -> [Float] {
        guard let visionModel else { throw ClassifierError.notLoaded }

        return try await Task.detached(priority: .userInitiated) {
            let pixelValues = try ImagePreprocessor.preprocess(image)
            let input = try MLDictionaryFeatureProvider(
                dictionary: ["pixel_values": MLFeatureValue(multiArray: pixelValues)]
            )
            let output = try visionModel.prediction(from: input)
            guard let embeds = output.featureValue(for: "image_embeds")?.multiArrayValue else {
                throw ClassifierError.missingOutput("image_embeds")
            }
            return multiArrayToFloats(embeds)
        }.value
    }

    func classify(
        image: CGImage, labels: [String]
    ) async throws -> [(label: String, score: Float)] {
        guard let textModel, let tokenizer else { throw ClassifierError.notLoaded }

        let maxLen = maxLength
        let padId = padTokenId

        let imageEmbedding = try await imageEmbedding(for: image)

        let textEmbeddings: [[Float]] = try await Task.detached(priority: .userInitiated) {
            var results: [[Float]] = []
            for label in labels {
                let encoded = tokenizer(label)
                let tokenCount = encoded.count
                var ids = encoded.map { Int32($0) }

                if ids.count > maxLen {
                    ids = Array(ids.prefix(maxLen))
                } else {
                    ids.append(contentsOf: Array(repeating: padId, count: maxLen - ids.count))
                }

                let inputIds = try MLMultiArray(shape: [1, NSNumber(value: maxLen)], dataType: .int32)
                let idPtr = UnsafeMutablePointer<Int32>(OpaquePointer(inputIds.dataPointer))
                for (i, id) in ids.enumerated() { idPtr[i] = id }

                let realCount = min(tokenCount, maxLen)
                let attentionMask = try MLMultiArray(shape: [1, NSNumber(value: maxLen)], dataType: .int32)
                let maskPtr = UnsafeMutablePointer<Int32>(OpaquePointer(attentionMask.dataPointer))
                for i in 0..<maxLen { maskPtr[i] = i < realCount ? 1 : 0 }

                let input = try MLDictionaryFeatureProvider(dictionary: [
                    "input_ids": MLFeatureValue(multiArray: inputIds),
                    "attention_mask": MLFeatureValue(multiArray: attentionMask),
                ])
                let output = try textModel.prediction(from: input)
                guard let embeds = output.featureValue(for: "text_embeds")?.multiArrayValue else {
                    throw ClassifierError.missingOutput("text_embeds")
                }
                results.append(multiArrayToFloats(embeds))
            }
            return results
        }.value

        var rawScores: [Float] = []
        for i in labels.indices {
            rawScores.append(dotProduct(imageEmbedding, textEmbeddings[i]))
        }

        let scaledScores = rawScores.map { $0 * logitScale }
        let probs = softmax(scaledScores)

        var scored: [(label: String, score: Float)] = []
        for (i, label) in labels.enumerated() {
            scored.append((label: label, score: probs[i]))
        }
        return scored.sorted { $0.score > $1.score }
    }

    enum ClassifierError: LocalizedError {
        case tokenizerNotFound, notLoaded, missingOutput(String)
        var errorDescription: String? {
            switch self {
            case .tokenizerNotFound: return "Tokenizer files not found in app bundle"
            case .notLoaded: return "Models not loaded yet"
            case .missingOutput(let name): return "Missing expected output: \(name)"
            }
        }
    }
}

// MARK: - Free functions (no actor isolation)

private func multiArrayToFloats(_ array: MLMultiArray) -> [Float] {
    let count = array.count
    switch array.dataType {
    case .float32:
        var result = [Float](repeating: 0, count: count)
        let src = UnsafePointer<Float>(OpaquePointer(array.dataPointer))
        _ = result.withUnsafeMutableBufferPointer { dst in
            memcpy(dst.baseAddress!, src, count * MemoryLayout<Float>.size)
        }
        return result
    case .float16:
        let src = UnsafePointer<Float16>(OpaquePointer(array.dataPointer))
        return (0..<count).map { Float(src[$0]) }
    default:
        return (0..<count).map { array[$0].floatValue }
    }
}

private func dotProduct(_ a: [Float], _ b: [Float]) -> Float {
    var result: Float = 0
    vDSP_dotpr(a, 1, b, 1, &result, vDSP_Length(min(a.count, b.count)))
    return result
}

private func softmax(_ x: [Float]) -> [Float] {
    let maxVal = x.max() ?? 0
    let exps = x.map { exp($0 - maxVal) }
    let sum = exps.reduce(0, +)
    return exps.map { $0 / sum }
}
```

### ContentView.swift

SwiftUI interface with photo picker, camera capture, custom label input, and results display.

```swift
import SwiftUI
import PhotosUI

struct ContentView: View {
    @StateObject private var classifier = MedSigLIPClassifier()

    @State private var selectedPhoto: PhotosPickerItem?
    @State private var displayImage: UIImage?
    @State private var labelsText = "pneumonia, pleural effusion, normal chest x-ray"
    @State private var results: [(label: String, score: Float)] = []
    @State private var isClassifying = false
    @State private var showCamera = false
    @State private var errorMessage: String?

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(spacing: 20) {
                    imageSection
                    labelsSection
                    classifyButton

                    if let times = classifier.loadTimes {
                        loadTimesSection(times)
                    }

                    if let loadError = classifier.error {
                        Text("Load error: \(loadError)")
                            .foregroundStyle(.red)
                            .font(.caption)
                            .padding(.horizontal)
                    }
                    if let errorMessage {
                        Text(errorMessage)
                            .foregroundStyle(.red)
                            .font(.caption)
                            .padding(.horizontal)
                    }

                    if !results.isEmpty {
                        resultsSection
                    }
                }
                .padding()
            }
            .navigationTitle("CLARA")
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    if classifier.isLoading { ProgressView() }
                }
            }
        }
        .task { await classifier.load() }
        .onChange(of: selectedPhoto) {
            Task { await loadPhoto() }
        }
        .fullScreenCover(isPresented: $showCamera) {
            CameraView(image: $displayImage).ignoresSafeArea()
        }
    }

    // MARK: - Sections

    private var imageSection: some View {
        Group {
            if let displayImage {
                Image(uiImage: displayImage)
                    .resizable()
                    .aspectRatio(contentMode: .fit)
                    .frame(maxHeight: 300)
                    .clipShape(RoundedRectangle(cornerRadius: 12))
            } else {
                RoundedRectangle(cornerRadius: 12)
                    .fill(.quaternary)
                    .frame(height: 200)
                    .overlay {
                        VStack(spacing: 8) {
                            Image(systemName: "photo.on.rectangle").font(.largeTitle)
                            Text("Select or capture an image").font(.subheadline)
                        }
                        .foregroundStyle(.secondary)
                    }
            }

            HStack(spacing: 16) {
                PhotosPicker(selection: $selectedPhoto, matching: .images) {
                    Label("Photos", systemImage: "photo.on.rectangle")
                }
                .buttonStyle(.bordered)

                Button { showCamera = true } label: {
                    Label("Camera", systemImage: "camera")
                }
                .buttonStyle(.bordered)
            }
        }
    }

    private var labelsSection: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("Labels (comma-separated)")
                .font(.subheadline)
                .foregroundStyle(.secondary)
            TextField("e.g. pneumonia, normal, fracture", text: $labelsText)
                .textFieldStyle(.roundedBorder)
                .autocorrectionDisabled()
                .textInputAutocapitalization(.never)
        }
    }

    private var classifyButton: some View {
        Button {
            Task { await classify() }
        } label: {
            HStack {
                if isClassifying { ProgressView().tint(.white) }
                Text(isClassifying ? "Classifying..." : "Classify")
            }
            .frame(maxWidth: .infinity)
        }
        .buttonStyle(.borderedProminent)
        .disabled(displayImage == nil || isClassifying || classifier.isLoading || labelsText.isEmpty)
    }

    private var resultsSection: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Results").font(.headline)

            ForEach(Array(results.enumerated()), id: \.offset) { _, item in
                VStack(alignment: .leading, spacing: 4) {
                    HStack {
                        Text(item.label).font(.subheadline.weight(.medium))
                        Spacer()
                        Text(String(format: "%.1f%%", item.score * 100))
                            .font(.caption.monospaced())
                            .foregroundStyle(.secondary)
                    }
                    GeometryReader { geo in
                        RoundedRectangle(cornerRadius: 4)
                            .fill(.blue.gradient)
                            .frame(width: max(0, geo.size.width * CGFloat(item.score)))
                    }
                    .frame(height: 8)
                }
            }
        }
        .padding()
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 12))
    }

    private func loadTimesSection(_ times: MedSigLIPClassifier.LoadTimes) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Text("Model Load Times")
                .font(.caption.weight(.semibold))
                .foregroundStyle(.secondary)
            HStack(spacing: 16) {
                loadTimeLabel("Vision", times.visionEncoder)
                loadTimeLabel("Text", times.textEncoder)
                loadTimeLabel("Tokenizer", times.tokenizer)
            }
            Text(String(format: "Total: %.2fs", times.total))
                .font(.caption2.monospaced())
                .foregroundStyle(.tertiary)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(10)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 8))
    }

    private func loadTimeLabel(_ name: String, _ time: TimeInterval) -> some View {
        VStack(alignment: .leading, spacing: 1) {
            Text(name).font(.caption2).foregroundStyle(.tertiary)
            Text(String(format: "%.2fs", time)).font(.caption.monospaced()).foregroundStyle(.secondary)
        }
    }

    // MARK: - Actions

    private func loadPhoto() async {
        guard let selectedPhoto else { return }
        if let data = try? await selectedPhoto.loadTransferable(type: Data.self),
           let uiImage = UIImage(data: data) {
            displayImage = uiImage
            results = []
        }
    }

    private func classify() async {
        guard let cgImage = displayImage?.cgImage else { return }
        isClassifying = true
        errorMessage = nil

        do {
            let labels = labelsText
                .split(separator: ",")
                .map { $0.trimmingCharacters(in: .whitespaces) }
                .filter { !$0.isEmpty }
            guard !labels.isEmpty else {
                isClassifying = false
                return
            }
            results = try await classifier.classify(image: cgImage, labels: labels)
        } catch {
            errorMessage = error.localizedDescription
        }
        isClassifying = false
    }
}

// MARK: - Camera View

struct CameraView: UIViewControllerRepresentable {
    @Binding var image: UIImage?
    @Environment(\.dismiss) private var dismiss

    func makeUIViewController(context: Context) -> UIImagePickerController {
        let picker = UIImagePickerController()
        picker.sourceType = .camera
        picker.delegate = context.coordinator
        return picker
    }

    func updateUIViewController(_ uiViewController: UIImagePickerController, context: Context) {}
    func makeCoordinator() -> Coordinator { Coordinator(self) }

    class Coordinator: NSObject, UIImagePickerControllerDelegate, UINavigationControllerDelegate {
        let parent: CameraView
        init(_ parent: CameraView) { self.parent = parent }

        func imagePickerController(_ picker: UIImagePickerController,
                                   didFinishPickingMediaWithInfo info: [UIImagePickerController.InfoKey: Any]) {
            if let uiImage = info[.originalImage] as? UIImage { parent.image = uiImage }
            parent.dismiss()
        }

        func imagePickerControllerDidCancel(_ picker: UIImagePickerController) {
            parent.dismiss()
        }
    }
}
```

### EmbeddingAPIClient.swift (optional, for server mode)

```swift
import Foundation

actor EmbeddingAPIClient {

    struct ClassifyRequest: Encodable {
        let embedding: [Float]
        let top_k: Int
    }

    struct ClassifyResponse: Decodable {
        let results: [Result]
        struct Result: Decodable {
            let label: String
            let score: Float
        }
    }

    enum APIError: LocalizedError {
        case invalidURL
        case httpError(Int, String)
        case decodingFailed(String)

        var errorDescription: String? {
            switch self {
            case .invalidURL: return "Invalid server URL"
            case .httpError(let code, let body): return "Server error \(code): \(body)"
            case .decodingFailed(let detail): return "Failed to decode response: \(detail)"
            }
        }
    }

    func classify(
        embedding: [Float], topK: Int = 5, serverURL: String
    ) async throws -> [(label: String, score: Float)] {
        guard let url = URL(string: serverURL)?
            .appendingPathComponent("api")
            .appendingPathComponent("classify") else {
            throw APIError.invalidURL
        }

        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.timeoutInterval = 30
        request.httpBody = try JSONEncoder().encode(ClassifyRequest(embedding: embedding, top_k: topK))

        let (data, response) = try await URLSession.shared.data(for: request)

        if let http = response as? HTTPURLResponse, !(200..<300).contains(http.statusCode) {
            throw APIError.httpError(http.statusCode, String(data: data, encoding: .utf8) ?? "")
        }

        let decoded = try JSONDecoder().decode(ClassifyResponse.self, from: data)
        return decoded.results.map { ($0.label, $0.score) }
    }
}
```

## Model Details

| Property | Value |
|----------|-------|
| Base model | google/medsiglip-448 (SigLIP architecture) |
| Vision encoder | ~400M params, input: (1, 3, 448, 448) float32 |
| Text encoder | ~400M params, input: (1, 64) int32 + attention mask |
| Embedding dim | 1152 |
| CoreML format | ML Program, float16, iOS 17+ |
| Tokenizer | SentencePiece (loaded via swift-transformers) |
| Max text tokens | 64 |
| Pad token ID | 1 (`</s>`) |
| Logit scale | 10.0 |

## How It Works

1. User picks/captures an image
2. `ImagePreprocessor` resizes to 448x448, normalizes pixels to [-1, 1]
3. Vision encoder produces a 1152-dim L2-normalized image embedding
4. User's comma-separated labels are tokenized (SentencePiece), padded to 64 tokens with attention masks
5. Text encoder produces a 1152-dim L2-normalized embedding per label
6. Dot products (= cosine similarity since embeddings are normalized) are computed
7. Scores are scaled by `logitScale` (10.0) and passed through softmax
8. Results are displayed sorted by probability

## Performance Notes

- First run: ~30-40s model load (CoreML compiles for Neural Engine, cached after)
- Subsequent runs: ~2-5s model load
- Inference: ~100-200ms per classification on Apple Silicon
- App size: ~400MB (both models combined)
