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
