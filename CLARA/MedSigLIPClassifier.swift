import CoreML
import Accelerate

@MainActor
final class MedSigLIPClassifier: ObservableObject {

    @Published var isLoading = true
    @Published var error: String?
    @Published var loadTime: TimeInterval?

    private var visionModel: MLModel?

    func load() async {
        do {
            let config = MLModelConfiguration()
            config.computeUnits = .cpuAndNeuralEngine

            let start = CFAbsoluteTimeGetCurrent()
            let vision = try await MedSigLIP_VisionEncoder.load(configuration: config)
            loadTime = CFAbsoluteTimeGetCurrent() - start
            visionModel = vision.model

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

    enum ClassifierError: LocalizedError {
        case notLoaded, missingOutput(String)
        var errorDescription: String? {
            switch self {
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
