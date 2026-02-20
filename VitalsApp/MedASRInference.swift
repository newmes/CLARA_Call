@preconcurrency import CoreML
import Foundation

/// CoreML model wrapper for MedASR inference.
///
/// Handles model loading, input preparation, prediction, and Float16 -> Float32
/// output conversion. All inference runs off the main thread.
///
/// Usage:
///     let inference = try await MedASRInference.load()
///     let logits = try await inference.predict(features: melFeatures, attentionMask: mask)
///     let tokenIDs = CTCDecoder.greedyDecode(logits: logits)
///     let text = decoder.decode(tokenIDs: tokenIDs)
final class MedASRInference {

    // MARK: - Configuration

    /// Fixed mel frame length the CoreML model expects (30 seconds at 16kHz).
    /// All inputs must be padded to this length; use the attention mask to
    /// indicate which frames are real vs padding.
    static let modelMelLength: Int = 3000

    /// Vocabulary size (512 SentencePiece tokens)
    static let vocabSize: Int = 512

    /// Hidden size of the encoder
    static let hiddenSize: Int = 512

    // MARK: - Properties

    private let model: MLModel

    // MARK: - Initialization

    private init(model: MLModel) {
        self.model = model
    }

    /// Load the MedASR CoreML model.
    ///
    /// First launch compiles the model for the Neural Engine (~30-40s, cached after).
    /// Subsequent launches take ~2-5s.
    ///
    /// - Parameter computeUnits: Compute units to use. Default is `.cpuAndNeuralEngine`.
    /// - Returns: A configured MedASRInference instance.
    static func load(
        computeUnits: MLComputeUnits = .cpuAndNeuralEngine
    ) async throws -> MedASRInference {
        let config = MLModelConfiguration()
        config.computeUnits = computeUnits

        // Load from the compiled model in the app bundle
        guard let modelURL = Bundle.main.url(forResource: "MedASR", withExtension: "mlmodelc") else {
            throw MedASRError.modelNotFound
        }

        let model = try await Task.detached(priority: .userInitiated) {
            try MLModel(contentsOf: modelURL, configuration: config)
        }.value

        return MedASRInference(model: model)
    }

    // MARK: - Prediction

    /// Run MedASR inference on precomputed mel spectrogram features.
    ///
    /// - Parameters:
    ///   - features: Log mel spectrogram `MLMultiArray` of shape `[1, 3000, 128]`.
    ///               Must be padded to `modelMelLength` (3000 frames).
    ///   - attentionMask: Binary mask `MLMultiArray` of shape `[1, 3000]`.
    ///                    1 = valid frame, 0 = padding.
    ///
    /// - Returns: CTC logits as `[Float]` of length `T' * vocabSize`, where
    ///            T' is the encoder output length (approximately 3000 / 4 after subsampling).
    ///            Logits are in row-major order: `[time_step_0_token_0, ..., time_step_0_token_511, time_step_1_token_0, ...]`
    func predict(
        features: MLMultiArray,
        attentionMask: MLMultiArray
    ) async throws -> MedASRCTCOutput {
        // Validate input shapes
        let melLength = features.shape[1].intValue
        guard melLength == Self.modelMelLength else {
            throw MedASRError.unsupportedMelLength(
                actual: melLength,
                expected: Self.modelMelLength
            )
        }

        // Run prediction off the main thread (COREML.md Section 8)
        let prediction = try await Task.detached(priority: .userInitiated) { [model] in
            let input = try MLDictionaryFeatureProvider(dictionary: [
                "input_features": MLFeatureValue(multiArray: features),
                "attention_mask": MLFeatureValue(multiArray: attentionMask),
            ])
            return try model.prediction(from: input)
        }.value

        // Extract logits
        guard let logitsFeature = prediction.featureValue(for: "logits"),
              let logitsArray = logitsFeature.multiArrayValue else {
            throw MedASRError.missingOutput("logits")
        }

        // Convert Float16 -> Float32 (COREML.md Section 8)
        let logits = multiArrayToFloats(logitsArray)

        // Determine output time steps
        // logitsArray shape is [1, T', vocabSize]
        let outputTimeSteps = logitsArray.shape[1].intValue
        let outputVocabSize = logitsArray.shape[2].intValue

        return MedASRCTCOutput(
            logits: logits,
            timeSteps: outputTimeSteps,
            vocabSize: outputVocabSize
        )
    }

    /// Convenience method: extract features and run inference in one call.
    ///
    /// - Parameters:
    ///   - audioSamples: Raw audio at 16kHz, mono, Float32
    ///   - extractor: A MelSpectrogramExtractor instance
    ///
    /// - Returns: MedASRCTCOutput containing CTC logits
    func transcribe(
        audioSamples: [Float],
        extractor: MelSpectrogramExtractor
    ) async throws -> MedASRCTCOutput {
        // Extract features, padded to the model's fixed input length
        let (features, mask) = extractor.extractFeatures(
            from: audioSamples,
            maxMelLength: Self.modelMelLength
        )

        return try await predict(features: features, attentionMask: mask)
    }

    // MARK: - Helpers

    /// Convert MLMultiArray to Float array, handling Float16 output.
    ///
    /// CoreML models with FLOAT16 precision return MLMultiArray with .float16 data type.
    /// We need to convert to Float for computation.
    private func multiArrayToFloats(_ array: MLMultiArray) -> [Float] {
        let count = array.count
        if array.dataType == .float16 {
            let src = UnsafePointer<Float16>(OpaquePointer(array.dataPointer))
            return (0..<count).map { Float(src[$0]) }
        } else if array.dataType == .float32 {
            var result = [Float](repeating: 0, count: count)
            memcpy(&result, array.dataPointer, count * MemoryLayout<Float>.size)
            return result
        } else {
            // Fallback: element-by-element access
            return (0..<count).map { array[$0].floatValue }
        }
    }
}

// MARK: - Output

/// Output from MedASR inference.
struct MedASRCTCOutput {
    /// Raw CTC logits, flat array of length `timeSteps * vocabSize`.
    /// Row-major: logits[t * vocabSize + v] = logit for time step t, token v.
    let logits: [Float]

    /// Number of output time steps (approximately input mel frames / 4).
    let timeSteps: Int

    /// Vocabulary size (512).
    let vocabSize: Int

    /// Get logits for a specific time step.
    func logits(atTimeStep t: Int) -> ArraySlice<Float> {
        let start = t * vocabSize
        return logits[start..<(start + vocabSize)]
    }
}

// MARK: - Errors

enum MedASRError: LocalizedError {
    case modelNotFound
    case unsupportedMelLength(actual: Int, expected: Int)
    case missingOutput(String)

    var errorDescription: String? {
        switch self {
        case .modelNotFound:
            return "MedASR.mlmodelc not found in app bundle. Ensure MedASR.mlpackage is included in your Xcode project."
        case .unsupportedMelLength(let actual, let expected):
            return "Mel spectrogram length \(actual) does not match expected \(expected). Pad the input to \(expected) frames and use the attention mask for variable-length audio."
        case .missingOutput(let name):
            return "CoreML model did not produce expected output '\(name)'. Verify the model was converted correctly."
        }
    }
}
