import Foundation

/// Greedy CTC decoder for MedASR output.
///
/// CTC decoding algorithm:
/// 1. Argmax over vocabulary dimension at each time step
/// 2. Collapse consecutive repeated tokens
/// 3. Remove blank tokens (epsilon, id=0)
/// 4. Decode remaining token IDs to text via SentencePiece tokenizer
///
/// Token mapping:
/// - Token 0: `<epsilon>` (CTC blank / pad token)
/// - Token 1: `<unk>` (unknown)
/// - Token 2: `</s>` (end of sentence)
/// - Tokens 3-511: SentencePiece vocabulary
final class CTCDecoder {

    // MARK: - Configuration

    /// CTC blank token ID (also the pad token)
    static let blankTokenID: Int = 0

    /// Unknown token ID
    static let unkTokenID: Int = 1

    /// End-of-sentence token ID
    static let eosTokenID: Int = 2

    // MARK: - Properties

    /// Vocabulary mapping from token ID to string piece.
    /// Loaded from tokenizer.json.
    private let vocabulary: [Int: String]

    /// SentencePiece word boundary marker
    /// SentencePiece uses the Unicode character U+2581 (lower one eighth block)
    /// as a word boundary / space marker at the beginning of tokens.
    private let sentencePieceSpaceMarker: Character = "\u{2581}"

    // MARK: - Initialization

    /// Initialize the CTC decoder with a vocabulary loaded from the tokenizer.
    ///
    /// - Parameter tokenizerURL: URL to the `tokenizer.json` file bundled in the app.
    ///                           This should be the file from the HuggingFace model repo.
    init(tokenizerURL: URL) throws {
        self.vocabulary = try Self.loadVocabulary(from: tokenizerURL)
    }

    /// Initialize with a pre-built vocabulary dictionary.
    ///
    /// - Parameter vocabulary: Mapping from token ID to string piece.
    init(vocabulary: [Int: String]) {
        self.vocabulary = vocabulary
    }

    // MARK: - Greedy Decoding

    /// Perform greedy CTC decoding on model output logits.
    ///
    /// - Parameter output: MedASRCTCOutput from the inference engine.
    /// - Returns: Decoded transcription text.
    func decode(output: MedASRCTCOutput) -> String {
        let tokenIDs = greedyDecode(output: output)
        return detokenize(tokenIDs: tokenIDs)
    }

    /// Greedy decode: argmax at each time step, collapse repeats, remove blanks.
    ///
    /// - Parameter output: MedASRCTCOutput containing CTC logits.
    /// - Returns: Array of non-blank, non-repeated token IDs.
    func greedyDecode(output: MedASRCTCOutput) -> [Int] {
        let timeSteps = output.timeSteps
        let vocabSize = output.vocabSize
        let logits = output.logits

        var tokenIDs = [Int]()
        var previousToken: Int = -1

        for t in 0..<timeSteps {
            // Argmax over vocabulary dimension
            let offset = t * vocabSize
            var maxVal: Float = -Float.infinity
            var maxIdx: Int = 0

            for v in 0..<vocabSize {
                let val = logits[offset + v]
                if val > maxVal {
                    maxVal = val
                    maxIdx = v
                }
            }

            // CTC: skip blank tokens and consecutive repeats
            if maxIdx != Self.blankTokenID && maxIdx != previousToken {
                tokenIDs.append(maxIdx)
            }

            previousToken = maxIdx
        }

        // Remove trailing EOS if present
        if let last = tokenIDs.last, last == Self.eosTokenID {
            tokenIDs.removeLast()
        }

        return tokenIDs
    }

    /// Perform greedy CTC decoding directly on raw logits.
    ///
    /// - Parameters:
    ///   - logits: Flat array of logits, length `timeSteps * vocabSize`.
    ///   - timeSteps: Number of time steps.
    ///   - vocabSize: Vocabulary size (default 512).
    /// - Returns: Array of non-blank, non-repeated token IDs.
    static func greedyDecode(
        logits: [Float],
        timeSteps: Int,
        vocabSize: Int = 512
    ) -> [Int] {
        var tokenIDs = [Int]()
        var previousToken: Int = -1

        for t in 0..<timeSteps {
            let offset = t * vocabSize
            var maxVal: Float = -Float.infinity
            var maxIdx: Int = 0

            for v in 0..<vocabSize {
                let val = logits[offset + v]
                if val > maxVal {
                    maxVal = val
                    maxIdx = v
                }
            }

            if maxIdx != blankTokenID && maxIdx != previousToken {
                tokenIDs.append(maxIdx)
            }

            previousToken = maxIdx
        }

        if let last = tokenIDs.last, last == eosTokenID {
            tokenIDs.removeLast()
        }

        return tokenIDs
    }

    // MARK: - Detokenization

    /// Convert token IDs to text using the SentencePiece vocabulary.
    ///
    /// SentencePiece encodes word boundaries with the U+2581 character at the
    /// start of tokens that begin a new word. We replace these with spaces
    /// and trim leading/trailing whitespace.
    ///
    /// - Parameter tokenIDs: Array of token IDs from greedy decoding.
    /// - Returns: Decoded text string.
    func detokenize(tokenIDs: [Int]) -> String {
        var pieces = [String]()

        for id in tokenIDs {
            guard id != Self.blankTokenID,
                  id != Self.eosTokenID else {
                continue
            }

            if let piece = vocabulary[id] {
                pieces.append(piece)
            } else {
                // Unknown token - skip or replace
                pieces.append("<unk>")
            }
        }

        // Join pieces and replace SentencePiece space marker with actual space
        var text = pieces.joined()
        text = text.replacingOccurrences(
            of: String(sentencePieceSpaceMarker),
            with: " "
        )

        // Trim leading/trailing whitespace
        text = text.trimmingCharacters(in: .whitespaces)

        return text
    }

    // MARK: - Vocabulary Loading

    /// Load vocabulary from a tokenizer.json file.
    ///
    /// The tokenizer.json file from HuggingFace contains a "model" -> "vocab"
    /// section that maps token strings to IDs, or an "added_tokens" section.
    /// We parse this to build our ID -> string mapping.
    ///
    /// Note: If using swift-transformers, you can use AutoTokenizer instead.
    /// This manual loading is provided as a lightweight alternative that
    /// doesn't require the swift-transformers dependency.
    private static func loadVocabulary(from url: URL) throws -> [Int: String] {
        let data = try Data(contentsOf: url)
        guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw CTCDecoderError.invalidTokenizerFormat
        }

        var vocabulary = [Int: String]()

        // Try "model" -> "vocab" format (SentencePiece/Unigram)
        if let model = json["model"] as? [String: Any],
           let vocab = model["vocab"] as? [[Any]] {
            // Unigram format: vocab is array of [piece, score] pairs
            // Token IDs are implicit (index in array)
            for (index, entry) in vocab.enumerated() {
                if let piece = entry.first as? String {
                    vocabulary[index] = piece
                }
            }
        }

        // Try "model" -> "vocab" as dictionary format (BPE)
        if vocabulary.isEmpty,
           let model = json["model"] as? [String: Any],
           let vocab = model["vocab"] as? [String: Int] {
            for (piece, id) in vocab {
                vocabulary[id] = piece
            }
        }

        // Add special/added tokens (these override regular vocab)
        if let addedTokens = json["added_tokens"] as? [[String: Any]] {
            for token in addedTokens {
                if let id = token["id"] as? Int,
                   let content = token["content"] as? String {
                    vocabulary[id] = content
                }
            }
        }

        guard !vocabulary.isEmpty else {
            throw CTCDecoderError.emptyVocabulary
        }

        return vocabulary
    }
}

// MARK: - Errors

enum CTCDecoderError: LocalizedError {
    case invalidTokenizerFormat
    case emptyVocabulary
    case tokenizerNotFound

    var errorDescription: String? {
        switch self {
        case .invalidTokenizerFormat:
            return "tokenizer.json has an unrecognized format. Expected HuggingFace tokenizer JSON with 'model.vocab' or 'added_tokens'."
        case .emptyVocabulary:
            return "No vocabulary entries found in tokenizer.json. Verify the file is from the google/medasr model."
        case .tokenizerNotFound:
            return "tokenizer.json not found in app bundle. Ensure the tokenizer files from google/medasr are included as bundled resources."
        }
    }
}

// MARK: - Convenience Extensions

extension CTCDecoder {

    /// Load a CTCDecoder from bundled tokenizer files.
    ///
    /// Looks for `tokenizer.json` in the specified bundle directory.
    ///
    /// - Parameter bundleDirectory: Name of the resource folder containing tokenizer files.
    ///                              Default is "tokenizer".
    /// - Returns: A configured CTCDecoder.
    static func fromBundle(directory: String = "tokenizer") throws -> CTCDecoder {
        guard let dirURL = Bundle.main.url(forResource: directory, withExtension: nil) else {
            throw CTCDecoderError.tokenizerNotFound
        }

        let tokenizerURL = dirURL.appendingPathComponent("tokenizer.json")
        guard FileManager.default.fileExists(atPath: tokenizerURL.path) else {
            throw CTCDecoderError.tokenizerNotFound
        }

        return try CTCDecoder(tokenizerURL: tokenizerURL)
    }
}

// MARK: - Full Pipeline Example

extension CTCDecoder {

    /// Full transcription pipeline: audio -> mel -> inference -> decode.
    ///
    /// Example usage:
    /// ```swift
    /// let extractor = MelSpectrogramExtractor()
    /// let inference = try await MedASRInference.load()
    /// let decoder = try CTCDecoder.fromBundle()
    ///
    /// let audioSamples: [Float] = loadAudio(url: audioURL)
    /// let text = try await decoder.fullPipeline(
    ///     audioSamples: audioSamples,
    ///     extractor: extractor,
    ///     inference: inference
    /// )
    /// print("Transcription: \(text)")
    /// ```
    func fullPipeline(
        audioSamples: [Float],
        extractor: MelSpectrogramExtractor,
        inference: MedASRInference
    ) async throws -> String {
        let output = try await inference.transcribe(
            audioSamples: audioSamples,
            extractor: extractor
        )
        return decode(output: output)
    }
}
