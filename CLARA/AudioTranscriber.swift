import AVFoundation
import Foundation

@MainActor
final class AudioTranscriber: ObservableObject {

    // MARK: - State

    enum State {
        case idle
        case loadingModel
        case ready
        case transcribing
    }

    @Published private(set) var state: State = .idle

    var onTranscription: ((String, Data?) -> Void)?

    // MARK: - Audio

    private var audioEngine: AVAudioEngine?
    private var pcmBuffer: [Float] = []
    private let bufferLock = NSLock()
    private static let sampleRate: Double = 16_000
    private static let maxBufferSamples = 480_000 // 30s at 16kHz

    // MARK: - Model Components

    private var melExtractor: MelSpectrogramExtractor?
    private var inference: AnyObject? // MedASRInference, erased for iOS 17 compat
    private var decoder: CTCDecoder?

    // MARK: - Inference Loop

    private var inferenceTask: Task<Void, Never>?
    private var isInferring = false
    private var isRecording = false

    // MARK: - Load Model

    func loadModel() async {
        guard state == .idle else { return }

        state = .loadingModel
        do {
            melExtractor = MelSpectrogramExtractor()
            let model = try await MedASRInference.load()
            inference = model
            decoder = try CTCDecoder.fromBundle(directory: "medasr_tokenizer")
            state = .ready
            print("[AudioTranscriber] model loaded")
        } catch {
            print("[AudioTranscriber] failed to load model: \(error)")
            state = .idle
        }
    }

    // MARK: - Start / Stop

    func startTranscribing() {
        guard state == .ready else { return }

        state = .transcribing
        isRecording = true
        startAudioEngine()
        startInferenceLoop()
    }

    /// Start audio engine only (for push-to-talk). No inference loop.
    func startListening() {
        guard state == .ready else { return }

        state = .transcribing
        startAudioEngine()
    }

    func beginUtterance() {
        bufferLock.lock()
        pcmBuffer.removeAll()
        bufferLock.unlock()
        isRecording = true
    }

    func endUtteranceAndTranscribe() {
        isRecording = false
        Task { await runInference() }
    }

    func stopTranscribing() {
        inferenceTask?.cancel()
        inferenceTask = nil

        audioEngine?.stop()
        audioEngine?.inputNode.removeTap(onBus: 0)
        audioEngine = nil

        bufferLock.lock()
        pcmBuffer.removeAll()
        bufferLock.unlock()

        isInferring = false

        if state == .transcribing {
            state = .ready
        }
    }

    // MARK: - Audio Engine

    private func startAudioEngine() {
        let engine = AVAudioEngine()
        let inputNode = engine.inputNode
        let hwFormat = inputNode.outputFormat(forBus: 0)

        guard let targetFormat = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: Self.sampleRate,
            channels: 1,
            interleaved: false
        ) else {
            print("[AudioTranscriber] failed to create target format")
            return
        }

        guard let converter = AVAudioConverter(from: hwFormat, to: targetFormat) else {
            print("[AudioTranscriber] failed to create audio converter")
            return
        }

        inputNode.installTap(onBus: 0, bufferSize: 4096, format: hwFormat) { [weak self] buffer, _ in
            guard let self else { return }

            let frameCount = AVAudioFrameCount(
                Double(buffer.frameLength) * Self.sampleRate / buffer.format.sampleRate
            )
            guard frameCount > 0,
                  let convertedBuffer = AVAudioPCMBuffer(pcmFormat: targetFormat, frameCapacity: frameCount)
            else { return }

            var error: NSError?
            converter.convert(to: convertedBuffer, error: &error) { _, outStatus in
                outStatus.pointee = .haveData
                return buffer
            }

            if let err = error {
                print("[AudioTranscriber] conversion error: \(err)")
                return
            }

            guard self.isRecording,
                  let channelData = convertedBuffer.floatChannelData else { return }
            let samples = Array(UnsafeBufferPointer(
                start: channelData[0],
                count: Int(convertedBuffer.frameLength)
            ))

            self.bufferLock.lock()
            self.pcmBuffer.append(contentsOf: samples)
            if self.pcmBuffer.count > Self.maxBufferSamples {
                self.pcmBuffer.removeFirst(self.pcmBuffer.count - Self.maxBufferSamples)
            }
            self.bufferLock.unlock()
        }

        do {
            try engine.start()
            audioEngine = engine
            print("[AudioTranscriber] audio engine started")
        } catch {
            print("[AudioTranscriber] engine start error: \(error)")
        }
    }

    // MARK: - Inference Loop

    private func startInferenceLoop() {
        inferenceTask = Task { [weak self] in
            while !Task.isCancelled {
                try? await Task.sleep(nanoseconds: 5_000_000_000) // 5s
                guard !Task.isCancelled else { break }
                await self?.runInference()
            }
        }
    }

    private func runInference() async {
        guard !isInferring,
              let melExtractor,
              let inference = inference as? MedASRInference,
              let decoder
        else { return }

        // Snapshot the buffer
        bufferLock.lock()
        let samples = Array(pcmBuffer)
        bufferLock.unlock()

        // Need at least 2s of audio
        let minSamples = Int(Self.sampleRate * 2)
        guard samples.count >= minSamples else { return }

        isInferring = true
        defer { isInferring = false }

        do {
            let text = try await decoder.fullPipeline(
                audioSamples: samples,
                extractor: melExtractor,
                inference: inference
            )

            guard !text.isEmpty else { return }
            onTranscription?(text, wavData(from: samples))
        } catch {
            print("[AudioTranscriber] inference error: \(error)")
        }
    }

    // MARK: - WAV Encoding

    private func wavData(from samples: [Float]) -> Data {
        let numSamples = samples.count
        let bytesPerSample = 2
        let dataSize = numSamples * bytesPerSample
        let sampleRate = UInt32(Self.sampleRate)

        var data = Data(capacity: 44 + dataSize)

        // RIFF header
        data.append(contentsOf: [0x52, 0x49, 0x46, 0x46]) // "RIFF"
        appendUInt32(&data, UInt32(36 + dataSize))
        data.append(contentsOf: [0x57, 0x41, 0x56, 0x45]) // "WAVE"

        // fmt chunk
        data.append(contentsOf: [0x66, 0x6D, 0x74, 0x20]) // "fmt "
        appendUInt32(&data, 16)                             // chunk size
        appendUInt16(&data, 1)                              // PCM
        appendUInt16(&data, 1)                              // mono
        appendUInt32(&data, sampleRate)
        appendUInt32(&data, sampleRate * UInt32(bytesPerSample)) // byte rate
        appendUInt16(&data, UInt16(bytesPerSample))         // block align
        appendUInt16(&data, 16)                             // bits per sample

        // data chunk
        data.append(contentsOf: [0x64, 0x61, 0x74, 0x61]) // "data"
        appendUInt32(&data, UInt32(dataSize))

        for sample in samples {
            let clamped = max(-1.0, min(1.0, sample))
            let int16 = Int16(clamped * Float(Int16.max))
            appendInt16(&data, int16)
        }

        return data
    }

    private func appendUInt16(_ data: inout Data, _ value: UInt16) {
        var v = value.littleEndian
        data.append(Data(bytes: &v, count: 2))
    }

    private func appendUInt32(_ data: inout Data, _ value: UInt32) {
        var v = value.littleEndian
        data.append(Data(bytes: &v, count: 4))
    }

    private func appendInt16(_ data: inout Data, _ value: Int16) {
        var v = value.littleEndian
        data.append(Data(bytes: &v, count: 2))
    }
}
