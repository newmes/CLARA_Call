import AVFoundation
import Combine
import WebRTC

@MainActor
final class DemoOrchestrator: ObservableObject {

    // MARK: - State Machine

    enum DemoStep {
        case idle
        case claraAsking
        case waitingForTap1
        case playingVideo1
        case consulting
        case claraResponding
        case waitingForTap2
        case playingVideo2
        case claraFollowUp
        case waitingForTap3
        case playingVideo3
        case done
    }

    @Published var step: DemoStep = .idle
    @Published var currentPlayer: AVPlayer?
    @Published var previewPlayer: AVPlayer?
    @Published var responseTimeMs: Int?
    @Published var audioLevel: CGFloat = 0

    var talkButtonEnabled: Bool {
        step == .waitingForTap1 || step == .waitingForTap2 || step == .waitingForTap3
    }

    var isListening: Bool {
        step == .playingVideo1 || step == .playingVideo2 || step == .playingVideo3
    }

    // MARK: - Dependencies

    var manager: WebRTCManager?
    var classifier: MedSigLIPClassifier?

    // MARK: - Own Instances

    private let careAIClient = CareAIClient()
    private var audioEngine: AVAudioEngine?
    private var playerNode: AVAudioPlayerNode?
    private var endObserver: Any?

    // MARK: - Public

    func start() {
        manager?.messages = []
        step = .idle
        configureAudioSession()

        // Show first video paused immediately so camera never flashes
        if let videoURL = Bundle.main.url(forResource: "p_reply_1", withExtension: "mp4", subdirectory: "demo_files") {
            let player = AVPlayer(url: videoURL)
            player.pause()
            previewPlayer = player
        }

        Task {
            try? await Task.sleep(nanoseconds: 1_500_000_000)
            guard step == .idle else { return }
            step = .claraAsking
            await playClaraQuestion()
        }
    }

    func handleTalkTap() {
        switch step {
        case .waitingForTap1:
            step = .playingVideo1
            previewPlayer = nil
            playVideo(named: "p_reply_1") {
                Task { @MainActor [weak self] in
                    await self?.processReply1()
                }
            }
        case .waitingForTap2:
            step = .playingVideo2
            playVideo(named: "p_reply_2") { [weak self] in
                Task { @MainActor in
                    await self?.playClaraFollowUp()
                }
            }
        case .waitingForTap3:
            step = .playingVideo3
            playVideo(named: "p_reply_3") { [weak self] in
                Task { @MainActor in
                    self?.step = .done
                }
            }
        default:
            break
        }
    }

    func cleanup() {
        playerNode?.stop()
        audioEngine?.stop()
        audioEngine = nil
        playerNode = nil
        if let obs = endObserver {
            NotificationCenter.default.removeObserver(obs)
            endObserver = nil
        }
        currentPlayer?.pause()
        currentPlayer = nil
        previewPlayer?.pause()
        previewPlayer = nil

        let session = RTCAudioSession.sharedInstance()
        session.lockForConfiguration()
        try? session.setActive(false)
        session.unlockForConfiguration()
    }

    // MARK: - Audio Session

    private func configureAudioSession() {
        let session = RTCAudioSession.sharedInstance()
        session.lockForConfiguration()
        do {
            try session.setCategory(AVAudioSession.Category.playback, with: [])
            try session.setMode(AVAudioSession.Mode.default)
            try session.setActive(true)
        } catch {
            print("[Demo] audio session error: \(error)")
        }
        session.unlockForConfiguration()
    }

    // MARK: - Clara Question Audio

    private func playClaraQuestion() async {
        guard let url = Bundle.main.url(forResource: "clara_question", withExtension: "wav", subdirectory: "demo_files") else {
            print("[Demo] clara_question.wav not found in bundle")
            step = .waitingForTap1
            return
        }

        manager?.messages.append(
            ChatMessage(text: "Have you noticed any new skin changes or rashes since yesterday? Could you show me?", isFromServer: true)
        )

        await playAudioFile(url: url)
        step = .waitingForTap1
    }

    // MARK: - Clara Follow-Up Audio

    private func playClaraFollowUp() async {
        step = .claraFollowUp

        manager?.messages.append(
            ChatMessage(text: "I will flag this for review by the medical team.", isFromServer: true)
        )

        guard let url = Bundle.main.url(forResource: "clara_answer", withExtension: "wav", subdirectory: "demo_files") else {
            print("[Demo] clara_answer.wav not found in bundle")
            step = .waitingForTap3
            return
        }

        await playAudioFile(url: url)
        step = .waitingForTap3
    }

    // MARK: - Video Playback

    private func playVideo(named name: String, then completion: @escaping () -> Void) {
        guard let url = Bundle.main.url(forResource: name, withExtension: "mp4", subdirectory: "demo_files") else {
            print("[Demo] \(name).mp4 not found in bundle")
            completion()
            return
        }

        let player = AVPlayer(url: url)
        currentPlayer = player

        endObserver = NotificationCenter.default.addObserver(
            forName: .AVPlayerItemDidPlayToEndTime,
            object: player.currentItem,
            queue: .main
        ) { [weak self] _ in
            guard let self else { return }
            Task { @MainActor in
                if let obs = self.endObserver {
                    NotificationCenter.default.removeObserver(obs)
                    self.endObserver = nil
                }
                completion()
            }
        }

        player.play()
    }

    // MARK: - Process Reply 1 (Audio extraction + frame + server call)

    private func processReply1() async {
        step = .consulting

        guard let manager else {
            step = .done
            return
        }

        let videoURL = Bundle.main.url(forResource: "p_reply_1", withExtension: "mp4", subdirectory: "demo_files")!
        let asset = AVURLAsset(url: videoURL)

        // Extract audio + frame in parallel
        async let audioB64Task = extractAudioBase64(from: asset)
        async let embeddingTask = extractFrameEmbedding(from: asset)

        let audioB64 = await audioB64Task
        let embedding = await embeddingTask

        guard let audioB64, let embedding else {
            print("[Demo] failed to extract audio or embedding")
            step = .done
            return
        }

        // POST to server
        do {
            let t0 = CFAbsoluteTimeGetCurrent()
            let response = try await careAIClient.consult(
                embedding: embedding,
                patientText: "",
                baseURL: manager.careAIBaseURL,
                audioBase64: audioB64
            )
            let elapsed = CFAbsoluteTimeGetCurrent() - t0
            responseTimeMs = Int(elapsed * 1000)

            let serverAudio = response.audio_base64.flatMap { Data(base64Encoded: $0) }
            manager.messages.append(
                ChatMessage(text: response.nurse_text, isFromServer: true, audioData: serverAudio)
            )

            if let serverAudio {
                step = .claraResponding
                await playAudioData(serverAudio)
            }
            step = .waitingForTap2
        } catch {
            print("[Demo] consult error: \(error)")
            step = .waitingForTap2
        }
    }

    // MARK: - Audio Extraction from MP4

    private func extractAudioBase64(from asset: AVURLAsset) async -> String? {
        await Task.detached {
            do {
                let audioTracks = try await asset.loadTracks(withMediaType: .audio)
                guard let audioTrack = audioTracks.first else {
                    print("[Demo] no audio track in asset")
                    return nil
                }

                let reader = try AVAssetReader(asset: asset)
                let outputSettings: [String: Any] = [
                    AVFormatIDKey: kAudioFormatLinearPCM,
                    AVSampleRateKey: 24000,
                    AVNumberOfChannelsKey: 1,
                    AVLinearPCMBitDepthKey: 16,
                    AVLinearPCMIsFloatKey: false,
                    AVLinearPCMIsBigEndianKey: false,
                ]
                let output = AVAssetReaderTrackOutput(track: audioTrack, outputSettings: outputSettings)
                reader.add(output)
                reader.startReading()

                var pcmData = Data()
                while let buffer = output.copyNextSampleBuffer(),
                      let blockBuffer = CMSampleBufferGetDataBuffer(buffer) {
                    var length = 0
                    var dataPointer: UnsafeMutablePointer<Int8>?
                    CMBlockBufferGetDataPointer(blockBuffer, atOffset: 0, lengthAtOffsetOut: nil, totalLengthOut: &length, dataPointerOut: &dataPointer)
                    if let dataPointer, length > 0 {
                        pcmData.append(UnsafeBufferPointer(start: dataPointer, count: length))
                    }
                }

                let wavData = Self.buildWAV(pcmData: pcmData, sampleRate: 24000, channels: 1, bitsPerSample: 16)
                return wavData.base64EncodedString()
            } catch {
                print("[Demo] audio extraction error: \(error)")
                return nil
            }
        }.value
    }

    nonisolated private static func buildWAV(pcmData: Data, sampleRate: Int, channels: Int, bitsPerSample: Int) -> Data {
        let dataLength = pcmData.count
        let byteRate = sampleRate * channels * (bitsPerSample / 8)
        let blockAlign = channels * (bitsPerSample / 8)

        var header = Data()
        header.append(contentsOf: "RIFF".utf8)
        header.append(contentsOf: UInt32(36 + dataLength).littleEndianBytes)
        header.append(contentsOf: "WAVE".utf8)
        header.append(contentsOf: "fmt ".utf8)
        header.append(contentsOf: UInt32(16).littleEndianBytes)                  // chunk size
        header.append(contentsOf: UInt16(1).littleEndianBytes)                   // PCM
        header.append(contentsOf: UInt16(channels).littleEndianBytes)
        header.append(contentsOf: UInt32(sampleRate).littleEndianBytes)
        header.append(contentsOf: UInt32(byteRate).littleEndianBytes)
        header.append(contentsOf: UInt16(blockAlign).littleEndianBytes)
        header.append(contentsOf: UInt16(bitsPerSample).littleEndianBytes)
        header.append(contentsOf: "data".utf8)
        header.append(contentsOf: UInt32(dataLength).littleEndianBytes)
        header.append(pcmData)
        return header
    }

    // MARK: - Frame Embedding

    private func extractFrameEmbedding(from asset: AVURLAsset) async -> [Float]? {
        guard let classifier else { return nil }

        do {
            let duration = try await asset.load(.duration)
            let midpoint = CMTime(seconds: duration.seconds * 0.5, preferredTimescale: 600)

            let generator = AVAssetImageGenerator(asset: asset)
            generator.appliesPreferredTrackTransform = true
            generator.requestedTimeToleranceBefore = .zero
            generator.requestedTimeToleranceAfter = .zero

            let (cgImage, _) = try await generator.image(at: midpoint)
            return try await classifier.imageEmbedding(for: cgImage)
        } catch {
            print("[Demo] frame embedding error: \(error)")
            return nil
        }
    }

    // MARK: - Audio Playback with Completion

    private func playAudioFile(url: URL) async {
        do {
            let data = try Data(contentsOf: url)
            await playAudioData(data)
        } catch {
            print("[Demo] failed to read audio file: \(error)")
        }
    }

    private func playAudioData(_ data: Data) async {
        // Stop previous playback
        playerNode?.stop()
        audioEngine?.stop()
        audioEngine = nil
        playerNode = nil

        do {
            let tempURL = FileManager.default.temporaryDirectory.appendingPathComponent("demo_playback.wav")
            try data.write(to: tempURL)
            let file = try AVAudioFile(forReading: tempURL)
            guard let buffer = AVAudioPCMBuffer(
                pcmFormat: file.processingFormat,
                frameCapacity: AVAudioFrameCount(file.length)
            ) else { return }
            try file.read(into: buffer)

            let engine = AVAudioEngine()
            let player = AVAudioPlayerNode()
            engine.attach(player)
            engine.connect(player, to: engine.mainMixerNode, format: buffer.format)
            try engine.start()

            audioEngine = engine
            playerNode = player

            // Install tap to monitor audio levels
            let mixer = engine.mainMixerNode
            let tapFormat = mixer.outputFormat(forBus: 0)
            mixer.installTap(onBus: 0, bufferSize: 1024, format: tapFormat) { [weak self] buffer, _ in
                guard let channelData = buffer.floatChannelData?[0] else { return }
                let frames = Int(buffer.frameLength)
                var sum: Float = 0
                for i in 0..<frames { sum += channelData[i] * channelData[i] }
                let rms = sqrt(sum / Float(max(frames, 1)))
                let level = CGFloat(min(rms * 4, 1.0))
                Task { @MainActor [weak self] in
                    self?.audioLevel = level
                }
            }

            await withCheckedContinuation { (continuation: CheckedContinuation<Void, Never>) in
                player.scheduleBuffer(buffer, completionCallbackType: .dataPlayedBack) { _ in
                    continuation.resume()
                }
                player.play()
            }

            mixer.removeTap(onBus: 0)
            audioLevel = 0
            player.stop()
            engine.stop()
        } catch {
            print("[Demo] audio playback error: \(error)")
        }
    }
}

// MARK: - Little-Endian Helpers

extension FixedWidthInteger {
    var littleEndianBytes: [UInt8] {
        withUnsafeBytes(of: littleEndian) { Array($0) }
    }
}
