import AVFoundation
import Foundation
import UIKit
import WebRTC

@MainActor
class WebRTCManager: ObservableObject {
    @Published var messages: [ChatMessage] = []
    @Published var localVideoTrack: RTCVideoTrack?
    @Published var isStreaming = false

    var classifier: MedSigLIPClassifier?
    var careAIBaseURL = "https://imperceptible-makena-overabusively.ngrok-free.dev"

    func setClassifier(_ classifier: MedSigLIPClassifier) {
        self.classifier = classifier
    }

    private var frameGrabber: VideoFrameGrabber?
    private var embeddingTimer: Task<Void, Never>?
    private static let privacyFrameInterval: TimeInterval = 5.0
    private var latestEmbedding: [Float]?
    private let careAIClient = CareAIClient()
    private var playbackEngine: AVAudioEngine?
    private var playerNode: AVAudioPlayerNode?

    private static var factory: RTCPeerConnectionFactory = {
        RTCInitializeSSL()
        let videoEncoderFactory = RTCDefaultVideoEncoderFactory()
        let videoDecoderFactory = RTCDefaultVideoDecoderFactory()
        return RTCPeerConnectionFactory(
            encoderFactory: videoEncoderFactory,
            decoderFactory: videoDecoderFactory
        )
    }()

    private var videoSource: RTCVideoSource?
    private var videoCapturer: RTCCameraVideoCapturer?

    // MARK: - Mock Data

    #if DEBUG
    private static let mockMessages: [ChatMessage] = [
        ChatMessage(text: "I've been having this rash on my arm for about three days now. It's red and a bit itchy.", isFromServer: false),
        ChatMessage(text: "I can see the area you're describing. It appears to be an erythematous, mildly raised rash on your left forearm. Can you tell me — has it been spreading, or has it stayed roughly the same size since you first noticed it?", isFromServer: true),
        ChatMessage(text: "It's gotten a little bigger. I also feel kind of tired lately.", isFromServer: false),
        ChatMessage(text: "Thank you for letting me know. The gradual spread along with fatigue could suggest a few things. Have you started any new medications, changed detergents or soaps recently, or been in contact with any plants or animals that are new to you?", isFromServer: true),
        ChatMessage(text: "Now that you mention it, I started a new antibiotic last week for a sinus infection.", isFromServer: false),
        ChatMessage(text: "That's very helpful context. Drug-related skin reactions can appear days after starting a new medication, which fits your timeline. I'd recommend you contact your prescribing physician to discuss the rash — they may want to switch your antibiotic. In the meantime, avoid scratching the area and apply a cool compress if the itching is bothersome.", isFromServer: true),
    ]
    #else
    private static let mockMessages: [ChatMessage] = []
    #endif

    // MARK: - Preview

    func startPreview() {
        guard videoCapturer == nil else { return }
        let source = Self.factory.videoSource()
        videoSource = source
        let capturer = RTCCameraVideoCapturer(delegate: source)
        videoCapturer = capturer
        let track = Self.factory.videoTrack(with: source, trackId: "video0")
        localVideoTrack = track
        startCameraCapture()
    }

    func stopPreview() {
        videoCapturer?.stopCapture()
        videoCapturer = nil
        videoSource = nil
        localVideoTrack = nil
    }

    // MARK: - Streaming

    func startStreaming() {
        if videoCapturer == nil { startPreview() }

        isStreaming = true
        configureAudioSession()

        // Attach frame grabber to local video track
        let grabber = VideoFrameGrabber()
        frameGrabber = grabber
        localVideoTrack?.add(grabber)

        // Start embedding loop (compute only, no network)
        embeddingTimer = Task { [weak self] in
            while !Task.isCancelled {
                try? await Task.sleep(nanoseconds: UInt64(Self.privacyFrameInterval * 1_000_000_000))
                guard !Task.isCancelled else { break }
                await self?.computeEmbedding()
            }
        }
    }

    private func computeEmbedding() async {
        guard let grabber = frameGrabber,
              let cgImage = grabber.grabLatestCGImage(),
              let classifier else { return }

        do {
            let embedding = try await classifier.imageEmbedding(for: cgImage)
            latestEmbedding = embedding
            print("[WebRTCManager] computed embedding (\(embedding.count) floats)")
        } catch {
            print("[WebRTCManager] embedding error: \(error)")
        }
    }

    // MARK: - Care AI

    func consultCareAI(patientText: String) {
        guard let embedding = latestEmbedding,
              !patientText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return }

        Task {
            do {
                let response = try await careAIClient.consult(
                    embedding: embedding, patientText: patientText, baseURL: careAIBaseURL
                )
                let hasB64 = response.audio_base64 != nil
                let serverAudio = response.audio_base64.flatMap { Data(base64Encoded: $0) }
                print("[CareAI] audio_base64 present: \(hasB64), decoded bytes: \(serverAudio?.count ?? 0)")
                messages.append(ChatMessage(text: response.nurse_text, isFromServer: true, audioData: serverAudio))
                if let serverAudio {
                    self.playAudio(data: serverAudio)
                }
                if let observations = response.visual_assessment?.general_observations {
                    print("[CareAI] visual: \(observations.joined(separator: ", "))")
                }
                if let concerns = response.nurse_structured?.preliminary_concerns {
                    print("[CareAI] concerns: \(concerns.joined(separator: ", "))")
                }
                if let totalMs = response.latency_ms?.total_ms {
                    print("[CareAI] latency: \(totalMs)ms")
                }
            } catch {
                print("[CareAI] consult error: \(error)")
            }
        }
    }

    // MARK: - TTS Playback

    func playAudio(data: Data) {
        // Stop any previous playback
        playerNode?.stop()
        playbackEngine?.stop()
        playbackEngine = nil
        playerNode = nil

        // Re-apply speaker override
        let session = RTCAudioSession.sharedInstance()
        session.lockForConfiguration()
        try? session.overrideOutputAudioPort(.speaker)
        session.unlockForConfiguration()

        do {
            let tempURL = FileManager.default.temporaryDirectory.appendingPathComponent("playback.wav")
            try data.write(to: tempURL)
            let file = try AVAudioFile(forReading: tempURL)
            guard let buffer = AVAudioPCMBuffer(
                pcmFormat: file.processingFormat,
                frameCapacity: AVAudioFrameCount(file.length)
            ) else {
                print("[WebRTCManager] failed to create PCM buffer")
                return
            }
            try file.read(into: buffer)

            let engine = AVAudioEngine()
            let player = AVAudioPlayerNode()
            engine.attach(player)
            engine.connect(player, to: engine.mainMixerNode, format: buffer.format)

            try engine.start()
            player.scheduleBuffer(buffer)
            player.play()

            playbackEngine = engine
            playerNode = player
            print("[WebRTCManager] playing via AVAudioEngine, duration: \(Double(buffer.frameLength) / buffer.format.sampleRate)s")
        } catch {
            print("[WebRTCManager] playback error: \(error)")
        }
    }

    // MARK: - Audio Session

    private func configureAudioSession() {
        let audioSession = RTCAudioSession.sharedInstance()
        audioSession.lockForConfiguration()
        do {
            try audioSession.setCategory(
                AVAudioSession.Category.playAndRecord,
                with: [.defaultToSpeaker]
            )
            try audioSession.setMode(AVAudioSession.Mode.default)
            try audioSession.overrideOutputAudioPort(.speaker)
            try audioSession.setActive(true)
        } catch {
            print("[WebRTCManager] audio session error: \(error)")
        }
        audioSession.unlockForConfiguration()
    }

    // MARK: - Camera

    private func startCameraCapture() {
        guard let capturer = videoCapturer else { return }

        let devices = RTCCameraVideoCapturer.captureDevices()
        guard let frontCamera = devices.first(where: { $0.position == .front }) ?? devices.first else {
            print("[WebRTCManager] no camera found")
            return
        }

        let formats = RTCCameraVideoCapturer.supportedFormats(for: frontCamera)
        let targetWidth: Int32 = 640
        let targetHeight: Int32 = 480

        let selectedFormat = formats
            .filter { format in
                let desc = CMVideoFormatDescriptionGetDimensions(format.formatDescription)
                return desc.width >= targetWidth && desc.height >= targetHeight
            }
            .sorted { a, b in
                let da = CMVideoFormatDescriptionGetDimensions(a.formatDescription)
                let db = CMVideoFormatDescriptionGetDimensions(b.formatDescription)
                return (da.width * da.height) < (db.width * db.height)
            }
            .first ?? formats.last

        guard let format = selectedFormat else {
            print("[WebRTCManager] no suitable format")
            return
        }

        capturer.startCapture(with: frontCamera, format: format, fps: 30)
    }

    // MARK: - Disconnect

    func disconnect() {
        embeddingTimer?.cancel()
        embeddingTimer = nil
        if let grabber = frameGrabber {
            localVideoTrack?.remove(grabber)
            frameGrabber = nil
        }

        stopPreview()
        isStreaming = false

        let audioSession = RTCAudioSession.sharedInstance()
        audioSession.lockForConfiguration()
        do {
            try audioSession.setActive(false)
        } catch {
            print("[WebRTCManager] deactivate audio session error: \(error)")
        }
        audioSession.unlockForConfiguration()
    }
}

// MARK: - Video Frame Grabber

private class VideoFrameGrabber: NSObject, RTCVideoRenderer {
    private var latestPixelBuffer: CVPixelBuffer?
    private let lock = NSLock()
    private let ciContext = CIContext()

    func setSize(_ size: CGSize) {}
    func renderFrame(_ frame: RTCVideoFrame?) {
        guard let frame,
              let rtcBuffer = frame.buffer as? RTCCVPixelBuffer else { return }
        lock.lock()
        latestPixelBuffer = rtcBuffer.pixelBuffer
        lock.unlock()
    }

    func grabLatestCGImage() -> CGImage? {
        lock.lock()
        let buffer = latestPixelBuffer
        lock.unlock()
        guard let buffer else { return nil }
        let ciImage = CIImage(cvPixelBuffer: buffer)
        return ciContext.createCGImage(ciImage, from: ciImage.extent)
    }
}
