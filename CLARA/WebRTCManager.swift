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
    private var audioPlayer: AVAudioPlayer?

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
                messages.append(ChatMessage(text: response.nurse_text, isFromServer: true))
                if let b64 = response.audio_base64, let audioData = Data(base64Encoded: b64) {
                    self.playAudio(data: audioData)
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

    private func playAudio(data: Data) {
        do {
            let player = try AVAudioPlayer(data: data)
            audioPlayer = player
            player.play()
        } catch {
            print("[WebRTCManager] audio playback error: \(error)")
        }
    }

    // MARK: - Audio Session

    private func configureAudioSession() {
        let audioSession = RTCAudioSession.sharedInstance()
        audioSession.lockForConfiguration()
        do {
            try audioSession.setCategory(AVAudioSession.Category.playAndRecord)
            try audioSession.setMode(AVAudioSession.Mode.videoChat)
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
