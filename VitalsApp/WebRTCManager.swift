import Foundation
import UIKit
import WebRTC

@MainActor
class WebRTCManager: ObservableObject {
    @Published var connectionState: RTCIceConnectionState = .new
    @Published var isSignalingConnected = false
    @Published var messages: [ChatMessage] = []
    @Published var localVideoTrack: RTCVideoTrack?
    @Published var isStreaming = false
    @Published var isPrivacyMode = false

    var classifier: MedSigLIPClassifier?
    var debugFrameEnabled = false

    func setClassifier(_ classifier: MedSigLIPClassifier) {
        self.classifier = classifier
    }

    private var frameGrabber: VideoFrameGrabber?
    private var embeddingTimer: Task<Void, Never>?
    private static let privacyFrameInterval: TimeInterval = 5.0

    private static var factory: RTCPeerConnectionFactory = {
        RTCInitializeSSL()
        let videoEncoderFactory = RTCDefaultVideoEncoderFactory()
        let videoDecoderFactory = RTCDefaultVideoDecoderFactory()
        return RTCPeerConnectionFactory(
            encoderFactory: videoEncoderFactory,
            decoderFactory: videoDecoderFactory
        )
    }()

    private var peerConnection: RTCPeerConnection?
    private var videoSource: RTCVideoSource?
    private var videoCapturer: RTCCameraVideoCapturer?
    private var localAudioTrack: RTCAudioTrack?
    fileprivate var signalingClient = SignalingClient()
    private var pendingICECandidates: [RTCIceCandidate] = []
    private var hasRemoteDescription = false
    private var signalingContinuation: CheckedContinuation<Void, Never>?
    private let serverURL: URL

    init(serverURL: URL = URL(string: "wss://imperceptible-makena-overabusively.ngrok-free.dev/ws")!) {
        self.serverURL = serverURL
    }

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

    // MARK: - Connect

    func connectSignaling() {
        setupSignaling()
        signalingClient.connect(to: serverURL)
    }

    func startStreaming() async {
        if videoCapturer == nil { startPreview() }

        // Wait for signaling if not yet connected
        if !isSignalingConnected {
            await withCheckedContinuation { continuation in
                signalingContinuation = continuation
            }
        }

        isStreaming = true
        configureAudioSession()
        createPeerConnection()
        addMediaTracks()
        await createAndSendOffer()
    }

    func sendSignalingMessage(_ message: SignalingMessage) {
        signalingClient.send(message)
    }

    func startPrivacyStreaming() async {
        if videoCapturer == nil { startPreview() }

        // Wait for signaling if not yet connected
        if !isSignalingConnected {
            await withCheckedContinuation { continuation in
                signalingContinuation = continuation
            }
        }

        isStreaming = true
        isPrivacyMode = true
        signalingClient.send(.privacyStart)

        // Attach frame grabber to local video track
        let grabber = VideoFrameGrabber()
        frameGrabber = grabber
        localVideoTrack?.add(grabber)

        // Start embedding loop
        embeddingTimer = Task { [weak self] in
            while !Task.isCancelled {
                try? await Task.sleep(nanoseconds: UInt64(Self.privacyFrameInterval * 1_000_000_000))
                guard !Task.isCancelled else { break }
                await self?.captureAndSendEmbedding()
            }
        }
    }

    private func captureAndSendEmbedding() async {
        guard let grabber = frameGrabber,
              let cgImage = grabber.grabLatestCGImage(),
              let classifier else { return }

        do {
            let embedding = try await classifier.imageEmbedding(for: cgImage)
            let timestamp = Date().timeIntervalSince1970

            if debugFrameEnabled,
               let jpegData = UIImage(cgImage: cgImage).jpegData(compressionQuality: 0.8) {
                signalingClient.send(.debugFrame(imageData: jpegData, embeddingValues: embedding, timestamp: timestamp))
                print("[WebRTCManager] sent debug frame + embedding (\(embedding.count) floats, \(jpegData.count) bytes)")
            } else {
                signalingClient.send(.embedding(values: embedding, timestamp: timestamp))
                print("[WebRTCManager] sent embedding (\(embedding.count) floats)")
            }
        } catch {
            print("[WebRTCManager] embedding error: \(error)")
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

    // MARK: - Peer Connection

    private func createPeerConnection() {
        let config = RTCConfiguration()
        config.iceServers = [RTCIceServer(urlStrings: ["stun:stun.l.google.com:19302"])]
        config.sdpSemantics = .unifiedPlan

        let constraints = RTCMediaConstraints(
            mandatoryConstraints: nil,
            optionalConstraints: ["DtlsSrtpKeyAgreement": "true"]
        )

        let pc = Self.factory.peerConnection(with: config, constraints: constraints, delegate: nil)
        peerConnection = pc

        let delegateAdapter = PeerConnectionDelegateAdapter(manager: self)
        pc?.delegate = delegateAdapter
        objc_setAssociatedObject(pc as Any, "delegateAdapter", delegateAdapter, .OBJC_ASSOCIATION_RETAIN)
    }

    // MARK: - Media Tracks

    private func addMediaTracks() {
        guard let pc = peerConnection else { return }

        // Audio
        let audioConstraints = RTCMediaConstraints(mandatoryConstraints: nil, optionalConstraints: nil)
        let audioSource = Self.factory.audioSource(with: audioConstraints)
        let audioTrack = Self.factory.audioTrack(with: audioSource, trackId: "audio0")
        pc.add(audioTrack, streamIds: ["stream0"])
        localAudioTrack = audioTrack

        // Video (reuse track from preview)
        if let videoTrack = localVideoTrack {
            pc.add(videoTrack, streamIds: ["stream0"])
        }
    }

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

    // MARK: - Signaling

    private func setupSignaling() {
        signalingClient.onConnect = { [weak self] in
            Task { @MainActor in
                self?.isSignalingConnected = true
                self?.signalingContinuation?.resume()
                self?.signalingContinuation = nil
            }
        }

        signalingClient.onMessage = { [weak self] message in
            Task { @MainActor in
                self?.handleSignalingMessage(message)
            }
        }

        signalingClient.onDisconnect = { error in
            Task { @MainActor in
                print("[WebRTCManager] signaling disconnected: \(String(describing: error))")
            }
        }
    }

    private func createAndSendOffer() async {
        guard let pc = peerConnection else { return }

        let constraints = RTCMediaConstraints(
            mandatoryConstraints: [
                "OfferToReceiveAudio": "true",
                "OfferToReceiveVideo": "false"
            ],
            optionalConstraints: nil
        )

        do {
            let sdp = try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<RTCSessionDescription, Error>) in
                pc.offer(for: constraints) { sdp, error in
                    if let error { continuation.resume(throwing: error) }
                    else if let sdp { continuation.resume(returning: sdp) }
                    else { continuation.resume(throwing: WebRTCError.noDescription) }
                }
            }

            try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
                pc.setLocalDescription(sdp) { error in
                    if let error { continuation.resume(throwing: error) }
                    else { continuation.resume() }
                }
            }

            signalingClient.send(.offer(sdp: sdp.sdp))
            print("[WebRTCManager] offer sent")
        } catch {
            print("[WebRTCManager] offer error: \(error)")
        }
    }

    private func handleSignalingMessage(_ message: SignalingMessage) {
        switch message {
        case .answer(let sdp):
            handleAnswer(sdp: sdp)
        case .candidate(let sdpMid, let sdpMLineIndex, let candidate):
            handleCandidate(sdpMid: sdpMid, sdpMLineIndex: sdpMLineIndex, candidate: candidate)
        case .transcription(let text, let timestamp):
            let chatMessage = ChatMessage(
                text: text,
                timestamp: Date(timeIntervalSince1970: timestamp),
                isFromServer: true
            )
            messages.append(chatMessage)
        default:
            break
        }
    }

    private func handleAnswer(sdp: String) {
        guard let pc = peerConnection else { return }
        let sessionDescription = RTCSessionDescription(type: .answer, sdp: sdp)

        Task {
            do {
                try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
                    pc.setRemoteDescription(sessionDescription) { error in
                        if let error { continuation.resume(throwing: error) }
                        else { continuation.resume() }
                    }
                }

                hasRemoteDescription = true
                for candidate in pendingICECandidates {
                    pc.add(candidate) { error in
                        if let error { print("[WebRTCManager] add buffered candidate error: \(error)") }
                    }
                }
                pendingICECandidates.removeAll()
                print("[WebRTCManager] remote description set, drained \(pendingICECandidates.count) candidates")
            } catch {
                print("[WebRTCManager] set remote description error: \(error)")
            }
        }
    }

    private func handleCandidate(sdpMid: String?, sdpMLineIndex: Int32, candidate: String) {
        let iceCandidate = RTCIceCandidate(sdp: candidate, sdpMLineIndex: sdpMLineIndex, sdpMid: sdpMid)
        if hasRemoteDescription {
            peerConnection?.add(iceCandidate) { error in
                if let error { print("[WebRTCManager] add candidate error: \(error)") }
            }
        } else {
            pendingICECandidates.append(iceCandidate)
        }
    }

    // MARK: - Disconnect

    func disconnect() {
        // Clean up privacy mode
        embeddingTimer?.cancel()
        embeddingTimer = nil
        if let grabber = frameGrabber {
            localVideoTrack?.remove(grabber)
            frameGrabber = nil
        }
        isPrivacyMode = false

        stopPreview()

        localAudioTrack?.isEnabled = false
        localAudioTrack = nil

        peerConnection?.close()
        peerConnection = nil

        signalingClient.disconnect()
        isSignalingConnected = false
        hasRemoteDescription = false
        pendingICECandidates.removeAll()
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

// MARK: - Peer Connection Delegate

private class PeerConnectionDelegateAdapter: NSObject, RTCPeerConnectionDelegate {
    weak var manager: WebRTCManager?

    init(manager: WebRTCManager) {
        self.manager = manager
    }

    func peerConnection(_ peerConnection: RTCPeerConnection, didChange stateChanged: RTCSignalingState) {
        print("[WebRTC] signaling state: \(stateChanged.rawValue)")
    }

    func peerConnection(_ peerConnection: RTCPeerConnection, didAdd stream: RTCMediaStream) {
        print("[WebRTC] added stream: \(stream.streamId)")
    }

    func peerConnection(_ peerConnection: RTCPeerConnection, didRemove stream: RTCMediaStream) {
        print("[WebRTC] removed stream: \(stream.streamId)")
    }

    func peerConnectionShouldNegotiate(_ peerConnection: RTCPeerConnection) {
        print("[WebRTC] should negotiate")
    }

    func peerConnection(_ peerConnection: RTCPeerConnection, didChange newState: RTCIceConnectionState) {
        print("[WebRTC] ICE connection state: \(newState.rawValue)")
        Task { @MainActor [weak self] in
            self?.manager?.connectionState = newState
        }
    }

    func peerConnection(_ peerConnection: RTCPeerConnection, didChange newState: RTCIceGatheringState) {
        print("[WebRTC] ICE gathering state: \(newState.rawValue)")
    }

    func peerConnection(_ peerConnection: RTCPeerConnection, didGenerate candidate: RTCIceCandidate) {
        Task { @MainActor [weak self] in
            self?.manager?.signalingClient.send(
                .candidate(sdpMid: candidate.sdpMid, sdpMLineIndex: candidate.sdpMLineIndex, candidate: candidate.sdp)
            )
        }
    }

    func peerConnection(_ peerConnection: RTCPeerConnection, didRemove candidates: [RTCIceCandidate]) {
        print("[WebRTC] removed \(candidates.count) candidates")
    }

    func peerConnection(_ peerConnection: RTCPeerConnection, didOpen dataChannel: RTCDataChannel) {
        print("[WebRTC] data channel opened: \(dataChannel.label)")
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

// MARK: - Errors

enum WebRTCError: Error {
    case noDescription
}
