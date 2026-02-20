import Foundation
import WebRTC

@MainActor
class WebRTCManager: ObservableObject {
    @Published var connectionState: RTCIceConnectionState = .new
    @Published var messages: [ChatMessage] = []
    @Published var localVideoTrack: RTCVideoTrack?
    @Published var isStreaming = false

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
    private var videoCapturer: RTCCameraVideoCapturer?
    private var localAudioTrack: RTCAudioTrack?
    fileprivate var signalingClient = SignalingClient()
    private var pendingICECandidates: [RTCIceCandidate] = []
    private var hasRemoteDescription = false
    private let serverURL: URL

    init(serverURL: URL = URL(string: "wss://imperceptible-makena-overabusively.ngrok-free.dev/ws")!) {
        self.serverURL = serverURL
    }

    // MARK: - Connect

    func connect() async {
        configureAudioSession()
        createPeerConnection()
        addMediaTracks()
        setupSignaling()
        signalingClient.connect(to: serverURL)
        isStreaming = true
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

        // Video
        let videoSource = Self.factory.videoSource()
        let capturer = RTCCameraVideoCapturer(delegate: videoSource)
        videoCapturer = capturer

        let videoTrack = Self.factory.videoTrack(with: videoSource, trackId: "video0")
        pc.add(videoTrack, streamIds: ["stream0"])
        localVideoTrack = videoTrack

        startCameraCapture()
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
                await self?.createAndSendOffer()
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
        videoCapturer?.stopCapture()

        localVideoTrack?.isEnabled = false
        localAudioTrack?.isEnabled = false
        localVideoTrack = nil
        localAudioTrack = nil

        peerConnection?.close()
        peerConnection = nil

        signalingClient.disconnect()
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

// MARK: - Errors

enum WebRTCError: Error {
    case noDescription
}
