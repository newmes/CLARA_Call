import AVFoundation
import SwiftUI
import WebRTC

// MARK: - Video Renderer (WebRTC)

struct VideoRendererView: UIViewRepresentable {
    let track: RTCVideoTrack?

    func makeUIView(context: Context) -> RTCMTLVideoView {
        let view = RTCMTLVideoView()
        view.videoContentMode = .scaleAspectFill
        view.transform = CGAffineTransform(scaleX: -1, y: 1)
        return view
    }

    func updateUIView(_ uiView: RTCMTLVideoView, context: Context) {
        if let track {
            track.add(uiView)
            context.coordinator.currentTrack = track
            context.coordinator.renderer = uiView
        }
    }

    func makeCoordinator() -> Coordinator { Coordinator() }

    static func dismantleUIView(_ uiView: RTCMTLVideoView, coordinator: Coordinator) {
        coordinator.currentTrack?.remove(uiView)
    }

    class Coordinator {
        var currentTrack: RTCVideoTrack?
        var renderer: RTCMTLVideoView?
    }
}

// MARK: - Video Player (AVPlayer)

struct VideoPlayerView: UIViewRepresentable {
    let player: AVPlayer

    func makeUIView(context: Context) -> PlayerUIView { PlayerUIView() }

    func updateUIView(_ uiView: PlayerUIView, context: Context) {
        uiView.playerLayer.player = player
    }

    class PlayerUIView: UIView {
        override class var layerClass: AnyClass { AVPlayerLayer.self }
        var playerLayer: AVPlayerLayer { layer as! AVPlayerLayer }

        override init(frame: CGRect) {
            super.init(frame: frame)
            playerLayer.videoGravity = .resizeAspectFill
        }

        required init?(coder: NSCoder) { fatalError() }
    }
}

// MARK: - Live Stream View

struct LiveStreamView: View {
    @Environment(\.dismiss) private var dismiss
    @ObservedObject var manager: WebRTCManager
    @ObservedObject var classifier: MedSigLIPClassifier
    @StateObject private var demo = DemoOrchestrator()
    @State private var cameraIsMain = true

    private let isPreview = ProcessInfo.processInfo.environment["XCODE_RUNNING_FOR_PREVIEWS"] != nil

    var body: some View {
        ZStack {
            Color.black.opacity(0.9).ignoresSafeArea()

            VStack(spacing: 12) {
                Spacer()
                mediaPanel
                chatOverlay
            }

            VStack {
                Spacer()
                ZStack {
                    talkButton

                    HStack {
                        Spacer()

                        Button {
                            demo.cleanup()
                            manager.disconnect()
                            dismiss()
                        } label: {
                            Image(systemName: "phone.down.fill")
                                .font(.subheadline.weight(.semibold))
                                .foregroundStyle(.white)
                                .padding(10)
                                .background(Color.red, in: Circle())
                        }
                    }
                }
                .padding(.horizontal, 16)
                .padding(.bottom, 16)
            }
        }
        .onAppear {
            guard !isPreview else { return }
            manager.startStreaming()
            demo.manager = manager
            demo.classifier = classifier
            demo.start()
        }
        .onDisappear {
            demo.cleanup()
            manager.disconnect()
        }
    }

    // MARK: - Subviews

    @ViewBuilder
    private var claraView: some View {
        Image(uiImage: UIImage(named: "CLARA_new") ?? UIImage())
            .resizable()
            .scaledToFill()
    }

    @ViewBuilder
    private var cameraView: some View {
        if manager.localVideoTrack != nil {
            VideoRendererView(track: manager.localVideoTrack)
        } else if let demoImg = UIImage(named: "demo1") {
            Image(uiImage: demoImg)
                .resizable()
                .scaledToFill()
        } else {
            RoundedRectangle(cornerRadius: 12)
                .fill(.quaternary)
                .overlay {
                    Image(systemName: "video.fill")
                        .font(.title2)
                        .foregroundStyle(.white.opacity(0.4))
                }
        }
    }

    // MARK: - Media Panel

    private var mediaPanel: some View {
        ZStack(alignment: .bottomTrailing) {
            // Main panel
            Group {
                if let player = demo.currentPlayer {
                    VideoPlayerView(player: player)
                } else if cameraIsMain {
                    cameraView
                } else {
                    claraView
                }
            }
            .frame(maxWidth: .infinity)
            .frame(height: 280)
            .offset(y: 30)
            .clipShape(RoundedRectangle(cornerRadius: 12))
            .overlay(
                demo.currentPlayer == nil && !cameraIsMain
                    ? RoundedRectangle(cornerRadius: 12).strokeBorder(.green, lineWidth: 2)
                    : nil
            )

            // PiP
            Group {
                if demo.currentPlayer != nil {
                    claraView
                } else if cameraIsMain {
                    claraView
                } else {
                    cameraView
                }
            }
            .frame(width: 100, height: 100)
            .clipShape(RoundedRectangle(cornerRadius: 12))
            .overlay(
                demo.currentPlayer == nil && cameraIsMain
                    ? RoundedRectangle(cornerRadius: 12).strokeBorder(.green, lineWidth: 2)
                    : nil
            )
            .padding(8)
        }
        .padding(.horizontal, 16)
        .onTapGesture {
            guard demo.currentPlayer == nil else { return }
            withAnimation(.easeInOut(duration: 0.3)) {
                cameraIsMain.toggle()
            }
        }
    }

    // MARK: - Talk Button

    private var talkButton: some View {
        HStack(spacing: 10) {
            Image(systemName: demo.isListening ? "mic.fill" : "mic")
                .font(.title3)
            Text(talkButtonLabel)
                .font(.subheadline.weight(.medium))
        }
        .foregroundStyle(.white)
        .padding(.horizontal, 28)
        .padding(.vertical, 16)
        .background(
            demo.isListening ? Color.red.opacity(0.8) : Color.green,
            in: Capsule()
        )
        .background(.ultraThinMaterial, in: Capsule())
        .opacity(demo.talkButtonEnabled ? 1.0 : 0.5)
        .allowsHitTesting(demo.talkButtonEnabled)
        .onTapGesture {
            demo.handleTalkTap()
        }
    }

    private var talkButtonLabel: String {
        switch demo.step {
        case .playingVideo1, .playingVideo2: return "Listening..."
        case .consulting: return "Processing..."
        case .claraResponding: return "CLARA speaking..."
        case .done: return "Demo Complete"
        default: return "Tap to Talk"
        }
    }

    // MARK: - Chat Overlay

    private var chatOverlay: some View {
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 8) {
                    ForEach(manager.messages.filter(\.isFromServer)) { message in
                        chatBubble(message)
                            .id(message.id)
                    }
                }
                .padding(.horizontal)
                .padding(.bottom, 80)
            }
            .onChange(of: manager.messages.count) {
                if let last = manager.messages.last {
                    withAnimation {
                        proxy.scrollTo(last.id, anchor: .bottom)
                    }
                }
            }
        }
        .mask(
            LinearGradient(
                stops: [
                    .init(color: .clear, location: 0),
                    .init(color: .black, location: 0.08),
                    .init(color: .black, location: 0.85),
                    .init(color: .clear, location: 1),
                ],
                startPoint: .top,
                endPoint: .bottom
            )
        )
    }

    private func chatBubble(_ message: ChatMessage) -> some View {
        HStack(alignment: .bottom, spacing: 8) {
            Text(message.text)
                .font(.subheadline)
                .padding(.horizontal, 12)
                .padding(.vertical, 8)
                .background(Color.white.opacity(0.2))
                .foregroundStyle(.white)
                .clipShape(RoundedRectangle(cornerRadius: 12))
        }
    }
}

#Preview {
    LiveStreamView(
        manager: WebRTCManager(),
        classifier: MedSigLIPClassifier()
    )
}
