import SwiftUI
import WebRTC

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

struct LiveStreamView: View {
    @Environment(\.dismiss) private var dismiss
    @ObservedObject var manager: WebRTCManager
    @ObservedObject var audioTranscriber: AudioTranscriber

    var body: some View {
        ZStack {
            // Camera preview
            VideoRendererView(track: manager.localVideoTrack)
                .ignoresSafeArea()

            // Gradient overlay
            VStack {
                LinearGradient(colors: [.black.opacity(0.6), .clear], startPoint: .top, endPoint: .bottom)
                    .frame(height: 120)
                Spacer()
                LinearGradient(colors: [.clear, .black.opacity(0.6)], startPoint: .top, endPoint: .bottom)
                    .frame(height: 200)
            }
            .ignoresSafeArea()

            // UI overlay
            VStack {
                topBar
                Spacer()
                if manager.isStreaming {
                    chatOverlay
                } else {
                    startButton
                }
            }
        }
        .onAppear {
            manager.startPreview()
            audioTranscriber.onTranscription = { [weak manager] text in
                guard let manager else { return }
                manager.messages.append(ChatMessage(text: text, isFromServer: false))
            }
        }
        .onDisappear {
            audioTranscriber.stopTranscribing()
            manager.disconnect()
        }
    }

    // MARK: - Top Bar

    private var topBar: some View {
        HStack {
            if manager.isStreaming {
                connectionIndicator
            }
            Spacer()
            if manager.isStreaming {
                Button(action: { manager.disconnect(); dismiss() }) {
                    Image(systemName: "xmark.circle.fill")
                        .font(.title2)
                        .foregroundStyle(.white)
                }
            }
        }
        .padding(.horizontal)
        .padding(.top, 8)
    }

    private var startButton: some View {
        Button {
            Task {
                await manager.startStreaming()
                audioTranscriber.startTranscribing()
            }
        } label: {
            Label("Start Streaming", systemImage: "video.fill")
                .font(.headline)
                .padding(.horizontal, 24)
                .padding(.vertical, 14)
                .background(.ultraThinMaterial, in: Capsule())
                .foregroundStyle(.white)
        }
        .padding(.bottom, 60)
    }

    private var connectionIndicator: some View {
        HStack(spacing: 6) {
            Circle()
                .fill(connectionColor)
                .frame(width: 10, height: 10)
            Text(connectionLabel)
                .font(.caption.weight(.medium))
                .foregroundStyle(.white)
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 6)
        .background(.ultraThinMaterial, in: Capsule())
    }

    private var connectionColor: Color {
        switch manager.connectionState {
        case .connected, .completed: return .green
        case .checking: return .yellow
        case .disconnected, .failed, .closed: return .red
        default: return .gray
        }
    }

    private var connectionLabel: String {
        switch manager.connectionState {
        case .new: return "New"
        case .checking: return "Connecting..."
        case .connected: return "Connected"
        case .completed: return "Connected"
        case .disconnected: return "Disconnected"
        case .failed: return "Failed"
        case .closed: return "Closed"
        @unknown default: return "Unknown"
        }
    }

    // MARK: - Chat Overlay

    private var chatOverlay: some View {
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 8) {
                    ForEach(manager.messages) { message in
                        chatBubble(message)
                            .id(message.id)
                    }
                }
                .padding(.horizontal)
            }
            .frame(maxHeight: 200)
            .onChange(of: manager.messages.count) {
                if let last = manager.messages.last {
                    withAnimation {
                        proxy.scrollTo(last.id, anchor: .bottom)
                    }
                }
            }
        }
        .padding(.bottom, 16)
    }

    private func chatBubble(_ message: ChatMessage) -> some View {
        HStack {
            if !message.isFromServer { Spacer() }
            HStack(spacing: 6) {
                if message.isFromServer {
                    Image(systemName: "bolt.heart.fill")
                        .symbolRenderingMode(.hierarchical)
                        .symbolEffect(.bounce.down.byLayer, options: .repeat(.periodic(delay: 0.7)))
                        .font(.caption)
                }
                Text(message.text)
                    .font(.subheadline)
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 8)
            .background(
                message.isFromServer
                    ? Color.white.opacity(0.2)
                    : Color.blue.opacity(0.7)
            )
            .foregroundStyle(.white)
            .clipShape(RoundedRectangle(cornerRadius: 12))
            if message.isFromServer { Spacer() }
        }
    }
}
