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
    @ObservedObject var classifier: MedSigLIPClassifier
    @State private var isHoldingTalk = false
    @State private var isTalkingToggled = false
    @State private var pushToTalkMode = true
    @State private var pipOffset: CGSize = .zero
    @State private var dragOffset: CGSize = .zero

    private let pipSize = CGSize(width: 120, height: 160)

    var body: some View {
        GeometryReader { geo in
            ZStack {
                // Background (extends behind safe area)
                Color.black.ignoresSafeArea()

                // UI overlay
                VStack {
                    topBar
                    Spacer()

                    // Caller's view — CLARA image
                    if let img = UIImage(named: "CLARA_Image") {
                        Image(uiImage: img)
                            .resizable()
                            .scaledToFit()
                            .clipShape(RoundedRectangle(cornerRadius: 20))
                            .padding(.horizontal, 24)
                    }

                    chatOverlay
                    talkControls
                }

                // PIP camera preview (on top)
                VideoRendererView(track: manager.localVideoTrack)
                    .frame(width: pipSize.width, height: pipSize.height)
                    .clipShape(RoundedRectangle(cornerRadius: 12))
                    .shadow(color: .black.opacity(0.5), radius: 6, x: 0, y: 3)
                    .position(pipPosition(in: geo.size))
                    .gesture(
                        DragGesture()
                            .onChanged { value in
                                dragOffset = value.translation
                            }
                            .onEnded { value in
                                pipOffset = CGSize(
                                    width: pipOffset.width + value.translation.width,
                                    height: pipOffset.height + value.translation.height
                                )
                                dragOffset = .zero
                                snapToCorner(in: geo.size)
                            }
                    )
            }
        }
        .onAppear {
            manager.startStreaming()
            audioTranscriber.startListening()
            audioTranscriber.onTranscription = { [weak manager] text in
                guard let manager else { return }
                manager.messages.append(ChatMessage(text: text, isFromServer: false))
                manager.consultCareAI(patientText: text)
            }
        }
        .onDisappear {
            audioTranscriber.stopTranscribing()
            manager.disconnect()
        }
    }

    // MARK: - PIP Positioning

    private func pipPosition(in containerSize: CGSize) -> CGPoint {
        let padding: CGFloat = 16
        let defaultX = containerSize.width - pipSize.width / 2 - padding
        let defaultY = padding + 60 + pipSize.height / 2 // below safe area
        return CGPoint(
            x: defaultX + pipOffset.width + dragOffset.width,
            y: defaultY + pipOffset.height + dragOffset.height
        )
    }

    private func snapToCorner(in containerSize: CGSize) {
        let padding: CGFloat = 16
        let currentPos = pipPosition(in: containerSize)

        let midX = containerSize.width / 2
        let midY = containerSize.height / 2

        let targetX: CGFloat = currentPos.x < midX
            ? padding + pipSize.width / 2
            : containerSize.width - pipSize.width / 2 - padding
        let targetY: CGFloat = currentPos.y < midY
            ? padding + 60 + pipSize.height / 2
            : containerSize.height - pipSize.height / 2 - padding - 80

        let defaultX = containerSize.width - pipSize.width / 2 - padding
        let defaultY = padding + 60 + pipSize.height / 2

        withAnimation(.spring(response: 0.35, dampingFraction: 0.75)) {
            pipOffset = CGSize(
                width: targetX - defaultX,
                height: targetY - defaultY
            )
        }
    }

    // MARK: - Top Bar

    private var topBar: some View {
        HStack {
            connectionIndicator
            Spacer()
            Button(action: { manager.disconnect(); dismiss() }) {
                Image(systemName: "xmark.circle.fill")
                    .font(.title2)
                    .foregroundStyle(.white)
            }
        }
        .padding(.horizontal)
        .padding(.top, 8)
    }

    private var connectionIndicator: some View {
        HStack(spacing: 6) {
            Circle()
                .fill(.green)
                .frame(width: 10, height: 10)
            Text("Private")
                .font(.caption.weight(.medium))
                .foregroundStyle(.white)
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 6)
        .background(.ultraThinMaterial, in: Capsule())
    }

    // MARK: - Push to Talk

    private var isListening: Bool {
        pushToTalkMode ? isHoldingTalk : isTalkingToggled
    }

    private var talkControls: some View {
        VStack(spacing: 12) {
            talkButton
            pttToggle
        }
        .padding(.bottom, 32)
    }

    private var talkButton: some View {
        HStack(spacing: 10) {
            Image(systemName: isListening ? "mic.fill" : "mic")
                .font(.title3)
            Text(isListening ? "Listening..." : (pushToTalkMode ? "Hold to Talk" : "Tap to Talk"))
                .font(.subheadline.weight(.medium))
        }
        .foregroundStyle(.white)
        .padding(.horizontal, 28)
        .padding(.vertical, 16)
        .background(
            isListening ? Color.red.opacity(0.8) : Color.white.opacity(0.2),
            in: Capsule()
        )
        .background(.ultraThinMaterial, in: Capsule())
        .gesture(pushToTalkMode ? holdGesture : nil)
        .onTapGesture {
            guard !pushToTalkMode else { return }
            if isTalkingToggled {
                isTalkingToggled = false
                audioTranscriber.endUtteranceAndTranscribe()
            } else {
                isTalkingToggled = true
                audioTranscriber.beginUtterance()
            }
        }
    }

    private var holdGesture: some Gesture {
        DragGesture(minimumDistance: 0)
            .onChanged { _ in
                if !isHoldingTalk {
                    isHoldingTalk = true
                    audioTranscriber.beginUtterance()
                }
            }
            .onEnded { _ in
                isHoldingTalk = false
                audioTranscriber.endUtteranceAndTranscribe()
            }
    }

    private var pttToggle: some View {
        HStack(spacing: 6) {
            Text("Hold")
                .foregroundStyle(pushToTalkMode ? .white : .white.opacity(0.5))
            Toggle("", isOn: $pushToTalkMode)
                .labelsHidden()
                .tint(.gray)
            Text("Tap")
                .foregroundStyle(!pushToTalkMode ? .white : .white.opacity(0.5))
        }
        .font(.caption.weight(.medium))
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
