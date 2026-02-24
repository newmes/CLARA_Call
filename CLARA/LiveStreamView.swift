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
    @State private var showTalkSettings = false
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
                        Button { showTalkSettings.toggle() } label: {
                            Image(systemName: "gearshape.fill")
                                .font(.subheadline)
                                .foregroundStyle(.white.opacity(0.7))
                                .padding(10)
                                .background(.ultraThinMaterial, in: Circle())
                        }
                        .popover(isPresented: $showTalkSettings,
                                 attachmentAnchor: .point(.top),
                                 arrowEdge: .bottom) {
                            pttToggle
                                .padding()
                                .presentationCompactAdaptation(.popover)
                        }

                        Spacer()

                        Button { manager.disconnect(); dismiss() } label: {
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
            audioTranscriber.startListening()
            audioTranscriber.onTranscription = { [weak manager] text, audioData in
                guard let manager else { return }
                manager.messages.append(ChatMessage(text: text, isFromServer: false, audioData: audioData))
                manager.consultCareAI(patientText: text)
            }
        }
        .onDisappear {
            audioTranscriber.stopTranscribing()
            manager.disconnect()
        }
    }

    // MARK: - Header

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
        } else if let demo = UIImage(named: "demo1") {
            Image(uiImage: demo)
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

    private var mediaPanel: some View {
        ZStack(alignment: .bottomTrailing) {
            Group {
                if cameraIsMain { cameraView } else { claraView }
            }
            .frame(maxWidth: .infinity)
            .frame(height: 280)
            .offset(y: 30)
            .clipShape(RoundedRectangle(cornerRadius: 12))
            .overlay(
                !cameraIsMain
                    ? RoundedRectangle(cornerRadius: 12).strokeBorder(.green, lineWidth: 2)
                    : nil
            )

            Group {
                if cameraIsMain { claraView } else { cameraView }
            }
            .frame(width: 100, height: 100)
            .clipShape(RoundedRectangle(cornerRadius: 12))
            .overlay(
                cameraIsMain
                    ? RoundedRectangle(cornerRadius: 12).strokeBorder(.green, lineWidth: 2)
                    : nil
            )
            .padding(8)
        }
        .padding(.horizontal, 16)
        .onTapGesture {
            withAnimation(.easeInOut(duration: 0.3)) {
                cameraIsMain.toggle()
            }
        }
    }

    // MARK: - Push to Talk

    private var isListening: Bool {
        pushToTalkMode ? isHoldingTalk : isTalkingToggled
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
            isListening ? Color.red.opacity(0.8) : Color.green,
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
        audioTranscriber: AudioTranscriber(),
        classifier: MedSigLIPClassifier()
    )
}
