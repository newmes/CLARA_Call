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
    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()

            VStack(spacing: 12) {
                Spacer()
                controlBar
                chatOverlay
                ZStack {
                    talkButton

                    HStack {
                        Spacer()
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
                    }
                }
                .padding(.horizontal, 16)
                .padding(.bottom, 16)
            }
        }
        .onAppear {
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

    // MARK: - Control Bar

    private var controlBar: some View {
        HStack(spacing: 16) {
            VideoRendererView(track: manager.localVideoTrack)
                .frame(width: 120, height: 120)
                .clipShape(RoundedRectangle(cornerRadius: 12))

            VStack(spacing: 10) {
                Text("Privacy Mode enabled")
                    .font(.caption.weight(.medium))
                    .foregroundStyle(.white.opacity(0.6))

                Button { manager.disconnect(); dismiss() } label: {
                    Text("Hang Up")
                        .font(.subheadline.weight(.semibold))
                        .foregroundStyle(.white)
                        .padding(.horizontal, 24)
                        .padding(.vertical, 10)
                        .background(Color.red, in: Capsule())
                }
            }
            .frame(maxWidth: .infinity)
        }
        .padding(16)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 20))
        .padding(.horizontal, 16)
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
                    ForEach(manager.messages) { message in
                        chatBubble(message)
                            .id(message.id)
                    }
                }
                .padding(.horizontal)
            }
            .onChange(of: manager.messages.count) {
                if let last = manager.messages.last {
                    withAnimation {
                        proxy.scrollTo(last.id, anchor: .bottom)
                    }
                }
            }
        }
        .padding(.vertical, 16)
    }

    private func chatBubble(_ message: ChatMessage) -> some View {
        HStack(alignment: .bottom, spacing: 8) {
            if !message.isFromServer { Spacer() }
            if message.isFromServer {
                Image(uiImage: UIImage(named: "CLARA_cropped")!)
                    .resizable()
                    .scaledToFill()
                    .frame(width: 28, height: 28)
                    .clipShape(Circle())
            }
            HStack(spacing: 6) {
                Text(message.text)
                    .font(.subheadline)
                if message.audioData != nil {
                    Image(systemName: "speaker.wave.2.fill")
                        .font(.caption2)
                        .foregroundStyle(.white.opacity(0.6))
                }
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
            .onTapGesture {
                if let audio = message.audioData {
                    manager.playAudio(data: audio)
                }
            }
            if message.isFromServer { Spacer() }
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
