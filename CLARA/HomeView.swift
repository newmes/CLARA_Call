import SwiftUI

struct HomeView: View {
    @StateObject private var manager = WebRTCManager()
    @StateObject private var audioTranscriber = AudioTranscriber()
    @StateObject private var classifier = MedSigLIPClassifier()
    @State private var showLiveStream = false

    private var isLoading: Bool {
        classifier.isLoading || audioTranscriber.state != .ready
    }

    private var loadingStatus: String {
        var pending: [String] = []
        if classifier.isLoading { pending.append("vision model") }
        if audioTranscriber.state != .ready { pending.append("speech model") }
        return "Loading \(pending.joined(separator: " & "))…"
    }

    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()

            VStack {
                Spacer()

                Text("CLARA")
                    .font(.system(size: 48, weight: .bold, design: .default))
                    .foregroundStyle(.white)
                Text("Clinical Longitudinal AI Research Assistant")
                    .font(.subheadline)
                    .foregroundStyle(.white.opacity(0.6))

                Spacer()

                if isLoading {
                    VStack(spacing: 12) {
                        ProgressView()
                            .tint(.white)
                        Text(loadingStatus)
                            .font(.caption)
                            .foregroundStyle(.white.opacity(0.6))
                    }
                    .padding(.bottom, 48)
                } else {
                    Button {
                        showLiveStream = true
                    } label: {
                        Text("Start Demo")
                            .font(.headline)
                            .padding(.horizontal, 36)
                            .padding(.vertical, 16)
                            .background(.white, in: Capsule())
                            .foregroundStyle(.black)
                    }
                    .padding(.bottom, 48)
                }
            }
        }
        .task {
            async let a: () = audioTranscriber.loadModel()
            async let b: () = classifier.load()
            _ = await (a, b)
            manager.setClassifier(classifier)
        }
        .fullScreenCover(isPresented: $showLiveStream) {
            LiveStreamView(manager: manager, audioTranscriber: audioTranscriber, classifier: classifier)
        }
    }
}
