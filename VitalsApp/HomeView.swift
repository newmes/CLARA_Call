import SwiftUI

struct HomeView: View {
    @StateObject private var manager = WebRTCManager()
    @StateObject private var audioTranscriber = AudioTranscriber()
    @StateObject private var classifier = MedSigLIPClassifier()
    @State private var showLiveStream = false

    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()

            VStack {
                Spacer()

                Text("Vitals")
                    .font(.system(size: 48, weight: .bold, design: .default))
                    .foregroundStyle(.white)

                Spacer()

                Button {
                    manager.connectSignaling()
                    showLiveStream = true
                } label: {
                    Text("Start Session")
                        .font(.headline)
                        .padding(.horizontal, 36)
                        .padding(.vertical, 16)
                        .background(.white, in: Capsule())
                        .foregroundStyle(.black)
                }
                .padding(.bottom, 48)
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
