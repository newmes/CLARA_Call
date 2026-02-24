import SwiftUI

struct HomeView: View {
    @StateObject private var manager = WebRTCManager()
    @StateObject private var classifier = MedSigLIPClassifier()
    @State private var showLiveStream = false

    private var isLoading: Bool {
        classifier.isLoading
    }

    private var loadingStatus: String {
        "Loading vision model…"
    }

    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()

            VStack {
                Spacer()

                Image(uiImage: UIImage(named: "CLARA_new") ?? UIImage())
                    .resizable()
                    .scaledToFit()
                    .frame(height: 170)
                    .clipShape(RoundedRectangle(cornerRadius: 12))

                Text("CLARA")
                    .font(.system(size: 48, weight: .bold))
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
            await classifier.load()
            manager.setClassifier(classifier)
        }
        .fullScreenCover(isPresented: $showLiveStream) {
            LiveStreamView(manager: manager, classifier: classifier)
        }
    }
}
