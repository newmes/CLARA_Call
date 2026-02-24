import SwiftUI

struct HomeView: View {
    @StateObject private var manager = WebRTCManager()
    @StateObject private var classifier = MedSigLIPClassifier()
    @State private var showLiveStream = false
    @State private var isPulsing = false

    private var isLoading: Bool {
        classifier.isLoading
    }

    var body: some View {
        ZStack {
            // Dark gradient background
            LinearGradient(
                colors: [.black, Color(.darkGray), .black],
                startPoint: .top,
                endPoint: .bottom
            )
            .ignoresSafeArea()

            VStack(spacing: 16) {
                Spacer()

                // Avatar with pulsing ring
                ZStack {
                    // Pulsing rings (only when not loading)
                    if !isLoading {
                        Circle()
                            .stroke(Color.green.opacity(0.4), lineWidth: 2)
                            .frame(width: 150, height: 150)
                            .scaleEffect(isPulsing ? 2.0 : 1.0)
                            .opacity(isPulsing ? 0 : 0.4)

                        Circle()
                            .stroke(Color.green.opacity(0.4), lineWidth: 2)
                            .frame(width: 150, height: 150)
                            .scaleEffect(isPulsing ? 2.0 : 1.0)
                            .opacity(isPulsing ? 0 : 0.4)
                            .animation(
                                .easeInOut(duration: 1.5)
                                    .repeatForever(autoreverses: false)
                                    .delay(0.5),
                                value: isPulsing
                            )
                    }

                    // Avatar image
                    Image("CLARA_new")
                        .resizable()
                        .scaledToFill()
                        .frame(width: 150, height: 150)
                        .clipShape(Circle())
                }
                .animation(
                    .easeInOut(duration: 1.5)
                        .repeatForever(autoreverses: false),
                    value: isPulsing
                )

                // Text section
                VStack(spacing: 6) {
                    Text("CLARA")
                        .font(.title.bold())
                        .foregroundStyle(.white)

                    Text("Clinical Longitudinal AI Research Assistant")
                        .font(.caption)
                        .foregroundStyle(.white.opacity(0.6))
                }

                Spacer()
                    .frame(height: 8)

                // Status
                if isLoading {
                    HStack(spacing: 8) {
                        ProgressView()
                            .tint(.white)
                        Text("Loading model...")
                            .font(.subheadline)
                            .foregroundStyle(.white.opacity(0.8))
                    }
                } else {
                    Text("Incoming Video Call...")
                        .font(.subheadline)
                        .foregroundStyle(.green)
                }

                Spacer()

                // Accept button (only when not loading)
                if !isLoading {
                    VStack(spacing: 8) {
                        Button {
                            showLiveStream = true
                        } label: {
                            Image(systemName: "phone.fill")
                                .font(.system(size: 24))
                                .foregroundStyle(.white)
                                .frame(width: 60, height: 60)
                                .background(Color.green)
                                .clipShape(Circle())
                        }

                        Text("Accept")
                            .font(.caption)
                            .foregroundStyle(.white.opacity(0.8))
                    }
                    .transition(.opacity)
                }

                Spacer()
                    .frame(height: 50)
            }
        }
        .onChange(of: isLoading) { _, loading in
            if !loading {
                isPulsing = true
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
