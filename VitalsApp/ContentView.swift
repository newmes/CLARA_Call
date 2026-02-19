import SwiftUI
import PhotosUI

struct ContentView: View {
    @StateObject private var classifier = MedSigLIPClassifier()

    @State private var selectedPhoto: PhotosPickerItem?
    @State private var displayImage: UIImage?
    @State private var labelsText = "pneumonia, pleural effusion, normal chest x-ray"
    @State private var results: [(label: String, score: Float)] = []
    @State private var isClassifying = false
    @State private var showCamera = false
    @State private var errorMessage: String?

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(spacing: 20) {
                    imageSection
                    labelsSection
                    classifyButton

                    if let times = classifier.loadTimes {
                        loadTimesSection(times)
                    }

                    if let loadError = classifier.error {
                        Text("Load error: \(loadError)")
                            .foregroundStyle(.red)
                            .font(.caption)
                            .padding(.horizontal)
                    }
                    if let errorMessage {
                        Text(errorMessage)
                            .foregroundStyle(.red)
                            .font(.caption)
                            .padding(.horizontal)
                    }

                    if !results.isEmpty {
                        resultsSection
                    }
                }
                .padding()
            }
            .navigationTitle("VitalsApp")
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    if classifier.isLoading { ProgressView() }
                }
            }
        }
        .task { await classifier.load() }
        .onChange(of: selectedPhoto) {
            Task { await loadPhoto() }
        }
        .fullScreenCover(isPresented: $showCamera) {
            CameraView(image: $displayImage).ignoresSafeArea()
        }
    }

    // MARK: - Sections

    private var imageSection: some View {
        Group {
            if let displayImage {
                Image(uiImage: displayImage)
                    .resizable()
                    .aspectRatio(contentMode: .fit)
                    .frame(maxHeight: 300)
                    .clipShape(RoundedRectangle(cornerRadius: 12))
            } else {
                RoundedRectangle(cornerRadius: 12)
                    .fill(.quaternary)
                    .frame(height: 200)
                    .overlay {
                        VStack(spacing: 8) {
                            Image(systemName: "photo.on.rectangle").font(.largeTitle)
                            Text("Select or capture an image").font(.subheadline)
                        }
                        .foregroundStyle(.secondary)
                    }
            }

            HStack(spacing: 16) {
                PhotosPicker(selection: $selectedPhoto, matching: .images) {
                    Label("Photos", systemImage: "photo.on.rectangle")
                }
                .buttonStyle(.bordered)

                Button { showCamera = true } label: {
                    Label("Camera", systemImage: "camera")
                }
                .buttonStyle(.bordered)
            }
        }
    }

    private var labelsSection: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("Labels (comma-separated)")
                .font(.subheadline)
                .foregroundStyle(.secondary)
            TextField("e.g. pneumonia, normal, fracture", text: $labelsText)
                .textFieldStyle(.roundedBorder)
                .autocorrectionDisabled()
                .textInputAutocapitalization(.never)
        }
    }

    private var classifyButton: some View {
        Button {
            Task { await classify() }
        } label: {
            HStack {
                if isClassifying { ProgressView().tint(.white) }
                Text(isClassifying ? "Classifying..." : "Classify")
            }
            .frame(maxWidth: .infinity)
        }
        .buttonStyle(.borderedProminent)
        .disabled(displayImage == nil || isClassifying || classifier.isLoading || labelsText.isEmpty)
    }

    private var resultsSection: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Results").font(.headline)

            ForEach(Array(results.enumerated()), id: \.offset) { _, item in
                VStack(alignment: .leading, spacing: 4) {
                    HStack {
                        Text(item.label).font(.subheadline.weight(.medium))
                        Spacer()
                        Text(String(format: "%.1f%%", item.score * 100))
                            .font(.caption.monospaced())
                            .foregroundStyle(.secondary)
                    }
                    GeometryReader { geo in
                        RoundedRectangle(cornerRadius: 4)
                            .fill(.blue.gradient)
                            .frame(width: max(0, geo.size.width * CGFloat(item.score)))
                    }
                    .frame(height: 8)
                }
            }
        }
        .padding()
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 12))
    }

    private func loadTimesSection(_ times: MedSigLIPClassifier.LoadTimes) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Text("Model Load Times")
                .font(.caption.weight(.semibold))
                .foregroundStyle(.secondary)
            HStack(spacing: 16) {
                loadTimeLabel("Vision", times.visionEncoder)
                loadTimeLabel("Text", times.textEncoder)
                loadTimeLabel("Tokenizer", times.tokenizer)
            }
            Text(String(format: "Total: %.2fs", times.total))
                .font(.caption2.monospaced())
                .foregroundStyle(.tertiary)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(10)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 8))
    }

    private func loadTimeLabel(_ name: String, _ time: TimeInterval) -> some View {
        VStack(alignment: .leading, spacing: 1) {
            Text(name).font(.caption2).foregroundStyle(.tertiary)
            Text(String(format: "%.2fs", time)).font(.caption.monospaced()).foregroundStyle(.secondary)
        }
    }

    // MARK: - Actions

    private func loadPhoto() async {
        guard let selectedPhoto else { return }
        if let data = try? await selectedPhoto.loadTransferable(type: Data.self),
           let uiImage = UIImage(data: data) {
            displayImage = uiImage
            results = []
        }
    }

    private func classify() async {
        guard let cgImage = displayImage?.cgImage else { return }
        isClassifying = true
        errorMessage = nil

        do {
            let labels = labelsText
                .split(separator: ",")
                .map { $0.trimmingCharacters(in: .whitespaces) }
                .filter { !$0.isEmpty }
            guard !labels.isEmpty else {
                isClassifying = false
                return
            }
            results = try await classifier.classify(image: cgImage, labels: labels)
        } catch {
            errorMessage = error.localizedDescription
        }
        isClassifying = false
    }
}

// MARK: - Camera View

struct CameraView: UIViewControllerRepresentable {
    @Binding var image: UIImage?
    @Environment(\.dismiss) private var dismiss

    func makeUIViewController(context: Context) -> UIImagePickerController {
        let picker = UIImagePickerController()
        picker.sourceType = .camera
        picker.delegate = context.coordinator
        return picker
    }

    func updateUIViewController(_ uiViewController: UIImagePickerController, context: Context) {}
    func makeCoordinator() -> Coordinator { Coordinator(self) }

    class Coordinator: NSObject, UIImagePickerControllerDelegate, UINavigationControllerDelegate {
        let parent: CameraView
        init(_ parent: CameraView) { self.parent = parent }

        func imagePickerController(_ picker: UIImagePickerController,
                                   didFinishPickingMediaWithInfo info: [UIImagePickerController.InfoKey: Any]) {
            if let uiImage = info[.originalImage] as? UIImage { parent.image = uiImage }
            parent.dismiss()
        }

        func imagePickerControllerDidCancel(_ picker: UIImagePickerController) {
            parent.dismiss()
        }
    }
}
