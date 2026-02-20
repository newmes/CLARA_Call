import Accelerate
import CoreML
import Foundation

/// Computes log mel spectrograms from raw audio, matching the Python
/// `LasrFeatureExtractor._torch_extract_fbank_features()` implementation.
///
/// Parameters (must match exactly):
/// - Sample rate: 16000 Hz
/// - n_fft: 512
/// - hop_length: 160
/// - win_length: 400
/// - Mel bins: 128, Kaldi scale, 125-7500 Hz
/// - Window: Hann (periodic=false)
/// - Pipeline: STFT -> power spectrum -> mel filterbank -> clamp(min=1e-5) -> log
///
/// Reference: WhisperKit's FeatureExtractor.swift for the vDSP FFT pattern.
final class MelSpectrogramExtractor {

    // MARK: - Configuration

    static let sampleRate: Int = 16_000
    static let nFFT: Int = 512
    static let hopLength: Int = 160
    static let winLength: Int = 400
    static let numMelBins: Int = 128
    static let melFMin: Float = 125.0
    static let melFMax: Float = 7500.0

    // MARK: - Precomputed State

    /// Hann window of length winLength (periodic=false, i.e., symmetric)
    private let hannWindow: [Float]

    /// Mel filterbank matrix [numMelBins x (nFFT/2 + 1)]
    /// Stored as a flat array in row-major order.
    private let melFilterbank: [Float]

    /// FFT setup for vDSP
    private let fftSetup: vDSP.FFT<DSPSplitComplex>

    /// Log base 2 of nFFT, needed for vDSP FFT
    private let log2n: vDSP_Length

    /// Number of frequency bins = nFFT / 2 + 1
    private let numFreqBins: Int

    // MARK: - Initialization

    init() {
        let nFFT = Self.nFFT
        let winLength = Self.winLength
        let numMelBins = Self.numMelBins

        self.numFreqBins = nFFT / 2 + 1
        self.log2n = vDSP_Length(log2(Double(nFFT)))

        // Symmetric Hann window (periodic=false): w[n] = 0.5 * (1 - cos(2*pi*n / (N-1)))
        var window = [Float](repeating: 0, count: winLength)
        vDSP_hann_window(&window, vDSP_Length(winLength), Int32(vDSP_HANN_NORM))
        self.hannWindow = window

        // FFT setup
        guard let setup = vDSP.FFT(log2n: self.log2n, radix: .radix2, ofType: DSPSplitComplex.self) else {
            fatalError("Failed to create FFT setup for log2n=\(self.log2n)")
        }
        self.fftSetup = setup

        // Build Kaldi-scale mel filterbank
        self.melFilterbank = Self.buildKaldiMelFilterbank(
            numMelBins: numMelBins,
            numFreqBins: nFFT / 2 + 1,
            sampleRate: Float(Self.sampleRate),
            fMin: Self.melFMin,
            fMax: Self.melFMax
        )
    }

    // MARK: - Public API

    /// Extract log mel spectrogram features from raw audio samples.
    ///
    /// - Parameters:
    ///   - audioSamples: Raw audio at 16kHz, mono, Float32
    ///   - maxMelLength: If provided, pad or truncate to this length.
    ///                   Must be 3000 (the CoreML model's fixed input shape).
    ///
    /// - Returns: A tuple of (features, attentionMask) as MLMultiArrays.
    ///   - features: shape [1, T, 128] where T = number of mel frames (or maxMelLength if specified)
    ///   - attentionMask: shape [1, T] where 1 = valid frame, 0 = padding
    func extractFeatures(
        from audioSamples: [Float],
        maxMelLength: Int? = nil
    ) -> (features: MLMultiArray, attentionMask: MLMultiArray) {
        // Compute STFT
        let powerSpectrum = computeSTFT(audioSamples)
        let numFrames = powerSpectrum.count / numFreqBins

        // Apply mel filterbank: [numFrames x numFreqBins] @ [numFreqBins x numMelBins]^T
        // = [numFrames x numMelBins]
        var melSpectrogram = applyMelFilterbank(powerSpectrum, numFrames: numFrames)

        // Clamp and log: log(max(x, 1e-5))
        let clampMin: Float = 1e-5
        let count = melSpectrogram.count
        for i in 0..<count {
            melSpectrogram[i] = log(max(melSpectrogram[i], clampMin))
        }

        // Determine actual frame count and target length
        let actualFrames = numFrames
        let targetLength = maxMelLength ?? actualFrames

        // Create MLMultiArrays
        let features = try! MLMultiArray(shape: [1, NSNumber(value: targetLength), NSNumber(value: Self.numMelBins)], dataType: .float32)
        let attentionMask = try! MLMultiArray(shape: [1, NSNumber(value: targetLength)], dataType: .float32)

        // Fill features
        let featurePtr = UnsafeMutablePointer<Float>(OpaquePointer(features.dataPointer))
        let maskPtr = UnsafeMutablePointer<Float>(OpaquePointer(attentionMask.dataPointer))

        let framesToCopy = min(actualFrames, targetLength)
        for t in 0..<framesToCopy {
            for m in 0..<Self.numMelBins {
                featurePtr[t * Self.numMelBins + m] = melSpectrogram[t * Self.numMelBins + m]
            }
            maskPtr[t] = 1.0
        }

        // Zero-pad remaining frames
        for t in framesToCopy..<targetLength {
            for m in 0..<Self.numMelBins {
                featurePtr[t * Self.numMelBins + m] = 0.0
            }
            maskPtr[t] = 0.0
        }

        return (features, attentionMask)
    }

    /// Compute the number of mel frames for a given number of audio samples.
    ///
    /// Matches the LasrFeatureExtractor's attention mask formula:
    ///     attention_mask[:, win_length - 1 :: hop_length]
    static func melFrameCount(for sampleCount: Int) -> Int {
        guard sampleCount >= winLength else { return 0 }
        return (sampleCount - winLength) / hopLength + 1
    }

    /// The fixed mel frame length expected by the CoreML model.
    /// All inputs must be padded to this length.
    static let modelInputLength: Int = 3000

    // MARK: - STFT

    /// Compute the Short-Time Fourier Transform and return the power spectrum.
    ///
    /// Returns a flat array of shape [numFrames, numFreqBins] in row-major order,
    /// where each element is |FFT[k]|^2 (power spectrum).
    private func computeSTFT(_ audio: [Float]) -> [Float] {
        let nFFT = Self.nFFT
        let hopLength = Self.hopLength
        let winLength = Self.winLength

        // Number of frames (non-centered STFT)
        let numFrames = max(0, (audio.count - winLength) / hopLength + 1)

        if numFrames == 0 {
            return []
        }

        // Allocate output
        var powerSpectrum = [Float](repeating: 0, count: numFrames * numFreqBins)

        // Temporary buffers for FFT (nFFT elements, zero-padded)
        var windowedFrame = [Float](repeating: 0, count: nFFT)
        var realPart = [Float](repeating: 0, count: nFFT / 2)
        var imagPart = [Float](repeating: 0, count: nFFT / 2)

        for frame in 0..<numFrames {
            let start = frame * hopLength

            // Zero the buffer
            vDSP.fill(&windowedFrame, with: 0)

            // Copy and window the frame
            for i in 0..<winLength {
                windowedFrame[i] = audio[start + i] * hannWindow[i]
            }
            // Remaining elements are already zero (zero-padding from winLength to nFFT)

            // Pack into split complex for vDSP FFT
            // vDSP_ctoz expects interleaved format, but we can use vDSP_fft_zrip
            // which takes real input packed as split complex
            windowedFrame.withUnsafeBufferPointer { bufPtr in
                bufPtr.baseAddress!.withMemoryRebound(to: DSPComplex.self, capacity: nFFT / 2) { complexPtr in
                    realPart.withUnsafeMutableBufferPointer { realBuf in
                        imagPart.withUnsafeMutableBufferPointer { imagBuf in
                            var splitComplex = DSPSplitComplex(
                                realp: realBuf.baseAddress!,
                                imagp: imagBuf.baseAddress!
                            )
                            vDSP_ctoz(complexPtr, 2, &splitComplex, 1, vDSP_Length(nFFT / 2))

                            // Forward FFT (in-place)
                            fftSetup.forward(input: splitComplex, output: &splitComplex)

                            // Compute power spectrum: |X[k]|^2 = real^2 + imag^2
                            // DC component (k=0): stored in realp[0], imagp[0] is Nyquist
                            let dc = splitComplex.realp[0]
                            let nyquist = splitComplex.imagp[0]

                            let offset = frame * numFreqBins
                            powerSpectrum[offset] = dc * dc  // DC bin

                            // Bins 1 to nFFT/2 - 1
                            for k in 1..<(nFFT / 2) {
                                let re = splitComplex.realp[k]
                                let im = splitComplex.imagp[k]
                                powerSpectrum[offset + k] = re * re + im * im
                            }

                            // Nyquist bin
                            powerSpectrum[offset + nFFT / 2] = nyquist * nyquist
                        }
                    }
                }
            }
        }

        return powerSpectrum
    }

    // MARK: - Mel Filterbank

    /// Apply the mel filterbank to the power spectrum.
    ///
    /// Input: power spectrum [numFrames x numFreqBins], flat row-major
    /// Output: mel spectrogram [numFrames x numMelBins], flat row-major
    private func applyMelFilterbank(_ powerSpectrum: [Float], numFrames: Int) -> [Float] {
        let numMelBins = Self.numMelBins

        // Matrix multiply: [numFrames x numFreqBins] @ [numFreqBins x numMelBins]
        // melFilterbank is stored as [numMelBins x numFreqBins] (row-major),
        // so we need to transpose it: use it as [numFreqBins x numMelBins] column-major
        // which is the same memory layout.
        //
        // Using vDSP_mmul: C = A * B
        //   A = powerSpectrum [numFrames x numFreqBins]
        //   B = melFilterbank transposed [numFreqBins x numMelBins]
        //   C = result [numFrames x numMelBins]

        var result = [Float](repeating: 0, count: numFrames * numMelBins)

        // vDSP_mmul(A, strideA, B, strideB, C, strideC, M, N, P)
        // C[M x N] = A[M x P] * B[P x N]
        // M = numFrames, N = numMelBins, P = numFreqBins
        vDSP_mmul(
            powerSpectrum, 1,     // A: [numFrames x numFreqBins]
            melFilterbank, 1,     // B: [numFreqBins x numMelBins] (transposed from [numMelBins x numFreqBins])
            &result, 1,           // C: [numFrames x numMelBins]
            vDSP_Length(numFrames),
            vDSP_Length(numMelBins),
            vDSP_Length(numFreqBins)
        )

        return result
    }

    /// Build a Kaldi-scale mel filterbank.
    ///
    /// Kaldi uses a different mel scale than HTK:
    ///   mel_kaldi(f) = 1127 * ln(1 + f/700)
    /// This is actually the same as the Slaney/HTK formula, but Kaldi
    /// uses natural log where HTK uses log10. The result is the same
    /// mel filterbank shape.
    ///
    /// Returns flat array [numMelBins x numFreqBins] in row-major order,
    /// but stored so that when treated as [numFreqBins x numMelBins] in
    /// column-major order, it gives the correct transpose for vDSP_mmul.
    private static func buildKaldiMelFilterbank(
        numMelBins: Int,
        numFreqBins: Int,
        sampleRate: Float,
        fMin: Float,
        fMax: Float
    ) -> [Float] {
        // Convert Hz to mel (Kaldi scale)
        func hzToMel(_ hz: Float) -> Float {
            return 1127.0 * log(1.0 + hz / 700.0)
        }
        func melToHz(_ mel: Float) -> Float {
            return 700.0 * (exp(mel / 1127.0) - 1.0)
        }

        let melMin = hzToMel(fMin)
        let melMax = hzToMel(fMax)

        // numMelBins + 2 points (including edges)
        let numPoints = numMelBins + 2
        var melPoints = [Float](repeating: 0, count: numPoints)
        for i in 0..<numPoints {
            melPoints[i] = melMin + Float(i) * (melMax - melMin) / Float(numPoints - 1)
        }

        // Convert mel points back to Hz, then to FFT bin indices
        let hzPoints = melPoints.map { melToHz($0) }
        let binFreqStep = sampleRate / Float(2 * (numFreqBins - 1))
        let fftBins = hzPoints.map { $0 / binFreqStep }

        // Build triangular filters
        // Store as [numFreqBins x numMelBins] for direct use in vDSP_mmul as B
        var filterbank = [Float](repeating: 0, count: numFreqBins * numMelBins)

        for m in 0..<numMelBins {
            let left = fftBins[m]
            let center = fftBins[m + 1]
            let right = fftBins[m + 2]

            for k in 0..<numFreqBins {
                let freq = Float(k)
                var weight: Float = 0.0

                if freq > left && freq <= center && center > left {
                    weight = (freq - left) / (center - left)
                } else if freq > center && freq < right && right > center {
                    weight = (right - freq) / (right - center)
                }

                // Store as [numFreqBins x numMelBins] row-major
                // so filterbank[k * numMelBins + m] for frequency bin k, mel bin m
                filterbank[k * numMelBins + m] = weight
            }
        }

        return filterbank
    }
}
