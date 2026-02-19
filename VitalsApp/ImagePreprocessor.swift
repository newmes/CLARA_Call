import CoreML
import CoreGraphics

enum ImagePreprocessor {

    static let imageSize = 448

    static func preprocess(_ image: CGImage) throws -> MLMultiArray {
        let width = imageSize
        let height = imageSize

        guard let context = CGContext(
            data: nil,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: width * 4,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue
        ) else {
            throw PreprocessorError.contextCreationFailed
        }

        context.interpolationQuality = .high
        context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))

        guard let data = context.data else {
            throw PreprocessorError.noPixelData
        }

        let pixelCount = width * height
        let ptr = data.bindMemory(to: UInt8.self, capacity: pixelCount * 4)

        let array = try MLMultiArray(
            shape: [1, 3, NSNumber(value: height), NSNumber(value: width)],
            dataType: .float32
        )
        let outPtr = UnsafeMutablePointer<Float>(OpaquePointer(array.dataPointer))
        let channelStride = pixelCount

        for i in 0..<pixelCount {
            let baseIn = i * 4
            let r = Float(ptr[baseIn])     / 255.0 * 2.0 - 1.0
            let g = Float(ptr[baseIn + 1]) / 255.0 * 2.0 - 1.0
            let b = Float(ptr[baseIn + 2]) / 255.0 * 2.0 - 1.0

            outPtr[0 * channelStride + i] = r
            outPtr[1 * channelStride + i] = g
            outPtr[2 * channelStride + i] = b
        }

        return array
    }

    enum PreprocessorError: LocalizedError {
        case contextCreationFailed
        case noPixelData

        var errorDescription: String? {
            switch self {
            case .contextCreationFailed: return "Failed to create CGContext for image resize"
            case .noPixelData: return "No pixel data in CGContext"
            }
        }
    }
}
