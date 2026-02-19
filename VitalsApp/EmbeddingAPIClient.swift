import Foundation

actor EmbeddingAPIClient {

    struct ClassifyRequest: Encodable {
        let embedding: [Float]
        let top_k: Int
    }

    struct ClassifyResponse: Decodable {
        let results: [Result]
        struct Result: Decodable {
            let label: String
            let score: Float
        }
    }

    enum APIError: LocalizedError {
        case invalidURL
        case httpError(Int, String)
        case decodingFailed(String)

        var errorDescription: String? {
            switch self {
            case .invalidURL: return "Invalid server URL"
            case .httpError(let code, let body): return "Server error \(code): \(body)"
            case .decodingFailed(let detail): return "Failed to decode response: \(detail)"
            }
        }
    }

    func classify(
        embedding: [Float], topK: Int = 5, serverURL: String
    ) async throws -> [(label: String, score: Float)] {
        guard let url = URL(string: serverURL)?
            .appendingPathComponent("api")
            .appendingPathComponent("classify") else {
            throw APIError.invalidURL
        }

        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.timeoutInterval = 30
        request.httpBody = try JSONEncoder().encode(ClassifyRequest(embedding: embedding, top_k: topK))

        let (data, response) = try await URLSession.shared.data(for: request)

        if let http = response as? HTTPURLResponse, !(200..<300).contains(http.statusCode) {
            throw APIError.httpError(http.statusCode, String(data: data, encoding: .utf8) ?? "")
        }

        let decoded = try JSONDecoder().decode(ClassifyResponse.self, from: data)
        return decoded.results.map { ($0.label, $0.score) }
    }
}
