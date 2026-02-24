import Foundation

actor CareAIClient {

    // MARK: - Request

    struct ConsultRequest: Encodable {
        let siglip_vector: [Float]
        let patient_text: String
        let drug_name: String?
        let indication: String?
        let skip_tts: Bool
        let audio_b64: String?
    }

    // MARK: - Response

    struct ConsultResponse: Decodable {
        let nurse_text: String
        let nurse_structured: NurseStructured?
        let visual_assessment: VisualAssessment?
        let audio_base64: String?
        let latency_ms: LatencyInfo?
    }

    struct NurseStructured: Decodable {
        let approach_style: String?
        let acknowledgment: String?
        let questions: [NurseQuestion]?
        let visual_followup: String?
        let preliminary_concerns: [String]?
    }

    struct NurseQuestion: Decodable {
        let question: String?
        let target_ae: String?
        let rationale: String?
    }

    struct VisualAssessment: Decodable {
        let findings: [Finding]?
        let general_observations: [String]?
    }

    struct Finding: Decodable {
        let ae_term: String?
        let estimated_grade: Int?
        let confidence: Double?
        let description: String?
    }

    struct LatencyInfo: Decodable {
        let siglip_ms: Double?
        let nurse_ms: Double?
        let tts_ms: Double?
        let total_ms: Double?
    }

    // MARK: - Errors

    enum CareAIError: LocalizedError {
        case invalidURL
        case httpError(Int, String)
        case decodingFailed(String)

        var errorDescription: String? {
            switch self {
            case .invalidURL: return "Invalid Care AI URL"
            case .httpError(let code, let body): return "Care AI error \(code): \(body)"
            case .decodingFailed(let detail): return "Failed to decode Care AI response: \(detail)"
            }
        }
    }

    // MARK: - API

    func consult(
        embedding: [Float], patientText: String, baseURL: String,
        audioBase64: String? = nil
    ) async throws -> ConsultResponse {
        guard let url = URL(string: baseURL)?
            .appendingPathComponent("v1")
            .appendingPathComponent("consult") else {
            throw CareAIError.invalidURL
        }

        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.timeoutInterval = 90
        request.httpBody = try JSONEncoder().encode(
            ConsultRequest(
                siglip_vector: embedding,
                patient_text: patientText,
                drug_name: nil,
                indication: nil,
                skip_tts: false,
                audio_b64: audioBase64
            )
        )

        print("[CareAI] POST \(url.absoluteString) (text: \(patientText.prefix(60))...)")

        let (data, response) = try await URLSession.shared.data(for: request)

        if let http = response as? HTTPURLResponse, !(200..<300).contains(http.statusCode) {
            throw CareAIError.httpError(http.statusCode, String(data: data, encoding: .utf8) ?? "")
        }

        do {
            return try JSONDecoder().decode(ConsultResponse.self, from: data)
        } catch {
            throw CareAIError.decodingFailed(error.localizedDescription)
        }
    }
}
