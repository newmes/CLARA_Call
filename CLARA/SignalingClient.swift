import Foundation

enum SignalingMessage {
    case offer(sdp: String)
    case answer(sdp: String)
    case candidate(sdpMid: String?, sdpMLineIndex: Int32, candidate: String)
    case transcription(text: String, timestamp: Double)
    case privacyStart
    case embedding(values: [Float], timestamp: Double)
    case debugFrame(imageData: Data, embeddingValues: [Float], timestamp: Double)

    func jsonData() throws -> Data {
        var dict: [String: Any] = [:]
        switch self {
        case .offer(let sdp):
            dict["type"] = "offer"
            dict["sdp"] = sdp
        case .answer(let sdp):
            dict["type"] = "answer"
            dict["sdp"] = sdp
        case .candidate(let sdpMid, let sdpMLineIndex, let candidate):
            dict["type"] = "candidate"
            dict["sdpMid"] = sdpMid
            dict["sdpMLineIndex"] = sdpMLineIndex
            dict["candidate"] = candidate
        case .transcription(let text, let timestamp):
            dict["type"] = "transcription"
            dict["text"] = text
            dict["timestamp"] = timestamp
        case .privacyStart:
            dict["type"] = "privacy_start"
        case .embedding(let values, let timestamp):
            dict["type"] = "embedding"
            dict["values"] = values
            dict["timestamp"] = timestamp
        case .debugFrame(let imageData, let embeddingValues, let timestamp):
            dict["type"] = "debug_frame"
            dict["image"] = imageData.base64EncodedString()
            dict["embedding"] = embeddingValues
            dict["timestamp"] = timestamp
        }
        return try JSONSerialization.data(withJSONObject: dict)
    }

    static func from(jsonData data: Data) -> SignalingMessage? {
        guard let dict = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let type = dict["type"] as? String else {
            return nil
        }
        switch type {
        case "offer":
            guard let sdp = dict["sdp"] as? String else { return nil }
            return .offer(sdp: sdp)
        case "answer":
            guard let sdp = dict["sdp"] as? String else { return nil }
            return .answer(sdp: sdp)
        case "candidate":
            guard let candidate = dict["candidate"] as? String else { return nil }
            let sdpMid = dict["sdpMid"] as? String
            let sdpMLineIndex = dict["sdpMLineIndex"] as? Int32 ?? 0
            return .candidate(sdpMid: sdpMid, sdpMLineIndex: sdpMLineIndex, candidate: candidate)
        case "transcription":
            guard let text = dict["text"] as? String else { return nil }
            let timestamp = dict["timestamp"] as? Double ?? Date().timeIntervalSince1970
            return .transcription(text: text, timestamp: timestamp)
        case "privacy_start":
            return .privacyStart
        case "embedding":
            guard let values = dict["values"] as? [Float] else { return nil }
            let timestamp = dict["timestamp"] as? Double ?? Date().timeIntervalSince1970
            return .embedding(values: values, timestamp: timestamp)
        case "debug_frame":
            guard let imageB64 = dict["image"] as? String,
                  let imageData = Data(base64Encoded: imageB64),
                  let embedding = dict["embedding"] as? [Float] else { return nil }
            let timestamp = dict["timestamp"] as? Double ?? Date().timeIntervalSince1970
            return .debugFrame(imageData: imageData, embeddingValues: embedding, timestamp: timestamp)
        default:
            return nil
        }
    }
}

class SignalingClient: NSObject {
    var onMessage: ((SignalingMessage) -> Void)?
    var onConnect: (() -> Void)?
    var onDisconnect: ((Error?) -> Void)?

    private var webSocket: URLSessionWebSocketTask?
    private var urlSession: URLSession?

    func connect(to url: URL) {
        let session = URLSession(configuration: .default, delegate: self, delegateQueue: nil)
        urlSession = session
        let task = session.webSocketTask(with: url)
        webSocket = task
        task.resume()
    }

    func send(_ message: SignalingMessage) {
        guard let data = try? message.jsonData(),
              let string = String(data: data, encoding: .utf8) else { return }
        webSocket?.send(.string(string)) { error in
            if let error {
                print("[SignalingClient] send error: \(error)")
            }
        }
    }

    func disconnect() {
        webSocket?.cancel(with: .normalClosure, reason: nil)
        webSocket = nil
        urlSession?.invalidateAndCancel()
        urlSession = nil
    }

    private func readMessage() {
        webSocket?.receive { [weak self] result in
            switch result {
            case .success(let message):
                let data: Data?
                switch message {
                case .string(let text):
                    data = text.data(using: .utf8)
                case .data(let d):
                    data = d
                @unknown default:
                    data = nil
                }
                if let data, let signalingMessage = SignalingMessage.from(jsonData: data) {
                    self?.onMessage?(signalingMessage)
                }
                self?.readMessage()
            case .failure(let error):
                print("[SignalingClient] receive error: \(error)")
                self?.onDisconnect?(error)
            }
        }
    }
}

extension SignalingClient: URLSessionWebSocketDelegate {
    func urlSession(_ session: URLSession, webSocketTask: URLSessionWebSocketTask,
                    didOpenWithProtocol protocol: String?) {
        print("[SignalingClient] connected")
        onConnect?()
        readMessage()
    }

    func urlSession(_ session: URLSession, webSocketTask: URLSessionWebSocketTask,
                    didCloseWith closeCode: URLSessionWebSocketTask.CloseCode, reason: Data?) {
        print("[SignalingClient] disconnected: \(closeCode)")
        onDisconnect?(nil)
    }
}
