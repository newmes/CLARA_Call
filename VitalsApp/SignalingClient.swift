import Foundation

enum SignalingMessage {
    case offer(sdp: String)
    case answer(sdp: String)
    case candidate(sdpMid: String?, sdpMLineIndex: Int32, candidate: String)
    case transcription(text: String, timestamp: Double)

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
    private var pingTimer: Timer?

    func connect(to url: URL) {
        let session = URLSession(configuration: .default, delegate: self, delegateQueue: nil)
        urlSession = session
        var request = URLRequest(url: url)
        request.addValue("1", forHTTPHeaderField: "ngrok-skip-browser-warning")
        request.addValue("websocket", forHTTPHeaderField: "Upgrade")
        let task = session.webSocketTask(with: request)
        webSocket = task
        task.resume()
    }

    private func startPing() {
        pingTimer?.invalidate()
        pingTimer = Timer.scheduledTimer(withTimeInterval: 10, repeats: true) { [weak self] _ in
            self?.webSocket?.sendPing { error in
                if let error {
                    print("[SignalingClient] ping error: \(error)")
                }
            }
        }
    }

    private func stopPing() {
        pingTimer?.invalidate()
        pingTimer = nil
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
        stopPing()
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
                    print("[SignalingClient] raw WS received: \(text)")
                    data = text.data(using: .utf8)
                case .data(let d):
                    print("[SignalingClient] raw WS received \(d.count) bytes")
                    data = d
                @unknown default:
                    data = nil
                }
                if let data, let signalingMessage = SignalingMessage.from(jsonData: data) {
                    print("[SignalingClient] parsed message: \(signalingMessage)")
                    self?.onMessage?(signalingMessage)
                } else {
                    print("[SignalingClient] failed to parse message")
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
        DispatchQueue.main.async { self.startPing() }
        onConnect?()
        readMessage()
    }

    func urlSession(_ session: URLSession, webSocketTask: URLSessionWebSocketTask,
                    didCloseWith closeCode: URLSessionWebSocketTask.CloseCode, reason: Data?) {
        print("[SignalingClient] disconnected: \(closeCode)")
        onDisconnect?(nil)
    }
}
