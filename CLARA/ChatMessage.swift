import Foundation

struct ChatMessage: Identifiable, Equatable {
    let id: UUID
    let text: String
    let timestamp: Date
    let isFromServer: Bool
    let audioData: Data?

    init(id: UUID = UUID(), text: String, timestamp: Date = Date(), isFromServer: Bool = true, audioData: Data? = nil) {
        self.id = id
        self.text = text
        self.timestamp = timestamp
        self.isFromServer = isFromServer
        self.audioData = audioData
    }
}
