import Foundation

struct ChatMessage: Identifiable, Equatable {
    let id: UUID
    let text: String
    let timestamp: Date
    let isFromServer: Bool

    init(id: UUID = UUID(), text: String, timestamp: Date = Date(), isFromServer: Bool = true) {
        self.id = id
        self.text = text
        self.timestamp = timestamp
        self.isFromServer = isFromServer
    }
}
