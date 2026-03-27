import AVFoundation
import Dispatch
import Foundation
@preconcurrency import MLX
@preconcurrency import MLXLMCommon
import MLXAudioCore
import MLXAudioSTT
import MLXNN
import Tokenizers

private final class Qwen3BridgeResponseBox: @unchecked Sendable {
    let textPtr: UnsafeMutablePointer<CChar>?
    let errorMessagePtr: UnsafeMutablePointer<CChar>?

    init(text: String? = nil, errorMessage: String? = nil) {
        self.textPtr = text.flatMap(duplicateCString)
        self.errorMessagePtr = errorMessage.flatMap(duplicateCString)
    }

    deinit {
        if let textPtr {
            free(textPtr)
        }
        if let errorMessagePtr {
            free(errorMessagePtr)
        }
    }
}

private final class Qwen3BridgeStore: @unchecked Sendable {
    static let shared = Qwen3BridgeStore()

    private let lock = NSLock()
    private var model: Qwen3ASRModel?
    private var repoID: String?
    private var modelDir: String?

    func withLock<T>(_ body: () throws -> T) rethrows -> T {
        lock.lock()
        defer { lock.unlock() }
        return try body()
    }

    func currentModel() -> Qwen3ASRModel? {
        model
    }

    func setModel(_ model: Qwen3ASRModel?, repoID: String?, modelDir: String?) {
        self.model = model
        self.repoID = repoID
        self.modelDir = modelDir
    }

    func matches(repoID: String, modelDir: String) -> Bool {
        self.repoID == repoID && self.modelDir == modelDir && model != nil
    }
}

private func duplicateCString(_ text: String) -> UnsafeMutablePointer<CChar>? {
    text.withCString { basePointer in
        guard let duplicated = strdup(basePointer) else {
            return nil
        }
        return duplicated
    }
}

private func makeSuccessResponse(_ text: String) -> UnsafeMutableRawPointer {
    Unmanaged.passRetained(Qwen3BridgeResponseBox(text: text)).toOpaque()
}

private func makeErrorResponse(_ message: String) -> UnsafeMutableRawPointer {
    Unmanaged.passRetained(Qwen3BridgeResponseBox(errorMessage: message)).toOpaque()
}

private func responseBox(
    from pointer: UnsafeRawPointer?
) -> Qwen3BridgeResponseBox? {
    guard let pointer else { return nil }
    return Unmanaged<Qwen3BridgeResponseBox>
        .fromOpaque(UnsafeMutableRawPointer(mutating: pointer))
        .takeUnretainedValue()
}

private func requiredString(
    _ pointer: UnsafePointer<CChar>?,
    fieldName: String
) throws -> String {
    guard let pointer else {
        throw NSError(
            domain: "HandyQwen3ASRBridge",
            code: 1,
            userInfo: [NSLocalizedDescriptionKey: "\(fieldName) is required"]
        )
    }
    return String(cString: pointer)
}

private func normalizedLanguage(_ pointer: UnsafePointer<CChar>?) -> String? {
    guard let pointer else { return nil }
    let value = String(cString: pointer).trimmingCharacters(in: .whitespacesAndNewlines)
    guard !value.isEmpty else { return nil }

    switch value.lowercased() {
    case "zh", "zh-hans", "zh-hant":
        return "Chinese"
    case "yue":
        return "Cantonese"
    case "en":
        return "English"
    case "ja":
        return "Japanese"
    case "ko":
        return "Korean"
    case "es":
        return "Spanish"
    case "fr":
        return "French"
    case "de":
        return "German"
    case "it":
        return "Italian"
    case "pt":
        return "Portuguese"
    case "ru":
        return "Russian"
    case "ar":
        return "Arabic"
    case "hi":
        return "Hindi"
    case "th":
        return "Thai"
    case "vi":
        return "Vietnamese"
    case "tr":
        return "Turkish"
    case "pl":
        return "Polish"
    case "nl":
        return "Dutch"
    case "sv":
        return "Swedish"
    case "da":
        return "Danish"
    case "fi":
        return "Finnish"
    case "cs":
        return "Czech"
    case "el":
        return "Greek"
    case "ro":
        return "Romanian"
    case "hu":
        return "Hungarian"
    default:
        return value
    }
}

private func generateTokenizerJSONIfMissing(in modelDir: URL) throws {
    let tokenizerJSONPath = modelDir.appendingPathComponent("tokenizer.json")
    guard !FileManager.default.fileExists(atPath: tokenizerJSONPath.path) else { return }

    let vocabURL = modelDir.appendingPathComponent("vocab.json")
    let mergesURL = modelDir.appendingPathComponent("merges.txt")
    let tokenizerConfigURL = modelDir.appendingPathComponent("tokenizer_config.json")

    guard FileManager.default.fileExists(atPath: vocabURL.path),
          FileManager.default.fileExists(atPath: mergesURL.path) else {
        return
    }

    let vocabData = try Data(contentsOf: vocabURL)
    let mergesText = try String(contentsOf: mergesURL, encoding: .utf8)
    let mergeLines = mergesText.components(separatedBy: "\n")
        .filter { !$0.hasPrefix("#") && !$0.isEmpty }

    let mergesJSON = mergeLines.map { line -> String in
        let escaped = line
            .replacingOccurrences(of: "\\", with: "\\\\")
            .replacingOccurrences(of: "\"", with: "\\\"")
        return "\"\(escaped)\""
    }.joined(separator: ",")

    var addedTokensJSON = "[]"
    if FileManager.default.fileExists(atPath: tokenizerConfigURL.path) {
        let configData = try Data(contentsOf: tokenizerConfigURL)
        if let configDict = try JSONSerialization.jsonObject(with: configData) as? [String: Any],
           let addedTokensDecoder = configDict["added_tokens_decoder"] as? [String: Any] {
            var tokens: [(Int, [String: Any])] = []
            for (idStr, value) in addedTokensDecoder {
                if let id = Int(idStr), let tokenDict = value as? [String: Any] {
                    let entry: [String: Any] = [
                        "id": id,
                        "content": tokenDict["content"] ?? "",
                        "single_word": tokenDict["single_word"] ?? false,
                        "lstrip": tokenDict["lstrip"] ?? false,
                        "rstrip": tokenDict["rstrip"] ?? false,
                        "normalized": tokenDict["normalized"] ?? false,
                        "special": tokenDict["special"] ?? false,
                    ]
                    tokens.append((id, entry))
                }
            }
            tokens.sort { $0.0 < $1.0 }
            let tokenData = try JSONSerialization.data(
                withJSONObject: tokens.map { $0.1 },
                options: []
            )
            addedTokensJSON = String(data: tokenData, encoding: .utf8) ?? "[]"
        }
    }

    let preTokenizerPattern = "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"
    let escapedPattern = preTokenizerPattern
        .replacingOccurrences(of: "\\", with: "\\\\")
        .replacingOccurrences(of: "\"", with: "\\\"")
    let vocabString = String(data: vocabData, encoding: .utf8) ?? "{}"

    let tokenizerJSON = """
    {
      "version": "1.0",
      "truncation": null,
      "padding": null,
      "added_tokens": \(addedTokensJSON),
      "normalizer": {"type": "NFC"},
      "pre_tokenizer": {
        "type": "Sequence",
        "pretokenizers": [
          {
            "type": "Split",
            "pattern": {"Regex": "\(escapedPattern)"},
            "behavior": "Isolated",
            "invert": false
          },
          {
            "type": "ByteLevel",
            "add_prefix_space": false,
            "trim_offsets": true,
            "use_regex": false
          }
        ]
      },
      "post_processor": null,
      "decoder": {
        "type": "ByteLevel",
        "add_prefix_space": true,
        "trim_offsets": true,
        "use_regex": true
      },
      "model": {
        "type": "BPE",
        "dropout": null,
        "unk_token": null,
        "continuing_subword_prefix": "",
        "end_of_word_suffix": "",
        "fuse_unk": false,
        "byte_fallback": false,
        "vocab": \(vocabString),
        "merges": [\(mergesJSON)]
      }
    }
    """

    try tokenizerJSON.write(to: tokenizerJSONPath, atomically: true, encoding: .utf8)
}

private func loadModelFromLocalDirectory(_ modelURL: URL) async throws -> Qwen3ASRModel {
    let configPath = modelURL.appendingPathComponent("config.json")
    let configData = try Data(contentsOf: configPath)
    let config = try JSONDecoder().decode(Qwen3ASRConfig.self, from: configData)
    let perLayerQuantization = config.perLayerQuantization

    let model = Qwen3ASRModel(config)

    try generateTokenizerJSONIfMissing(in: modelURL)
    model.tokenizer = try await AutoTokenizer.from(modelFolder: modelURL)

    let files = try FileManager.default.contentsOfDirectory(
        at: modelURL,
        includingPropertiesForKeys: nil
    )
    let safetensorFiles = files.filter { $0.pathExtension == "safetensors" }
    if safetensorFiles.isEmpty {
        throw NSError(
            domain: "HandyQwen3ASRBridge",
            code: 2,
            userInfo: [NSLocalizedDescriptionKey: "No safetensors files found in \(modelURL.path)"]
        )
    }

    var weights: [String: MLXArray] = [:]
    for file in safetensorFiles {
        let fileWeights = try MLX.loadArrays(url: file)
        weights.merge(fileWeights) { _, new in new }
    }

    let skipLmHead = config.textConfig.tieWordEmbeddings
    let sanitizedWeights = Qwen3ASRModel.sanitize(weights: weights, skipLmHead: skipLmHead)

    if perLayerQuantization != nil {
        quantize(model: model) { path, _ in
            if path.hasPrefix("audio_tower") {
                return nil
            }
            if sanitizedWeights["\(path).scales"] != nil {
                return perLayerQuantization?.quantization(layer: path)?.asTuple
            }
            return nil
        }
    }

    try model.update(parameters: ModuleParameters.unflattened(sanitizedWeights), verify: .all)
    eval(model)
    return model
}

private func warmupModel(_ model: Qwen3ASRModel) throws {
    let warmupAudio = MLXArray(Array(repeating: Float(0), count: 32_000))
    _ = try model.generate(audio: warmupAudio)
}

private func loadModelBlocking(repoID: String, localModelDir: String) throws -> Qwen3ASRModel {
    final class ResultBox: @unchecked Sendable {
        var model: Qwen3ASRModel?
        var error: String?
    }

    let semaphore = DispatchSemaphore(value: 0)
    let box = ResultBox()

    Task.detached(priority: .userInitiated) {
        defer { semaphore.signal() }
        do {
            let modelURL = URL(fileURLWithPath: localModelDir, isDirectory: true)
            let model = try await loadModelFromLocalDirectory(modelURL)
            try warmupModel(model)
            box.model = model
        } catch {
            box.error = error.localizedDescription
        }
    }

    semaphore.wait()

    if let model = box.model {
        return model
    }

    throw NSError(
        domain: "HandyQwen3ASRBridge",
        code: 3,
        userInfo: [NSLocalizedDescriptionKey: box.error ?? "Failed to load Qwen3ASR model"]
    )
}

private func transcribeWithLoadedModel(
    samples: UnsafePointer<Float>,
    sampleCount: Int,
    sampleRate: Int,
    language: String?
) throws -> String {
    if sampleCount == 0 {
        return ""
    }

    let audio: MLXArray

    if sampleRate == 16_000 {
        let buffer = UnsafeBufferPointer(start: samples, count: sampleCount)
        audio = MLXArray(buffer, [sampleCount])
    } else {
        let input = Array(UnsafeBufferPointer(start: samples, count: sampleCount))
        let prepared = try resampleAudio(input, from: sampleRate, to: 16_000)
        audio = MLXArray(prepared)
    }

    let output = try Qwen3BridgeStore.shared.withLock {
        guard let model = Qwen3BridgeStore.shared.currentModel() else {
            throw NSError(
                domain: "HandyQwen3ASRBridge",
                code: 4,
                userInfo: [NSLocalizedDescriptionKey: "Qwen3ASR model is not loaded"]
            )
        }

        if let language {
            return model.generate(audio: audio, language: language)
        } else {
            return model.generate(audio: audio)
        }
    }

    return output.text
}

@_cdecl("handy_qwen3asr_is_available")
public func handy_qwen3asr_is_available() -> Int32 {
    1
}

@_cdecl("handy_qwen3asr_load_model")
public func handy_qwen3asr_load_model(
    _ repoID: UnsafePointer<CChar>?,
    _ localModelDir: UnsafePointer<CChar>?
) -> UnsafeMutableRawPointer {
    do {
        let repoID = try requiredString(repoID, fieldName: "repoID")
        let localModelDir = try requiredString(localModelDir, fieldName: "localModelDir")
        let response = try Qwen3BridgeStore.shared.withLock {
            if Qwen3BridgeStore.shared.matches(repoID: repoID, modelDir: localModelDir) {
                return makeSuccessResponse("")
            }

            let model = try loadModelBlocking(repoID: repoID, localModelDir: localModelDir)
            Qwen3BridgeStore.shared.setModel(model, repoID: repoID, modelDir: localModelDir)
            return makeSuccessResponse("")
        }
        return response
    } catch {
        return makeErrorResponse(error.localizedDescription)
    }
}

@_cdecl("handy_qwen3asr_transcribe")
public func handy_qwen3asr_transcribe(
    _ samples: UnsafePointer<Float>?,
    _ sampleCount: Int,
    _ sampleRate: Int32,
    _ language: UnsafePointer<CChar>?
) -> UnsafeMutableRawPointer {
    do {
        guard let samples else {
            throw NSError(
                domain: "HandyQwen3ASRBridge",
                code: 5,
                userInfo: [NSLocalizedDescriptionKey: "Audio samples pointer is null"]
            )
        }

        let text = try transcribeWithLoadedModel(
            samples: samples,
            sampleCount: sampleCount,
            sampleRate: Int(sampleRate),
            language: normalizedLanguage(language)
        )
        return makeSuccessResponse(text)
    } catch {
        return makeErrorResponse(error.localizedDescription)
    }
}

@_cdecl("handy_qwen3asr_unload_model")
public func handy_qwen3asr_unload_model() {
    Qwen3BridgeStore.shared.withLock {
        Qwen3BridgeStore.shared.setModel(nil, repoID: nil, modelDir: nil)
    }
}

@_cdecl("handy_qwen3asr_response_success")
public func handy_qwen3asr_response_success(_ response: UnsafeRawPointer?) -> Int32 {
    guard let response = responseBox(from: response) else { return 0 }
    return response.errorMessagePtr == nil ? 1 : 0
}

@_cdecl("handy_qwen3asr_response_text")
public func handy_qwen3asr_response_text(_ response: UnsafeRawPointer?) -> UnsafePointer<CChar>? {
    guard let response = responseBox(from: response), let textPtr = response.textPtr else {
        return nil
    }
    return UnsafePointer(textPtr)
}

@_cdecl("handy_qwen3asr_response_error_message")
public func handy_qwen3asr_response_error_message(
    _ response: UnsafeRawPointer?
) -> UnsafePointer<CChar>? {
    guard let response = responseBox(from: response),
        let errorMessagePtr = response.errorMessagePtr
    else {
        return nil
    }
    return UnsafePointer(errorMessagePtr)
}

@_cdecl("handy_qwen3asr_free_response")
public func handy_qwen3asr_free_response(_ response: UnsafeMutableRawPointer?) {
    guard let response else { return }
    Unmanaged<Qwen3BridgeResponseBox>.fromOpaque(response).release()
}
