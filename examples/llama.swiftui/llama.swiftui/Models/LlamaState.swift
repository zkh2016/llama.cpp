import Foundation
import CoreML
import SwiftUI

var sharedQueue: [String] = []
let queueCapacity = 7 * 8
var start_time: UInt64 = 0
var total_time = 0.0
var total_cnt = 0

struct Model: Identifiable {
    var id = UUID()
    var name: String
    var url: String
    var filename: String
    var status: String?
}

class ConditionalChatTTS: ObservableObject {
    @Published var messageLog = ""
    private var TTSContext: LlamaContext?
    let NS_PER_S = 1_000_000_000.0

    init(){
        //self.messageLog = messageLog
        var ttsUrl: URL? {
            Bundle.main.url(forResource: "tts-ggml-model-Q8_0", withExtension: "gguf", subdirectory: "models")
        }

        if let ttsUrl {
            self.messageLog += "Loading tts model...\n"
            do {
                TTSContext = try LlamaContext.create_context(path: ttsUrl.path(), use_metal: false)
            }catch {
                self.messageLog += "Loaded tts model \(ttsUrl.lastPathComponent) failed\n"
            }
            self.messageLog += "Loaded tts model \(ttsUrl.lastPathComponent)\n"

            // Assuming that the model is successfully loaded, update the downloaded models
        } else {
            self.messageLog += "tts model not found: \(ttsUrl?.path())\n"
        }
    }
    func forward(){
        
    }
    func prepare_inputs_embeds(){
        
    }
    func prefill_text(
        input_ids:[Int],
        position_ids: [Int]
    ){
        
    }
    func generate(){
    
    }
    
    func consumer() {
        guard let TTSContext else {
            return
        }
        
        var generate_str = ""
        var generate_cnt = 0
        Task.detached {[weak self] in
            guard let self = self else { return }
            while true{
                while sharedQueue.isEmpty {
                    await Task.yield()
                }
                let input_str = sharedQueue.removeFirst()
                let t_start = DispatchTime.now().uptimeNanoseconds
                await TTSContext.completion_init(text: input_str)
                
                while await !TTSContext.is_done {
                    let result = await TTSContext.completion_loop(no_stop:true)
                    generate_str += result
                    generate_cnt += 1
                    if(generate_cnt == 50){
                        total_cnt += 7
                        let t_end = DispatchTime.now().uptimeNanoseconds
                        total_time = Double(t_end - start_time) / NS_PER_S
                        let tts_time = Double(t_end - t_start) / NS_PER_S
                        print(
                            """
                                \(generate_str) \n
                                buffer.cnt = \(sharedQueue.count)\n
                                tts Done
                                llm Generated \(total_cnt) t\n
                                tts_time \(tts_time)\n
                                total_time \(total_time)s
                            """
                        )
                        generate_cnt = 0
                        generate_str = ""
                        break
                    }
                }
            }
            
        }
    }
    
    func only_generate_by_str(text: String) async -> String {
        guard let TTSContext else {
            return ""
        }
        var input_str = text
        await TTSContext.completion_init(text: input_str)
        var generate_str = ""
        var generate_cnt = 0
        while await !TTSContext.is_done {
            let result = await TTSContext.completion_loop(no_stop:true)
            generate_str += result
            generate_cnt += 1
            if(generate_cnt == 50){
                break
            }
        }
        return generate_str
    }
    
    func generate_by_str(text: String, stream: Bool = true) async {
        guard let TTSContext else {
            return
        }
        self.messageLog += text
        var input_str = "[Sbreak]\(text)[Pbreak][oral_2][laugh_0][break_4]"
        var generate_cnt = 0
        var generate_str = ""
        
        let t_start = DispatchTime.now().uptimeNanoseconds
        await TTSContext.completion_init(text: input_str)
        let t_heat_end = DispatchTime.now().uptimeNanoseconds
        let t_heat = Double(t_heat_end - t_start) / NS_PER_S
        
        while await !TTSContext.is_done {
            let result = await TTSContext.completion_loop()
            if(stream){
                await MainActor.run {
                    self.messageLog += "\(result)"
                    generate_cnt += 1
                }
            }else{
                generate_str += result
                generate_cnt += 1
            }
        }
            
        let t_end = DispatchTime.now().uptimeNanoseconds
        let t_generation = Double(t_end - t_heat_end) / self.NS_PER_S
        let tokens_per_second = Double(generate_cnt) / t_generation
        
        await TTSContext.clear()
        
        if(!stream){
            self.messageLog += generate_str
        }
        await MainActor.run {
            self.messageLog += """
                \n
                Done
                Heat up took \(t_heat)s
                Generated \(generate_cnt)t/\(t_generation)s, \(tokens_per_second) t/s\n\n
                """
        }
    }
    
    func decode_to_mel_specs(){
        
    }
}

@MainActor
class LlamaState: ObservableObject {
    @Published var messageLog = ""
    @Published var cacheCleared = false
    @Published var downloadedModels: [Model] = []
    @Published var undownloadedModels: [Model] = []
    let NS_PER_S = 1_000_000_000.0
    var model: MLModel?

    private var llamaContext: LlamaContext?
//    private var TTSContext: LlamaContext?
    //var tts : ConditionalChatTTS?

    private var defaultModelUrl: URL? {
       // Bundle.main.url(forResource: "Model-4.1B-Q4_0", withExtension: "gguf", subdirectory: "models")
        Bundle.main.url(forResource: "qwen2.5-7b-instruct-q4_0", withExtension: "gguf", subdirectory: "models")
        // Bundle.main.url(forResource: "llama-2-7b-chat", withExtension: "Q2_K.gguf", subdirectory: "models")
    }
//    private var defaultTTSModelUrl: URL? {
//        Bundle.main.url(forResource: "tts-ggml-model-Q4_0", withExtension: "gguf", subdirectory: "models")
//        // Bundle.main.url(forResource: "llama-2-7b-chat", withExtension: "Q2_K.gguf", subdirectory: "models")
//    }

    init() {
//        guard let modelURL = Bundle.main.url(forResource: "ane_minicpmo_audio_f32_b1", withExtension: "mlmodelc", subdirectory: "models"),
//              let coreMLModel = try? MLModel(contentsOf: modelURL) else {
//                    fatalError("Failed to load model")
//              }
//        self.model = coreMLModel
//        print("load audio model success")
        
        loadModelsFromDisk()
        loadDefaultModels()
        //tts = ConditionalChatTTS()
    }

    private func loadModelsFromDisk() {
        do {
            let documentsURL = getDocumentsDirectory()
            let modelURLs = try FileManager.default.contentsOfDirectory(at: documentsURL, includingPropertiesForKeys: nil, options: [.skipsHiddenFiles, .skipsSubdirectoryDescendants])
            for modelURL in modelURLs {
                let modelName = modelURL.deletingPathExtension().lastPathComponent
                downloadedModels.append(Model(name: modelName, url: "", filename: modelURL.lastPathComponent, status: "downloaded"))
            }
        } catch {
            print("Error loading models from disk: \(error)")
        }
    }

    private func loadDefaultModels() {
        do {
            try loadModel(modelUrl: defaultModelUrl)
        } catch {
            messageLog += "Error!\n"
        }

        for model in defaultModels {
            let fileURL = getDocumentsDirectory().appendingPathComponent(model.filename)
            if FileManager.default.fileExists(atPath: fileURL.path) {

            } else {
                var undownloadedModel = model
                undownloadedModel.status = "download"
                undownloadedModels.append(undownloadedModel)
            }
        }
    }

    func getDocumentsDirectory() -> URL {
        let paths = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)
        return paths[0]
    }
    private let defaultModels: [Model] = [
        Model(name: "TinyLlama-1.1B (Q4_0, 0.6 GiB)",url: "https://huggingface.co/TheBloke/TinyLlama-1.1B-1T-OpenOrca-GGUF/resolve/main/tinyllama-1.1b-1t-openorca.Q4_0.gguf?download=true",filename: "tinyllama-1.1b-1t-openorca.Q4_0.gguf", status: "download"),
        Model(
            name: "TinyLlama-1.1B Chat (Q8_0, 1.1 GiB)",
            url: "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q8_0.gguf?download=true",
            filename: "tinyllama-1.1b-chat-v1.0.Q8_0.gguf", status: "download"
        ),

        Model(
            name: "TinyLlama-1.1B (F16, 2.2 GiB)",
            url: "https://huggingface.co/ggml-org/models/resolve/main/tinyllama-1.1b/ggml-model-f16.gguf?download=true",
            filename: "tinyllama-1.1b-f16.gguf", status: "download"
        ),

        Model(
            name: "Phi-2.7B (Q4_0, 1.6 GiB)",
            url: "https://huggingface.co/ggml-org/models/resolve/main/phi-2/ggml-model-q4_0.gguf?download=true",
            filename: "phi-2-q4_0.gguf", status: "download"
        ),

        Model(
            name: "Phi-2.7B (Q8_0, 2.8 GiB)",
            url: "https://huggingface.co/ggml-org/models/resolve/main/phi-2/ggml-model-q8_0.gguf?download=true",
            filename: "phi-2-q8_0.gguf", status: "download"
        ),

        Model(
            name: "Mistral-7B-v0.1 (Q4_0, 3.8 GiB)",
            url: "https://huggingface.co/TheBloke/Mistral-7B-v0.1-GGUF/resolve/main/mistral-7b-v0.1.Q4_0.gguf?download=true",
            filename: "mistral-7b-v0.1.Q4_0.gguf", status: "download"
        ),
        Model(
            name: "OpenHermes-2.5-Mistral-7B (Q3_K_M, 3.52 GiB)",
            url: "https://huggingface.co/TheBloke/OpenHermes-2.5-Mistral-7B-GGUF/resolve/main/openhermes-2.5-mistral-7b.Q3_K_M.gguf?download=true",
            filename: "openhermes-2.5-mistral-7b.Q3_K_M.gguf", status: "download"
        )
    ]
    func loadModel(modelUrl: URL?) throws {
        if let modelUrl {
            messageLog += "Loading model...\n"
            llamaContext = try LlamaContext.create_context(path: modelUrl.path())
            
            messageLog += "Loaded model \(modelUrl.lastPathComponent)\n"

            // Assuming that the model is successfully loaded, update the downloaded models
            updateDownloadedModels(modelName: modelUrl.lastPathComponent, status: "downloaded")
        } else {
            messageLog += "Load a model from the list below\n"
        }
        
//        if let ttsUrl {
//            messageLog += "Loading tts model...\n"
//            TTSContext = try LlamaContext.create_context(path: ttsUrl.path())
//            messageLog += "Loaded tts model \(ttsUrl.lastPathComponent)\n"
//
//            // Assuming that the model is successfully loaded, update the downloaded models
//            updateDownloadedModels(modelName: ttsUrl.lastPathComponent, status: "downloaded")
//        } else {
//            messageLog += "Load a model from the list below\n"
//        }
    }


    private func updateDownloadedModels(modelName: String, status: String) {
        undownloadedModels.removeAll { $0.name == modelName }
    }

    func complete(text: String, stream: Bool=true) async -> String {
        guard let llamaContext else {
            return ""
        }
        messageLog += text
        var input_str = "<|im_start|>user\n\(text)<|im_end|>\n<|im_start|>assistant\n"
        var generate_str = ""
        var generate_cnt = 0
        
        let t_start = DispatchTime.now().uptimeNanoseconds
        await llamaContext.completion_init(text: input_str)
        let t_heat_end = DispatchTime.now().uptimeNanoseconds
        let t_heat = Double(t_heat_end - t_start) / NS_PER_S

        //Task.detached {
            while await !llamaContext.is_done {
                let result = await llamaContext.completion_loop()
                if(stream){
                    await MainActor.run {
                        self.messageLog += "\(result)"
                        generate_str += "\(result)"
                        generate_cnt += 1
                    }
                }else{
                    generate_str += result
                    generate_cnt += 1
                }
//
            }

            let t_end = DispatchTime.now().uptimeNanoseconds
            let t_generation = Double(t_end - t_heat_end) / self.NS_PER_S
            //let tokens_per_second = Double(await llamaContext.n_len) / t_generation
            let tokens_per_second = Double(generate_cnt) / t_generation

            await llamaContext.clear()
        if(!stream){
            self.messageLog += generate_str
        }
            await MainActor.run {
                self.messageLog += """
                    \n
                    Done
                    Heat up took \(t_heat)s
                    Generated \(generate_cnt)t/\(t_generation)s, \(tokens_per_second) t/s\n\n
                    """
            }
            //return generate_str
        //}
        
        return generate_str
//        await completeTTS(text: "[Sbreak]" + generate_str + "[Pbreak][oral_2][laugh_0][break_4]")
//        await self.tts?.generate_by_str(text: "[Sbreak]" + generate_str + "[Pbreak][oral_2][laugh_0][break_4]")
//        await MainActor.run {
//            self.messageLog += self.tts?.messageLog ?? ""
//        }
    }

    func bench() async {
        guard let llamaContext else {
            return
        }

        messageLog += "\n"
        messageLog += "Running benchmark...\n"
        messageLog += "Model info: "
        messageLog += await llamaContext.model_info() + "\n"

        let t_start = DispatchTime.now().uptimeNanoseconds
        let _ = await llamaContext.bench(pp: 8, tg: 4, pl: 1) // heat up
        let t_end = DispatchTime.now().uptimeNanoseconds

        let t_heat = Double(t_end - t_start) / NS_PER_S
        messageLog += "Heat up time: \(t_heat) seconds, please wait...\n"

        // if more than 5 seconds, then we're probably running on a slow device
        if t_heat > 5.0 {
            messageLog += "Heat up time is too long, aborting benchmark\n"
            return
        }

        let result = await llamaContext.bench(pp: 512, tg: 128, pl: 1, nr: 3)

        messageLog += "\(result)"
        messageLog += "\n"
    }

    func clear() async {
        guard let llamaContext else {
            return
        }

        await llamaContext.clear()
        messageLog = ""
    }
}
