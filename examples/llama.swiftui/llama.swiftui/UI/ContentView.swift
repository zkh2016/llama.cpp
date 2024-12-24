import SwiftUI

struct ContentView: View {
//    @StateObject var llamaState = LlamaState()
//    @StateObject var ttsState  = ConditionalChatTTS()
    @StateObject var llamaState = Benchmark()
    //@StateObject var vitState = ViT()

    @State private var multiLineText = ""
    @State private var showingHelp = false    // To track if Help Sheet should be shown

    var body: some View {
        NavigationView {
            VStack {
                ScrollView(.vertical, showsIndicators: true) {
                    //Text("\(llamaState.messageLog) \(ttsState.messageLog)")
                    Text("\(llamaState.messageLog)")
                        .font(.system(size: 12))
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .padding()
                        .onTapGesture {
                            UIApplication.shared.sendAction(#selector(UIResponder.resignFirstResponder), to: nil, from: nil, for: nil)
                        }
                }

                TextEditor(text: $multiLineText)
                    .frame(height: 80)
                    .padding()
                    .border(Color.gray, width: 1)

                HStack {
//                    Button("TestLLM") {
//                        sendText()
//                    }
//                    Button("TestLLMStream") {
//                        sendTextStream()
//                    }
//
//                    Button("TestTTS") {
//                        test_tts()
//                    }
                    Button("bench") {
                        bench()
                    }

                    Button("bench_sync") {
                       bench_sync()
                    }

                    Button("test_prefill") {
                        test_prefill()
                    }
//                    Button("Copy") {
//                        UIPasteboard.general.string = llamaState.messageLog
//                    }
                }
                .buttonStyle(.bordered)
                .padding()

//                NavigationLink(destination: DrawerView(llamaState: llamaState)) {
//                    Text("View Models")
//                }
                .padding()

            }
            .padding()
            .navigationBarTitle("Model Settings", displayMode: .inline)

        }
    }

    func sendText() {
        
        Task {
            var prompt = multiLineText
            multiLineText = ""
            let generate_str = await llamaState.complete(text: prompt, stream: false)
            //multiLineText = ""
            //await ttsState.generate_by_str(text: generate_str, stream: false)
        }
    }

    func sendTextStream() {
        
        Task {
            var prompt = multiLineText
            multiLineText = ""
            let generate_str = await llamaState.complete(text: prompt)
            //multiLineText = ""
            //await ttsState.generate_by_str(text: generate_str)
        }
    }
    func test_tts() {
        Task {
            var prompt = multiLineText
            //await ttsState.generate_by_str(text: prompt)
        }
    }
    
    func bench() {
        
        Task {
            var prompt = multiLineText
            multiLineText = ""
            let generate_str = await llamaState.complete(text: prompt, stream: false)
        }
    }
    func bench_sync() {
        
        Task {
            var prompt = multiLineText
            multiLineText = ""
            llamaState.producer(text: prompt)
            llamaState.tts?.consumer()
            
        }
    }

    func test_prefill() {
        Task {
            //await llamaState.clear()
            //await llamaState.test_vit_and_llm_prefill(text: multiLineText)
            await llamaState.test_async_vit_and_llm(text: multiLineText)
            multiLineText = ""
            //vitState.performance()
        }
    }
    struct DrawerView: View {

        @ObservedObject var llamaState: LlamaState
        @State private var showingHelp = false
        func delete(at offsets: IndexSet) {
            offsets.forEach { offset in
                let model = llamaState.downloadedModels[offset]
                let fileURL = getDocumentsDirectory().appendingPathComponent(model.filename)
                do {
                    try FileManager.default.removeItem(at: fileURL)
                } catch {
                    print("Error deleting file: \(error)")
                }
            }

            // Remove models from downloadedModels array
            llamaState.downloadedModels.remove(atOffsets: offsets)
        }

        func getDocumentsDirectory() -> URL {
            let paths = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)
            return paths[0]
        }
        var body: some View {
            List {
                Section(header: Text("Download Models From Hugging Face")) {
                    HStack {
                        InputButton(llamaState: llamaState)
                    }
                }
                Section(header: Text("Downloaded Models")) {
                    ForEach(llamaState.downloadedModels) { model in
                        DownloadButton(llamaState: llamaState, modelName: model.name, modelUrl: model.url, filename: model.filename)
                    }
                    .onDelete(perform: delete)
                }
                Section(header: Text("Default Models")) {
                    ForEach(llamaState.undownloadedModels) { model in
                        DownloadButton(llamaState: llamaState, modelName: model.name, modelUrl: model.url, filename: model.filename)
                    }
                }

            }
            .listStyle(GroupedListStyle())
            .navigationBarTitle("Model Settings", displayMode: .inline).toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("Help") {
                        showingHelp = true
                    }
                }
            }.sheet(isPresented: $showingHelp) {    // Sheet for help modal
                VStack(alignment: .leading) {
                    VStack(alignment: .leading) {
                        Text("1. Make sure the model is in GGUF Format")
                               .padding()
                        Text("2. Copy the download link of the quantized model")
                               .padding()
                    }
                    Spacer()
                   }
            }
        }
    }
}

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}
