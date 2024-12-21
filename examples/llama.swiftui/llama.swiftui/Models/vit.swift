import Foundation
import CoreML
import SwiftUI


class ViT: ObservableObject {
    @Published var messageLog = ""
    let NS_PER_S = 1_000_000_000.0
    var model : exported_minicpm3o_f16?

    init(){
        //self.messageLog = messageLog
        do {
            print("start load vit .....")
            self.model =  try exported_minicpm3o_f16(configuration: MLModelConfiguration())
        }catch {
            print("load exported_minicpm3o_f16 failed")
        }
    }
    
    func test(){
        let input = try! MLMultiArray(shape: [1, 3, 14, 14336], dataType: .float32)
        let tgt_sizes = try! MLMultiArray(shape: [1, 2], dataType: .float32)
        print("call prediction")
        let output = try! self.model?.prediction(pixel_values:input, tgt_sizes: tgt_sizes)
//        let t0 = DispatchTime.now().uptimeNanoseconds
//        for i in 0..<10{
//            try! self.model?.prediction(pixel_values:input, tgt_sizes: tgt_sizes)
//        }
//        let t1 = DispatchTime.now().uptimeNanoseconds
//        let vit_time = Double(t1 - t0) / NS_PER_S
//        print("vit time: \(vit_time)s")
        print(output)
        print(output?.featureNames)
        print(output?.var_1781.shape)
        
        for i in 0..<10{
            print("元素 \(i): \(output?.var_1781[i])")
        }
    }
    func forward(){
       
    }
}
