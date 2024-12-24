import Foundation
import CoreML
import SwiftUI


class ViT: ObservableObject {
    @Published var messageLog = ""
    let NS_PER_S = 1_000_000_000.0
    var vpm : ane_vit_fp16?
    var resampler : exported_resampler_f16?

    init(){
        //self.messageLog = messageLog
        do {
            print("start load vit .....")
            self.vpm =  try ane_vit_fp16(configuration: MLModelConfiguration())
            print("start load resampler .....")
            self.resampler =  try exported_resampler_f16(configuration: MLModelConfiguration())
        }catch {
            print("load vpm or resampler failed")
        }
    }
    
    func performance(){
        let input = try! MLMultiArray(shape: [1, 3, 14, 14336], dataType: .float32)
        let tgt_sizes = try! MLMultiArray(shape: [1, 2], dataType: .float32)
        print("call vpm prediction")
        let output = try! self.vpm?.prediction(input:input, tgt_sizes: tgt_sizes)
        let t0 = DispatchTime.now().uptimeNanoseconds
        for i in 0..<10{
            try! self.vpm?.prediction(input:input, tgt_sizes: tgt_sizes)
        }
        let t1 = DispatchTime.now().uptimeNanoseconds
        let vit_time = Double(t1 - t0) / NS_PER_S/10
        print("vpm time: \(vit_time)s")
        print(output)
        print(output?.featureNames)
        print(output?.output.shape)
        
        for i in 0..<10{
            print("元素 \(i): \(output?.output[i])")
        }

        //let input = try! MLMultiArray(shape: [1, 1024, 1152], dataType: .float32)
        let patch_sizes = try! MLMultiArray(shape: [1, 2], dataType: .float32)
        print("call resampler prediction")
        let resampler_output = try! self.resampler?.prediction(features_1:output!.output, patch_sizes: patch_sizes)
        let tt0 = DispatchTime.now().uptimeNanoseconds
        for i in 0..<10{
            try! self.resampler?.prediction(features_1:output!.output, patch_sizes: patch_sizes)
        }
        let tt1 = DispatchTime.now().uptimeNanoseconds
        let resampler_time = Double(tt1 - tt0) / NS_PER_S/10
        print("resampler time: \(resampler_time)s")
        print(resampler_output)
        print(resampler_output?.featureNames)
        print(resampler_output?.var_7249.shape)
        
        for i in 0..<10{
            print("元素 \(i): \(resampler_output?.var_7249[i])")
        }
    }
    
    func forward(input: MLMultiArray, tgt_sizes: MLMultiArray, patch_sizes: MLMultiArray) -> MLMultiArray {
        let vpm_output = try! self.vpm?.prediction(input:input, tgt_sizes: tgt_sizes)
        let output = try! self.resampler?.prediction(features_1:vpm_output!.output, patch_sizes: patch_sizes)
        return output!.var_7249
    }
}
