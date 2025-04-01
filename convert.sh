#!/bin/bash
model_path=$1
vit_dtype=$2
version=$3

echo "model path: "${model_path}

python examples/llava/minicpmv-surgery.py -m ${model_path}

case "${vit_dtype}" in
    fp16)
        python examples/llava/minicpmv-convert-image-encoder-to-gguf.py -m ${model_path} --minicpmv-projector ${model_path}/minicpmv.projector --output-dir ${model_path} --image-mean 0.5 0.5 0.5 --image-std 0.5 0.5 0.5 --minicpmv_version ${version}
        ;;
    int8)
        #int8
        gptq_model=$4
        python examples/llava/minicpmv-convert-image-encoder-to-gguf-Q8_0.py -m ${model_path} --minicpmv-projector ${model_path}/minicpmv.projector --output-dir ${model_path} --image-mean 0.5 0.5 0.5 --image-std 0.5 0.5 0.5 --minicpmv_version ${version} --gptq ${gptq_model}
        ;;
esac

python examples/llava/layer_skip.py -m ${model_path}

python convert-hf-to-gguf.py ${model_path}/model

./build/bin/quantize ${model_path}/model/ggml-model-f16.gguf ${model_path}/model/ggml-model-Q4_K_M.gguf Q4_K_M

python get_lora.py
python convert-lora-to-ggml.py /DATA/disk1/zhangkaihuo/3b_sft_4k_for_zkh_lora/
