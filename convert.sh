#!/bin/bash
model_path=$1

echo "model path: "${model_path}

python examples/llava/minicpmv-surgery.py -m ${model_path}

python examples/llava/minicpmv-convert-image-encoder-to-gguf.py -m ${model_path} --minicpmv-projector ${model_path}/minicpmv.projector --output-dir ${model_path} --image-mean 0.5 0.5 0.5 --image-std 0.5 0.5 0.5 --minicpmv_version 6

python examples/llava/layer_skip.py -m ${model_path}

python convert-hf-to-gguf.py ${model_path}/model

./build/bin/quantize ${model_path}/model/ggml-model-f16.gguf ${model_path}/model/ggml-model-Q4_K_M.gguf Q4_K_M
