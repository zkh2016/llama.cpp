## tmp_project_l

### Prepare models and code

Clone llama.cpp:
```bash
git clone git@github.com:OpenBMB/llama.cpp.git
cd llama.cpp

```

### Usage of tmp_project_l

Replace ['tmp_project_l'] with the actual model location

```bash
git checkout tmp_project_l
python ./examples/llava/layer_skip.py -m ['tmp_project_l']
git checkout minicpmv-main
python ./examples/llava/minicpmv-surgery.py -m ['tmp_project_l']
python ./examples/llava/minicpmv-convert-image-encoder-to-gguf.py -m ['tmp_project_l'] --minicpmv-projector ['tmp_project_l']/minicpmv.projector --output-dir ['tmp_project_l']/ --image-mean 0.5 0.5 0.5 --image-std 0.5 0.5 0.5 --minicpmv_version 4
```

add 'res = "llama-bpe"' in convert_hf_to_gguf.py 514 line
```bash
python ./convert_hf_to_gguf.py ['tmp_project_l']/model
```

delete code in convert_hf_to_gguf.py 470 line
```bash
python ./convert_hf_to_gguf.py ['tmp_project_l']/model_skip
```

Build for Linux or Mac

```bash
git checkout tmp_project_l
make
```

Inference on Linux or Mac
```
# run f16 version
./minicpmv-cli -m ['tmp_project_l']/model/ggml-model-f16.gguf --mmproj ['tmp_project_l']/mmproj-model-f16.gguf -c 4096 --temp 0.7 --top-p 0.8 --top-k 100 --repeat-penalty 1.05 --image xx.jpg -p "What is in the image?"
```