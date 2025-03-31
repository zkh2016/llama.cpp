
#model=/DATA/disk1/zhangkaihuo/Lenovo_V2_5/model/ggml-model-Q4_K_M.gguf
model=/DATA/disk1/zhangkaihuo/3b_sft_4k_for_zkh/model/ggml-model-Q4_K_M.gguf
lora_model=/DATA/disk1/zhangkaihuo/3b_sft_4k_for_zkh_lora/ggml-adapter-model.bin
fp_model=/DATA/disk1/zhangkaihuo/Lenovo_V2_5/model/ggml-model-f16.gguf
#mmproj=/DATA/disk1/zhangkaihuo/Lenovo_V2_5/mmproj-model-f16.gguf
mmproj=/DATA/disk1/zhangkaihuo/3b_sft_4k_for_zkh/mmproj-model-f16.gguf
skip_model=/DATA/disk1/zhangkaihuo/llama3/ggml-model-f16.gguf
image=/DATA/disk1/zhangkaihuo/Lenovo_V2_5/dataset/Difficult-Single-Chart-Histogram_003.png
temperature=0.3
top_p=0.3
seed=42

cmd=$1
image=$2

case "${cmd}" in
    skip)
        #set -x
        ./build/bin/minicpmv-cli -m ${model} --mmproj ${mmproj} --skip-model ${skip_model} -c 8192 --temp ${temperature} --top-p ${top_p} --top-k 100 --repeat-penalty 1.05 -p "Please extract information from the PPT image given you and provide a brief description." --image ${image} -ngl 100 --skip-layers 8 --log-disable --seed ${seed}
        ;;
    fp_skip)
        #set -x
        ./build/bin/minicpmv-cli -m ${fp_model} --mmproj ${mmproj} --skip-model ${skip_model} -c 8192 --temp ${temperature} --top-p ${top_p} --top-k 100 --repeat-penalty 1.05 -p "Please extract information from the PPT image given you and provide a brief description." --image ${image} -ngl 100 --skip-layers 8 --log-disable --seed ${seed}
        ;;
#
#no skip 
    no_skip)
        set -x
        ./build/bin/minicpmv-cli -m ${model} --mmproj ${mmproj} -c 8192 --temp ${temperature} --top-p ${top_p} --top-k 100 --repeat-penalty 1.05 -p "Please extract information from the PPT image given you and provide a brief description." --image ${image} -ngl 100 --seed ${seed}
        ;;
    fp_no_skip)
        #set -x
        ./build/bin/minicpmv-cli -m ${fp_model} --mmproj ${mmproj} -c 8192 --temp ${temperature} --top-p ${top_p} --top-k 100 --repeat-penalty 1.05 -p "Please extract information from the PPT image given you and provide a brief description." --image ${image} -ngl 100 --skip-layers 8 --log-disable --seed ${seed}
        ;;

#int8 vit and skip
    vit_int8)
        mmproj=/DATA/disk1/zhangkaihuo/Lenovo_V2_5/mmproj-model-Q8_0_M.gguf
        #mmproj=/DATA/disk1/zhangkaihuo/project/lenovo/llama.cpp/mmproj-model-Q8_0_M.gguf
        set -x
        ./build/bin/minicpmv-cli -m ${model} --mmproj ${mmproj} --skip-model ${skip_model} -c 8192 --temp ${temperature} --top-p ${top_p} --top-k 100 --repeat-penalty 1.05 -p "Please extract information from the PPT image given you and provide a brief description." --image ${image} -ngl 100 --skip-layers 8 --seed ${seed} --log-disable
        ;;
    lora)
        set -x
        ./build/bin/minicpmv-cli -m ${model} --mmproj ${mmproj} -c 8192 --temp ${temperature} --top-p ${top_p} --top-k 100 --repeat-penalty 1.05 -p "Please extract information from the PPT image given you and provide a brief description." --image ${image} -ngl 100 --skip-layers 8 --seed ${seed} --lora ${lora_model} --log-disable -t 1 
        ;;
esac
