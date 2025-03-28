

from transformers import AutoModel
import torch

import re

def trans(key, value):
    if 'lora_B' in key:
        return value.t().contiguous()
    return value

def replace_lora_key(original_key, value):
    """将 xxx_lora.lora_A 结构替换为 xxx.lora_A.weight"""
    key = re.sub(
        r'(\w+)_lora\.lora_A',    # 匹配任意单词 + _lora.lora_A
        r'\1.lora_A.weight',      # 替换为 单词 + .lora_A.weight
        original_key
    )
    key = re.sub(
        r'(\w+)_lora\.lora_B',    # 匹配任意单词 + _lora.lora_A
        r'\1.lora_B.weight',      # 替换为 单词 + .lora_A.weight
        key 
    )
    print(original_key, "->", key, value.dtype, value.shape)
    return key

# 测试用例
#original = "model.layers.0.self_attn.q_proj_lora.lora_A"
#new_str = replace_lora_key(original)
#print(f"替换前：{original}")
#print(f"替换后：{new_str}")


# 配置参数
model_path = '/DATA/disk1/zhangkaihuo/3b_sft_4k_for_zkh'
output_path = '/DATA/disk1/zhangkaihuo/3b_sft_4k_for_zkh_lora/adapter_model.bin' # 指定输出文件路径

# 1. 加载模型
model = AutoModel.from_pretrained(model_path, trust_remote_code=True, torch_dtype="auto", attn_implementation="sdpa")

# 2. 获取模型的状态字典
state_dict = model.state_dict()

# 3. 过滤包含'lora'的key
prefix = "llm."
lora_weights = {
    #replace_lora_key(key, value): value.t().contiguous()
    replace_lora_key(key, value): trans(key, value)
    for key, value in state_dict.items() 
    if "lora" in key.lower()  # 不区分大小写匹配lora
}

# 可选：检查是否找到LoRA权重
if not lora_weights:
    print("警告：未找到包含'lora'的权重参数！")

# 4. 保存结果
torch.save(lora_weights, output_path)
print(f"成功保存LoRA权重至 {output_path}")

