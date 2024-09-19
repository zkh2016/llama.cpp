import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer

torch.manual_seed(0)

#model_path = "D:\\project\\skip\\lenovo2715-16"
model_path = "D:\\project\\skip\\lenovo2715"
out_model_path = "D:\\project\\llama3\\"

model = AutoModel.from_pretrained(model_path, trust_remote_code=True,
    attn_implementation='sdpa', torch_dtype=torch.bfloat16) # sdpa or flash_attention_2, no eager
model = model.eval().cuda()
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

llm = model.llm
state_dict = llm.state_dict()


new_state_dict = {}

print(state_dict.keys())
print(state_dict["model.embed_tokens.weight"].shape)

new_state_dict["model.embed_tokens.weight"] = torch.randn((1, 4096), dtype=state_dict["model.embed_tokens.weight"].dtype) 
new_state_dict["lm_head.weight"] = torch.randn((4096, 1), dtype=state_dict["lm_head.weight"].dtype) 
new_state_dict["model.norm.weight"] = state_dict["model.norm.weight"]

'''
model.layers.0.self_attn.q_proj.weight
model.layers.0.self_attn.k_proj.weight
model.layers.0.self_attn.v_proj.weight
model.layers.0.self_attn.o_proj.weight
model.layers.0.mlp.gate_proj.weight
model.layers.0.mlp.up_proj.weight
model.layers.0.mlp.down_proj.weight
model.layers.0.input_layernorm.weight
model.layers.0.post_attention_layernorm.weight
'''

# skip_list = [0, 5, 10, 16, 21, 26, 31]
skip_list = []
for i in range(16):
    skip_list.append(i + 16)

for i in range(len(skip_list)):
    print("convert layer ", i)
    old_i = skip_list[i]
    new_state_dict[f"model.layers.{i}.self_attn.q_proj.weight"] = state_dict[f"model.layers.{old_i}.self_attn.q_proj.weight"]
    new_state_dict[f"model.layers.{i}.self_attn.k_proj.weight"] = state_dict[f"model.layers.{old_i}.self_attn.k_proj.weight"]
    new_state_dict[f"model.layers.{i}.self_attn.v_proj.weight"] = state_dict[f"model.layers.{old_i}.self_attn.v_proj.weight"]
    new_state_dict[f"model.layers.{i}.self_attn.o_proj.weight"] = state_dict[f"model.layers.{old_i}.self_attn.o_proj.weight"]
    new_state_dict[f"model.layers.{i}.mlp.gate_proj.weight"] = state_dict[f"model.layers.{old_i}.mlp.gate_proj.weight"]
    new_state_dict[f"model.layers.{i}.mlp.up_proj.weight"] = state_dict[f"model.layers.{old_i}.mlp.up_proj.weight"]
    new_state_dict[f"model.layers.{i}.mlp.down_proj.weight"] = state_dict[f"model.layers.{old_i}.mlp.down_proj.weight"]
    new_state_dict[f"model.layers.{i}.input_layernorm.weight"] = state_dict[f"model.layers.{old_i}.input_layernorm.weight"]
    new_state_dict[f"model.layers.{i}.post_attention_layernorm.weight"] = state_dict[f"model.layers.{old_i}.post_attention_layernorm.weight"]

torch.save(new_state_dict, out_model_path + "pytorch_model.bin")

image = Image.open("D:\\project\\minicpmv2.7\\Archive\\0.jpg")

question = "tell me about the image"

msgs = [{'role':'user', 'content':[image, question]}]

ans = model.chat(image=None, msgs=msgs, tokenizer=tokenizer)
print(ans)