import os
import torch
import struct
import argparse
from transformers import AutoModel, AutoTokenizer

def save_last_n_layers(model, n, save_path):
    state_dict = model.state_dict()
    layer_keys = [key for key in state_dict.keys() if "layer" in key]
    sorted_layer_keys = sorted(layer_keys, key=lambda x: int(x.split('.')[2]))
    last_n_layer_keys = sorted_layer_keys[-(n*9):]
    new_state_dict = {}
    for key in last_n_layer_keys:
        new_key = key.split('.')
        id = new_key[2]
        id = int(id)+n-32
        new_key[2] = str(id)
        new_key = '.'.join(new_key)
        print(key, new_key)
        new_state_dict[new_key] = state_dict[key]
        
    for key in state_dict.keys():
        if "layer" not in key:
            if "model.embed_tokens.weight" in key or "lm_head.weight" in key:
                new_state_dict[key] = torch.zeros([1,4096])
            else:
                new_state_dict[key] = state_dict[key]
            print(key, state_dict[key].shape, new_state_dict[key].shape)
    torch.save(new_state_dict, save_path)
    
    embedding_layer = model.model.embed_tokens
    indexs = [128010, 128011, 128020, 128021]
    with open(f"{model_path}/model_skip/sp.raw", "wb") as f:
        for index in indexs:
            indices = torch.tensor([index])
            embedding_vector = embedding_layer(indices)
            tensor_list = embedding_vector.squeeze()
            print(tensor_list[:3])
            for res in tensor_list:
                res = struct.pack('f', res)
                f.write(res)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", help="Path to MiniCPM-V model")
    args = ap.parse_args()
    
    model_path = args.model
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True, local_files_only=True, torch_dtype=torch.float16).llm
    config = model.config
    config.auto_map = {
        "AutoConfig": "configuration_minicpm.MiniCPMConfig",
        "AutoModel": "modeling_minicpm.MiniCPMModel",
        "AutoModelForCausalLM": "modeling_minicpm.MiniCPMForCausalLM",
        "AutoModelForSeq2SeqLM": "modeling_minicpm.MiniCPMForCausalLM",
        "AutoModelForSequenceClassification": "modeling_minicpm.MiniCPMForSequenceClassification"
    }
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tok.save_pretrained(f"{args.model}/model_skip")
    save_last_n_layers(model, 8, f'{args.model}/model_skip/pytorch_model.bin')
