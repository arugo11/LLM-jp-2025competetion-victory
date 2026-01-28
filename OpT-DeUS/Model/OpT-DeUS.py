from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import transformers
from datasets import load_dataset, load_from_disk, concatenate_datasets
from functools import partial
from datasets import Dataset
from transformers import DataCollatorForLanguageModeling,set_seed
from transformers import AutoTokenizer
from transformers import optimization
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer,get_wsd_schedule
import torch
import pandas as pd
import sys
from accelerate import Accelerator
import random
import numpy as np
import torch.nn.functional as F
import re
import os
from accelerate.utils import DistributedDataParallelKwargs
from torch.optim import Optimizer
from pathlib import Path
from peft import LoraConfig, LoraModel, get_peft_model, PeftModel
from transformers import TrainerCallback
from datasets import config
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetCount, nvmlDeviceGetUtilizationRates,nvmlDeviceGetMemoryInfo
from huggingface_hub import login
import copy
import time
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
set_seed(42)
import ot
def compute_ot_mapping(W1, W2, reg=0.06, numItermax=10000): # Solve OT Problem (cf. line 7)
    # 1. 計算安定性とエラー回避のため、重みを float32 に変換
    W1 = W1.to(dtype=torch.float32)
    W2 = W2.to(dtype=torch.float32)
    
    # 2. デバイス(CPU/GPU)を取得し、分布ベクトルを同じデバイス・型で作成
    device = W1.device
    a = torch.ones(W1.shape[0], device=device, dtype=torch.float32) / W1.shape[0]
    b = torch.ones(W2.shape[0], device=device, dtype=torch.float32) / W2.shape[0]
    
    # 3. コスト行列の計算
    M = torch.cdist(W1, W2, p=2)  # Initialize Cost Matrix (cf. line 4)
    
    # 4. Sinkhorn アルゴリズムの実行
    T = ot.sinkhorn(a, b, M, reg, numItermax=numItermax)  # Initialize alpha and beta (cf. line 3)
    
    # 必要であれば、戻り値を元の型(BFloat16)に戻す処理が必要かもしれませんが、
    # 続く行列演算(apply_mapping)でPyTorchが自動的に型昇格(float32)するため、
    # ここでは float32 のまま返しても動作するはずです。
    return T

def apply_mapping(W, T): 
    # T (float32) と W (float32) で計算を行う
    return T.T @ W

def average_weight_ot(previous_index, next_index, model):
    previous = model.model.layers[previous_index].state_dict()
    next = model.model.layers[next_index].state_dict()
    T_cache = {}
    fused_weight = {}
    
    for key in previous:
        # 修正: 計算用にあらかじめ float32 に変換し、元の型(dtype)を記録しておく
        target_dtype = previous[key].dtype  # おそらく torch.bfloat16
        W1 = previous[key].to(dtype=torch.float32)
        W2 = next[key].to(dtype=torch.float32)

        # 以降の計算はすべて float32 で行われるためエラーが解消されます
        
        if key in ["input_layernorm.weight"]: 
            fused_weight[key] = (W1 + W2) / 2
            
        if key in ["self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight"]:
            T_out = compute_ot_mapping(W1, W2)
            T_cache[key] = T_out
            W1_aligned = apply_mapping(W1, T_out)
            fused_weight[key] = (W1_aligned + W2) / 2 
            
        if key in ["self_attn.o_proj.weight", "mlp.down_proj.weight"]:
            T_out = compute_ot_mapping(W1, W2)
            T_cache[key] = T_out
            W1_aligned = apply_mapping(W1, T_out)
            fused_weight[key] = (W1_aligned + W2) / 2 
            
        if key == "post_attention_layernorm.weight":
            T_in = T_cache["self_attn.o_proj.weight"]
            I = torch.zeros_like(T_in)
            I.fill_diagonal_(1.0)
            T_in = 0.5 * T_in + 0.5 * I 
            T_cache[key] = T_in
            W1_aligned = W1 @ T_in 
            fused_weight[key] = (W1_aligned + W2) / 2 
            
        if key in ["mlp.gate_proj.weight", "mlp.up_proj.weight"]: 
            T_in = T_cache["self_attn.o_proj.weight"]
            W1 = W1 @ T_in 
            T_out = compute_ot_mapping(W1, W2)
            T_cache[key] = T_out
            W1_aligned = apply_mapping(W1, T_out)
            fused_weight[key] = (W1_aligned + W2) / 2 

        # 修正: 計算結果を元の型 (BFloat16) に戻して保存
        if key in fused_weight:
            fused_weight[key] = fused_weight[key].to(dtype=target_dtype)

    return fused_weight

model = AutoModelForCausalLM.from_pretrained("HayatoHongoEveryonesAI/open-instruct-grpo-fast", revision="grpo_fast__3__1766613992") #Load Basemodel
tokenizer = AutoTokenizer.from_pretrained("HayatoHongoEveryonesAI/open-instruct-grpo-fast", revision="grpo_fast__3__1766613992")
Expand = AutoModelForCausalLM.from_pretrained("HayatoHongoEveryonesAI/open-instruct-grpo-fast", revision="grpo_fast__3__1766613992",num_hidden_layers=48)
index_mapping = { # Interpolation positions are set to 1 at the top half of model
    0:0,
    1:1,
    2:2,
    3:3,
    4:4,
    5:5,
    6:6,
    7:7,
    8:8,
    9:9,
    10:10,
    11:11,
    12:12,
    13:13,
    14:14,
    15:15,
    16:1,
    17:16,
    18:1,
    19:17,
    20:1,
    21:18,
    22:1,
    23:19,
    24:1,
    25:20,
    26:1,
    27:21,
    28:1,
    29:22,
    30:1,
    31:23,
    32:1,
    33:24,
    34:1,
    35:25,
    36:1,
    37:26,
    38:1,
    39:27,
    40:1,
    41:28,
    42:1,
    43:29,
    44:1,
    45:30,
    46:1,
    47:31
}


for new_idx, old_idx in index_mapping.items():
    Expand.model.layers[new_idx].load_state_dict(model.model.layers[old_idx].state_dict())
for i in [16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46]:#These will be the trainable layers
    print("Calculating the", i, " layer")
    Expand.model.layers[i].load_state_dict(average_weight_ot(i - 1, i + 1, Expand))
    nn.init.zeros_(Expand.model.layers[i].self_attn.o_proj.weight)  #(cf. line 11)
    nn.init.zeros_(Expand.model.layers[i].mlp.down_proj.weight) #(cf. line 11)
Expand.save_pretrained("HayatoHongoEveryonesAI/open-instruct-grpo-fast-expand",
                       safe_serialization=True, 
                        max_shard_size="5GB")
tokenizer.save_pretrained("HayatoHongoEveryonesAI/open-instruct-grpo-fast-expand")














