
import datasets; 
datasets.load_dataset(
    'nvidia/OpenMathInstruct-2', 
    split='train', 
    cache_dir='~/LLM-jp-2025competetion-victory/datasets/hf_cache'
)