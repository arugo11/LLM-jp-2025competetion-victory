from datasets import load_dataset
ds = load_dataset("togethercomputer/RedPajama-Data-1T", config="arxiv", split="train", trust_remote_code=True)