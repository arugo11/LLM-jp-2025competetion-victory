## テスト用実行コマンド
uv run python main.py \
    --model_path models/HayatoHongoEveryonesAI/llm-jp-4-8b-instruct \
    --input_path sample_problems.jsonl \
    --output_path output.jsonl \
    --log_path inference_log.jsonl \
    --tir-llm-host 127.0.0.1 \
    --tir-llm-port 8000 \
    --tir-sandbox-host 127.0.0.1 \
    --tir-sandbox-port 6000 \
    --max-new-tokens 512 \
    --temperature 0.0 \
    --retry-temperature 0.2 \
    --repair-attempts 3 \
    --format-retry-attempts 5 \
    --direct-answer-attempts 2 \
    --log-raw-output \
    --enable-wandb \
    --wandb-project "miyako-personal/llm-jp-4-tir"