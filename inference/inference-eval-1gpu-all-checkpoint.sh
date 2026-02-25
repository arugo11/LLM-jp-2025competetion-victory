base_path="./models/HayatoHongoEveryonesAI/llm-jp-4-8b-instruct-sft-v5-2"

for checkpoint_dir in "$base_path"/checkpoint*; do
    if [ -d "$checkpoint_dir" ]; then
        # パスからファイル名(フォルダ名)だけを取り出す
        folder_name=$(basename "$checkpoint_dir")
        
        echo "Evaluating checkpoint: $folder_name"

        qsub -v MODEL_USER=HayatoHongoEveryonesAI/llm-jp-4-8b-instruct-sft-v5-2,MODEL_REPO=$folder_name \
                ./inference-eval-1gpu-math.sh
    fi
done
    
    