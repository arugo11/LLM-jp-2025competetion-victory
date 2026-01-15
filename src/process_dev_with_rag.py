import json
import os
from tqdm import tqdm

import embeddings


def load_dev_jsonl(file_path):
    """dev.jsonlファイルを読み込んでリストとして返す"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data


def save_processed_jsonl(data, output_path):
    """処理済みデータをjsonlファイルとして保存"""
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in data:
            json.dump(item, f, ensure_ascii=False)
            f.write('\n')


def process_problems_with_rag(data, retriever, top_k=3):
    """各問題に対してRAG検索を実行し、結果を結合"""
    processed_data = []
    
    for item in tqdm(data, desc="Processing problems with RAG"):
        problem = item['problem']
        
        # RAGで類似問題と解答を検索
        rag_results = retriever.invoke_answers(problem)
        
        # 検索結果を整形
        retrieved_examples = []
        for i, result in enumerate(rag_results[:top_k]):
            example = f"【例{i+1}】\n問題: {result['question']}\n解答: {result['answer']}"
            retrieved_examples.append(example)
        
        # 検索結果を結合して新しいproblemを作成
        if retrieved_examples:
            retrieved_text = "\n\n".join(retrieved_examples)
            enhanced_problem = f"以下の類似問題を参考にして、次の問題を解いてください。\n\n{retrieved_text}\n\n【解くべき問題】\n{problem}"
        else:
            enhanced_problem = problem
        
        # 元のデータをコピーして、problemを置き換え
        new_item = item.copy()
        new_item['original_problem'] = problem  # 元の問題を保持
        new_item['problem'] = enhanced_problem  # RAGで拡張した問題に置き換え
        new_item['rag_results'] = [
            {
                'question': r['question'],
                'answer': r['answer']
            } for r in rag_results[:top_k]
        ]  # 検索結果も保存
        
        processed_data.append(new_item)
    
    return processed_data


def main():
    # パス設定
    dev_jsonl_path = "data/dev.jsonl"
    vectorstore_path = "data/qa_vectorstore"
    output_path = "data/dev_with_rag.jsonl"
    
    # ベクトルストアの読み込み
    if not os.path.exists(vectorstore_path):
        print("Error: Vectorstore not found. Please run main.py first to create the vectorstore.")
        return
    
    print("Loading vectorstore...")
    retriever = embeddings.Retriever.load(vectorstore_path)
    print("Vectorstore loaded successfully.")
    
    # dev.jsonlファイルの読み込み
    print(f"Loading {dev_jsonl_path}...")
    dev_data = load_dev_jsonl(dev_jsonl_path)
    print(f"Loaded {len(dev_data)} problems.")
    
    # RAGで処理
    print("Processing problems with RAG...")
    processed_data = process_problems_with_rag(dev_data, retriever, top_k=3)
    
    # 結果を保存
    print(f"Saving results to {output_path}...")
    save_processed_jsonl(processed_data, output_path)
    print(f"Saved {len(processed_data)} processed problems.")
    
    # サンプル表示
    print("\n" + "="*50)
    print("Sample of processed data (first item):")
    print("="*50)
    if processed_data:
        sample = processed_data[0]
        print(f"Original problem: {sample.get('original_problem', 'N/A')[:100]}...")
        print(f"\nEnhanced problem: {sample['problem'][:300]}...")
        print(f"\nNumber of RAG results: {len(sample.get('rag_results', []))}")


if __name__ == "__main__":
    main()