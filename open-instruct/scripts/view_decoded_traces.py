#!/usr/bin/env python3
"""
トレースファイルからデコード済みテキストを読みやすく表示するスクリプト

Usage:
    python scripts/view_decoded_traces.py traces_flattened.csv [options]
"""

import csv
import argparse
from pathlib import Path


def display_decoded_samples(csv_file: str, num_samples: int = 10, step: int = None, score_filter: float = None):
    """デコード済みサンプルを表示"""
    print(f"Reading from: {csv_file}\n")
    
    samples = []
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # フィルタリング
            if step is not None and int(row['training_step']) != step:
                continue
            if score_filter is not None:
                score = float(row['score'])
                if score_filter > 0 and score == 0:
                    continue
                if score_filter == 0 and score > 0:
                    continue
            
            samples.append(row)
    
    if len(samples) == 0:
        print("No samples found matching the criteria.")
        return
    
    # 表示するサンプル数を制限
    display_samples = samples[:num_samples]
    
    print(f"Displaying {len(display_samples)} out of {len(samples)} matching samples\n")
    
    for i, sample in enumerate(display_samples, 1):
        print("=" * 80)
        print(f"Sample {i} (Step {sample['training_step']}, Index {sample['sample_index']})")
        print(f"Score: {sample['score']} | Ground Truth: {sample['ground_truth']}")
        print(f"Dataset: {sample['dataset']} | Finish Reason: {sample['finish_reason']}")
        print("-" * 80)
        
        # raw_queriesからプロンプトを抽出
        raw_query = sample.get('raw_queries', '').strip()
        # 空文字列やクォートのみの場合は空とみなす
        if raw_query and raw_query not in ['', '""', "''"]:
            # system promptの後の部分を探す（大文字小文字を区別しない）
            raw_query_lower = raw_query.lower()
            user_marker_pos = -1
            if 'user:' in raw_query_lower:
                user_marker_pos = raw_query_lower.find('user:')
            elif 'User:' in raw_query:
                user_marker_pos = raw_query.find('User:')
            
            if user_marker_pos >= 0:
                # "user:"の後の部分を取得（コロンとスペースをスキップ）
                prompt_start = user_marker_pos + 5  # "user:"の長さ
                # コロンの後のスペースをスキップ
                while prompt_start < len(raw_query) and raw_query[prompt_start] in [' ', ':']:
                    prompt_start += 1
                prompt = raw_query[prompt_start:].strip()
                # 最初の500文字を表示（200文字から拡張）
                if len(prompt) > 500:
                    print(f"Query:\n{prompt[:500]}...")
                else:
                    print(f"Query:\n{prompt}")
            else:
                # user:が見つからない場合、raw_query全体を表示
                if len(raw_query) > 500:
                    print(f"Query:\n{raw_query[:500]}...")
                else:
                    print(f"Query:\n{raw_query}")
        else:
            # raw_queriesが空の場合
            print("Query: (not available in raw_queries column)")
        
        print("\nResponse:")
        decoded_response = sample.get('decoded_responses', '')
        print(decoded_response)
        print("\n")


def main():
    parser = argparse.ArgumentParser(description='Display decoded responses from trace CSV')
    parser.add_argument('csv_file', type=str, help='Path to traces_flattened.csv')
    parser.add_argument('-n', '--num-samples', type=int, default=10,
                       help='Number of samples to display (default: 10)')
    parser.add_argument('-s', '--step', type=int, default=None,
                       help='Filter by training step (default: all steps)')
    parser.add_argument('--correct-only', action='store_true',
                       help='Show only correct samples (score > 0)')
    parser.add_argument('--incorrect-only', action='store_true',
                       help='Show only incorrect samples (score == 0)')
    
    args = parser.parse_args()
    
    if not Path(args.csv_file).exists():
        print(f"Error: File not found: {args.csv_file}")
        return
    
    score_filter = None
    if args.correct_only:
        score_filter = 1.0
    elif args.incorrect_only:
        score_filter = 0.0
    
    display_decoded_samples(args.csv_file, args.num_samples, args.step, score_filter)


if __name__ == '__main__':
    main()

