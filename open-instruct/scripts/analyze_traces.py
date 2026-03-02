#!/usr/bin/env python3
"""
トレースファイルを読み込んで分析・ダウンロード用に変換するスクリプト

Usage:
    python scripts/analyze_traces.py <trace_file_path> [--output-dir <output_dir>] [--format <csv|json|both>]
"""

import json
import argparse
import os
import csv
from pathlib import Path
from typing import List, Dict, Any


def load_traces(trace_file_path: str) -> List[Dict[str, Any]]:
    """トレースファイルを読み込む"""
    print(f"Loading traces from: {trace_file_path}")
    traces = []
    
    if not os.path.exists(trace_file_path):
        raise FileNotFoundError(f"Trace file not found: {trace_file_path}")
    
    with open(trace_file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                trace = json.loads(line.strip())
                traces.append(trace)
            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse line {line_num}: {e}")
                continue
    
    print(f"Loaded {len(traces)} training steps")
    return traces


def extract_trace_statistics(traces: List[Dict[str, Any]]) -> Dict[str, Any]:
    """トレースから統計情報を抽出"""
    stats = {
        'total_steps': len(traces),
        'steps': [],
        'average_scores': [],
        'solve_rates': [],
        'total_samples': [],
    }
    
    for trace in traces:
        step = trace.get('training_step', 0)
        scores = trace.get('scores', [])
        
        if len(scores) > 0:
            avg_score = sum(scores) / len(scores)
            solve_rate = sum(1 for s in scores if s > 0) / len(scores)
            max_score = max(scores) if scores else 0
        else:
            avg_score = 0
            solve_rate = 0
            max_score = 0
        
        stats['steps'].append(step)
        stats['average_scores'].append(avg_score)
        stats['solve_rates'].append(solve_rate)
        stats['total_samples'].append(len(scores))
    
    return stats


def flatten_traces(traces: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """トレースをフラット化して、各サンプルを1行に展開"""
    flattened = []
    
    for trace in traces:
        training_step = trace.get('training_step', 0)
        scores = trace.get('scores', [])
        responses = trace.get('responses', [])
        queries = trace.get('queries', [])
        ground_truths = trace.get('ground_truths', [])
        datasets = trace.get('datasets', [])
        finish_reasons = trace.get('finish_reasons', [])
        
        # 各サンプルを展開
        num_samples = max(len(scores), len(responses), len(queries))
        
        for i in range(num_samples):
            row = {
                'training_step': training_step,
                'sample_index': i,
                'score': scores[i] if i < len(scores) else None,
                'response': responses[i] if i < len(responses) else None,
                'query': queries[i] if i < len(queries) else None,
                'ground_truth': ground_truths[i] if i < len(ground_truths) else None,
                'dataset': datasets[i] if i < len(datasets) else None,
                'finish_reason': finish_reasons[i] if i < len(finish_reasons) else None,
            }
            
            # その他のメトリクスも追加
            for key, value in trace.items():
                if key not in ['scores', 'responses', 'queries', 'ground_truths', 
                              'datasets', 'finish_reasons', 'training_step']:
                    if isinstance(value, (int, float, str, bool, type(None))):
                        row[key] = value
                    elif isinstance(value, list) and len(value) > i:
                        row[key] = value[i]
            
            flattened.append(row)
    
    return flattened


def escape_csv_value(value):
    """CSVの値をエスケープ"""
    if value is None:
        return ''
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, bool):
        return str(value)
    # 文字列の場合、改行やカンマをエスケープ
    s = str(value)
    if '\n' in s or ',' in s or '"' in s:
        return '"' + s.replace('"', '""') + '"'
    return s


def save_as_csv(flattened_traces: List[Dict[str, Any]], output_path: str):
    """フラット化されたトレースをCSV形式で保存"""
    print(f"Saving CSV to: {output_path}")
    
    if len(flattened_traces) == 0:
        print("Warning: No data to save")
        return
    
    # すべてのキーを収集
    all_keys = set()
    for row in flattened_traces:
        all_keys.update(row.keys())
    
    # キーをソート（主要なキーを先に）
    priority_keys = ['training_step', 'sample_index', 'score', 'query', 'response', 
                     'ground_truth', 'dataset', 'finish_reason']
    ordered_keys = [k for k in priority_keys if k in all_keys]
    ordered_keys.extend(sorted(all_keys - set(priority_keys)))
    
    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=ordered_keys)
        writer.writeheader()
        
        for row in flattened_traces:
            # 値をエスケープ
            escaped_row = {k: escape_csv_value(row.get(k)) for k in ordered_keys}
            writer.writerow(escaped_row)
    
    print(f"CSV saved: {len(flattened_traces)} rows, {len(ordered_keys)} columns")


def save_as_json(flattened_traces: List[Dict[str, Any]], output_path: str):
    """フラット化されたトレースをJSON形式で保存"""
    print(f"Saving JSON to: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(flattened_traces, f, ensure_ascii=False, indent=2)
    print(f"JSON saved: {len(flattened_traces)} records")


def save_statistics(stats: Dict[str, Any], output_path: str):
    """統計情報を保存"""
    print(f"Saving statistics to: {output_path}")
    
    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['training_step', 'average_score', 'solve_rate', 'total_samples'])
        writer.writeheader()
        
        for i in range(len(stats['steps'])):
            writer.writerow({
                'training_step': stats['steps'][i],
                'average_score': stats['average_scores'][i],
                'solve_rate': stats['solve_rates'][i],
                'total_samples': stats['total_samples'][i],
            })
    
    print(f"Statistics saved: {len(stats['steps'])} steps")
    
    # サマリーも出力
    print("\n=== Summary Statistics ===")
    print(f"Total training steps: {stats['total_steps']}")
    if stats['average_scores']:
        print(f"Overall average score: {sum(stats['average_scores']) / len(stats['average_scores']):.4f}")
        print(f"Overall solve rate: {sum(stats['solve_rates']) / len(stats['solve_rates']):.4f}")
        print(f"Total samples processed: {sum(stats['total_samples'])}")


def main():
    parser = argparse.ArgumentParser(description='Analyze and export trace files')
    parser.add_argument('trace_file', type=str, help='Path to trace JSONL file')
    parser.add_argument('--output-dir', type=str, default='.', 
                       help='Output directory for converted files (default: current directory)')
    parser.add_argument('--format', type=str, choices=['csv', 'json', 'both'], default='both',
                       help='Output format: csv, json, or both (default: both)')
    parser.add_argument('--prefix', type=str, default='traces',
                       help='Prefix for output files (default: traces)')
    
    args = parser.parse_args()
    
    # トレースファイルを読み込む
    traces = load_traces(args.trace_file)
    
    if len(traces) == 0:
        print("No traces found in file!")
        return
    
    # 出力ディレクトリを作成
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ファイル名を取得
    trace_file_name = Path(args.trace_file).stem
    
    # 統計情報を抽出
    stats = extract_trace_statistics(traces)
    stats_file = output_dir / f"{args.prefix}_statistics.csv"
    save_statistics(stats, str(stats_file))
    
    # トレースをフラット化
    flattened_traces = flatten_traces(traces)
    
    # 指定された形式で保存
    if args.format in ['csv', 'both']:
        csv_file = output_dir / f"{args.prefix}_flattened.csv"
        save_as_csv(flattened_traces, str(csv_file))
    
    if args.format in ['json', 'both']:
        json_file = output_dir / f"{args.prefix}_flattened.json"
        save_as_json(flattened_traces, str(json_file))
    
    print(f"\n✅ Analysis complete! Files saved to: {output_dir}")


if __name__ == '__main__':
    main()

