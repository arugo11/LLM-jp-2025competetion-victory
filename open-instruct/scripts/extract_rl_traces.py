#!/usr/bin/env python3
"""
RLトレース抽出スクリプト
save traceファイルからRLトレースを取得します。

Usage:
    python scripts/extract_rl_traces.py <trace_file> [options]
"""

import json
import argparse
import csv
import statistics
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

# pandasとnumpyはオプショナル（CSV形式や統計情報が必要な場合のみ）
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


def extract_trace_id(trace_file: str) -> Optional[str]:
    """
    トレースファイルのパスからIDを抽出
    
    Args:
        trace_file: トレースファイルのパス（例: output/grpo_fast__3__1769303402/traces_grpo_fast__3__1769303402.jsonl）
        
    Returns:
        抽出されたID（見つからない場合はNone）
    """
    trace_path = Path(trace_file)
    
    # ファイル名から抽出を試みる（traces_grpo_fast__3__1769303402.jsonl -> grpo_fast__3__1769303402）
    stem = trace_path.stem
    if stem.startswith("traces_"):
        # traces_プレフィックスを削除
        id_part = stem[7:]  # "traces_"の長さは7
        if id_part:
            return id_part
    
    # ディレクトリ名から抽出（output/grpo_fast__3__1769303402/...）
    parent_dir = trace_path.parent.name
    if parent_dir and parent_dir != "." and "__" in parent_dir:
        # __を含むディレクトリ名はIDの可能性が高い
        return parent_dir
    
    return None


def load_rl_traces(trace_file: str) -> List[Dict[str, Any]]:
    """
    JSONL形式のトレースファイルからRLトレースを読み込む
    
    Args:
        trace_file: トレースファイルのパス
        
    Returns:
        RLトレースのリスト
    """
    traces = []
    print(f"トレースファイルを読み込み中: {trace_file}")
    
    if not Path(trace_file).exists():
        raise FileNotFoundError(f"トレースファイルが見つかりません: {trace_file}")
    
    with open(trace_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                trace = json.loads(line.strip())
                traces.append(trace)
            except json.JSONDecodeError as e:
                print(f"警告: 行 {line_num} のパースに失敗しました: {e}")
                continue
    
    print(f"読み込んだトレース数: {len(traces)}")
    return traces


def extract_rl_traces(traces: List[Dict[str, Any]], 
                     filter_step: int = None,
                     min_score: float = None,
                     max_score: float = None,
                     max_samples: int = None,
                     human_readable: bool = False) -> List[Dict[str, Any]]:
    """
    RLトレースから必要な情報を抽出
    
    Args:
        traces: トレースデータのリスト
        filter_step: 特定のトレーニングステップのみ抽出（Noneの場合は全て）
        min_score: 最小スコアでフィルタリング（Noneの場合は全て）
        max_score: 最大スコアでフィルタリング（Noneの場合は全て）
        max_samples: 最大サンプル数（Noneの場合は全て）
        human_readable: 人間が読める形式（トークンIDを除外、デコード済みテキストのみ）
        
    Returns:
        抽出されたRLトレースのリスト
    """
    extracted_traces = []
    total_samples_collected = 0
    
    for trace in traces:
        # トレーニングステップでフィルタリング
        if filter_step is not None:
            if trace.get('training_step') != filter_step:
                continue
        
        # スコアでフィルタリング
        scores = trace.get('scores', [])
        if min_score is not None or max_score is not None:
            if not scores:
                continue
            if min_score is not None and not any(s >= min_score for s in scores):
                continue
            if max_score is not None and not any(s <= max_score for s in scores):
                continue
        
        # サンプル数を制限
        num_samples_in_trace = len(scores) if scores else 0
        if max_samples is not None:
            remaining_samples = max_samples - total_samples_collected
            if remaining_samples <= 0:
                break
            
            if num_samples_in_trace > remaining_samples:
                # このトレースのサンプルを制限
                if human_readable:
                    rl_trace = {
                        'training_step': trace.get('training_step'),
                        'scores': scores[:remaining_samples],
                        'finish_reasons': trace.get('finish_reasons', [])[:remaining_samples],
                        'queries': trace.get('raw_queries', [])[:remaining_samples],  # デコード済みテキストを使用
                        'responses': trace.get('decoded_responses', [])[:remaining_samples],  # デコード済みテキストを使用
                        'datasets': trace.get('datasets', [])[:remaining_samples],
                    }
                    # ground_truthsがデコード済みかどうか確認
                    ground_truths = trace.get('ground_truths', [])
                    if ground_truths and len(ground_truths) > 0 and isinstance(ground_truths[0], str):
                        rl_trace['ground_truths'] = ground_truths[:remaining_samples]
                    else:
                        rl_trace['ground_truths'] = []
                else:
                    rl_trace = {
                        'training_step': trace.get('training_step'),
                        'scores': scores[:remaining_samples],
                        'finish_reasons': trace.get('finish_reasons', [])[:remaining_samples],
                        'responses': trace.get('responses', [])[:remaining_samples],
                        'queries': trace.get('queries', [])[:remaining_samples],
                        'ground_truths': trace.get('ground_truths', [])[:remaining_samples],
                        'datasets': trace.get('datasets', [])[:remaining_samples],
                        'raw_queries': trace.get('raw_queries', [])[:remaining_samples],
                        'decoded_responses': trace.get('decoded_responses', [])[:remaining_samples],
                        'indices': trace.get('indices', [])[:remaining_samples] if trace.get('indices') else [],
                    }
                total_samples_collected = max_samples
            else:
                # このトレース全体を含める
                if human_readable:
                    rl_trace = {
                        'training_step': trace.get('training_step'),
                        'scores': trace.get('scores', []),
                        'finish_reasons': trace.get('finish_reasons', []),
                        'queries': trace.get('raw_queries', []),  # デコード済みテキストを使用
                        'responses': trace.get('decoded_responses', []),  # デコード済みテキストを使用
                        'datasets': trace.get('datasets', []),
                    }
                    # ground_truthsがデコード済みかどうか確認
                    ground_truths = trace.get('ground_truths', [])
                    if ground_truths and len(ground_truths) > 0 and isinstance(ground_truths[0], str):
                        rl_trace['ground_truths'] = ground_truths
                    else:
                        rl_trace['ground_truths'] = []
                else:
                    rl_trace = {
                        'training_step': trace.get('training_step'),
                        'scores': trace.get('scores', []),
                        'finish_reasons': trace.get('finish_reasons', []),
                        'responses': trace.get('responses', []),
                        'queries': trace.get('queries', []),
                        'ground_truths': trace.get('ground_truths', []),
                        'datasets': trace.get('datasets', []),
                        'raw_queries': trace.get('raw_queries', []),
                        'decoded_responses': trace.get('decoded_responses', []),
                        'indices': trace.get('indices', []),
                    }
                total_samples_collected += num_samples_in_trace
        else:
            # サンプル数制限なし
            if human_readable:
                rl_trace = {
                    'training_step': trace.get('training_step'),
                    'scores': trace.get('scores', []),
                    'finish_reasons': trace.get('finish_reasons', []),
                    'queries': trace.get('raw_queries', []),  # デコード済みテキストを使用
                    'responses': trace.get('decoded_responses', []),  # デコード済みテキストを使用
                    'datasets': trace.get('datasets', []),
                }
                # ground_truthsがデコード済みかどうか確認
                ground_truths = trace.get('ground_truths', [])
                if ground_truths and len(ground_truths) > 0 and isinstance(ground_truths[0], str):
                    rl_trace['ground_truths'] = ground_truths
                else:
                    rl_trace['ground_truths'] = []
            else:
                rl_trace = {
                    'training_step': trace.get('training_step'),
                    'scores': trace.get('scores', []),
                    'finish_reasons': trace.get('finish_reasons', []),
                    'responses': trace.get('responses', []),
                    'queries': trace.get('queries', []),
                    'ground_truths': trace.get('ground_truths', []),
                    'datasets': trace.get('datasets', []),
                    'raw_queries': trace.get('raw_queries', []),
                    'decoded_responses': trace.get('decoded_responses', []),
                    'indices': trace.get('indices', []),
                }
        
        # reward_metricsを追加
        reward_metrics = trace.get('reward_metrics', {})
        if reward_metrics:
            for k, v in reward_metrics.items():
                rl_trace[f'reward_{k}'] = v
        
        # batch情報があれば追加（human_readableの場合は除外）
        if not human_readable:
            for key in ['queries', 'ground_truths', 'datasets', 'raw_queries', 
                       'decoded_responses', 'indices', 'scores']:
                if key in trace and key not in rl_trace:
                    rl_trace[key] = trace[key]
        
        extracted_traces.append(rl_trace)
        
        # サンプル数制限に達した場合は終了
        if max_samples is not None and total_samples_collected >= max_samples:
            break
    
    return extracted_traces


def print_trace_summary(traces: List[Dict[str, Any]], max_samples: int = 5):
    """トレースの簡潔なサマリーを表示（標準出力用）"""
    print(f"\n=== RLトレースサマリー ({len(traces)}ステップ) ===\n")
    
    for idx, trace in enumerate(traces[:max_samples]):
        step = trace.get('training_step', '?')
        scores = trace.get('scores', [])
        decoded_responses = trace.get('decoded_responses', [])
        raw_queries = trace.get('raw_queries', [])
        datasets = trace.get('datasets', [])
        
        print(f"ステップ {step}:")
        if scores:
            avg_score = statistics.mean(scores) if len(scores) > 0 else 0
            correct_count = sum(1 for s in scores if s == 10.0)
            print(f"  平均スコア: {avg_score:.2f}, 正解数: {correct_count}/{len(scores)}")
        
        # 最初の数件のサンプルを表示
        num_samples_to_show = min(3, len(scores))
        for i in range(num_samples_to_show):
            print(f"\n  サンプル {i+1}:")
            if raw_queries and i < len(raw_queries):
                query_preview = raw_queries[i][:200] + "..." if len(raw_queries[i]) > 200 else raw_queries[i]
                print(f"    クエリ: {query_preview}")
            if decoded_responses and i < len(decoded_responses):
                response_preview = decoded_responses[i][:200] + "..." if len(decoded_responses[i]) > 200 else decoded_responses[i]
                print(f"    レスポンス: {response_preview}")
            if scores and i < len(scores):
                print(f"    スコア: {scores[i]}")
            if datasets and i < len(datasets):
                print(f"    データセット: {datasets[i]}")
        
        if len(scores) > num_samples_to_show:
            print(f"  ... 他 {len(scores) - num_samples_to_show} サンプル")
        print()
    
    if len(traces) > max_samples:
        print(f"... 他 {len(traces) - max_samples} ステップ\n")


def save_traces(traces: List[Dict[str, Any]], 
                output_format: str,
                output_file: str = None,
                stdout: bool = False,
                human_readable: bool = False,
                trace_file: str = None):
    """
    トレースを指定された形式で保存
    
    Args:
        traces: トレースデータのリスト
        output_format: 出力形式 ('jsonl', 'json', 'csv')
        output_file: 出力ファイルパス（Noneの場合は自動生成）
        stdout: 標準出力に出力するか（デフォルト: False）
        human_readable: 人間が読める形式かどうか（ファイル名に反映）
        trace_file: トレースファイルのパス（ID抽出用）
    """
    if len(traces) == 0:
        print("警告: 保存するトレースがありません")
        return
    
    if stdout:
        # 標準出力の場合は簡潔なサマリーを表示
        print_trace_summary(traces)
        return
    
    # ファイルに出力（output_fileが指定されていない場合は自動生成）
    if output_file is None:
        # トレースファイル名から自動生成
        base_name = "rl_traces"
        
        # IDを抽出して追加
        if trace_file:
            trace_id = extract_trace_id(trace_file)
            if trace_id:
                base_name += f"_{trace_id}"
        
        # サンプル数を追加
        total_samples = sum(len(trace.get('scores', [])) for trace in traces)
        if total_samples > 0:
            base_name += f"_{total_samples}"
        
        if human_readable:
            base_name += "_human_readable"
        
        output_file = f"{base_name}.{output_format}"
    elif human_readable:
        # 既存のファイル名に_human_readableを追加
        output_path = Path(output_file)
        stem = output_path.stem
        suffix = output_path.suffix
        if not stem.endswith("_human_readable"):
            output_file = str(output_path.parent / f"{stem}_human_readable{suffix}")
    
    if output_format == 'jsonl':
        output_path = Path(output_file)
        with open(output_path, 'w', encoding='utf-8') as f:
            for trace in traces:
                f.write(json.dumps(trace, ensure_ascii=False) + '\n')
        print(f"JSONL形式で保存しました: {output_path}")
    
    elif output_format == 'json':
        output_path = Path(output_file)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(traces, f, ensure_ascii=False, indent=2)
        print(f"JSON形式で保存しました: {output_path}")
    
    elif output_format == 'csv':
        if stdout:
            print("警告: CSV形式は標準出力には出力できません。ファイルに保存します。")
            if output_file is None:
                output_file = "rl_traces.csv"
        if not HAS_PANDAS:
            # pandasがない場合は標準ライブラリのcsvモジュールを使用
            save_traces_csv_stdlib(traces, output_file)
        else:
            # pandasがある場合はpandasを使用
            save_traces_csv_pandas(traces, output_file)


def save_traces_csv_stdlib(traces: List[Dict[str, Any]], output_file: Optional[str] = None):
    """標準ライブラリのcsvモジュールを使用してCSV形式で保存"""
    rows = []
    for trace in traces:
        training_step = trace.get('training_step', '')
        scores = trace.get('scores', [])
        responses = trace.get('responses', [])
        queries = trace.get('queries', [])
        raw_queries = trace.get('raw_queries', [])
        decoded_responses = trace.get('decoded_responses', [])
        ground_truths = trace.get('ground_truths', [])
        datasets = trace.get('datasets', [])
        finish_reasons = trace.get('finish_reasons', [])
        
        # 各レスポンスごとに1行を作成
        num_responses = max(len(scores), len(responses), len(queries))
        for i in range(num_responses):
            row = {
                'training_step': training_step,
                'response_index': i,
                'score': scores[i] if i < len(scores) else '',
                'finish_reason': finish_reasons[i] if i < len(finish_reasons) else '',
                'response': str(responses[i]) if i < len(responses) else '',
                'decoded_response': decoded_responses[i] if i < len(decoded_responses) else '',
                'query': str(queries[i]) if i < len(queries) else '',
                'raw_query': raw_queries[i] if i < len(raw_queries) else '',
                'ground_truth': str(ground_truths[i]) if i < len(ground_truths) else '',
                'dataset': datasets[i] if i < len(datasets) else '',
            }
            
            # reward_metricsを追加
            for key, value in trace.items():
                if key.startswith('reward_'):
                    row[key] = value
            
            rows.append(row)
    
    if not rows:
        print("警告: CSV形式で保存するデータがありません")
        return
    
    # すべてのキーを収集
    all_keys = set()
    for row in rows:
        all_keys.update(row.keys())
    
    # キーをソート（主要なキーを先に）
    priority_keys = ['training_step', 'response_index', 'score', 'query', 'response', 
                     'ground_truth', 'dataset', 'finish_reason', 'decoded_response', 'raw_query']
    ordered_keys = [k for k in priority_keys if k in all_keys]
    ordered_keys.extend(sorted(all_keys - set(priority_keys)))
    
    if output_file:
        with open(output_file, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=ordered_keys)
            writer.writeheader()
            writer.writerows(rows)
        print(f"CSV形式で保存しました: {output_file} ({len(rows)}行)")
    else:
        writer = csv.DictWriter(sys.stdout, fieldnames=ordered_keys)
        writer.writeheader()
        writer.writerows(rows)


def save_traces_csv_pandas(traces: List[Dict[str, Any]], output_file: Optional[str] = None):
    """pandasを使用してCSV形式で保存"""
    rows = []
    for trace in traces:
        training_step = trace.get('training_step', '')
        scores = trace.get('scores', [])
        responses = trace.get('responses', [])
        queries = trace.get('queries', [])
        raw_queries = trace.get('raw_queries', [])
        decoded_responses = trace.get('decoded_responses', [])
        ground_truths = trace.get('ground_truths', [])
        datasets = trace.get('datasets', [])
        finish_reasons = trace.get('finish_reasons', [])
        
        # 各レスポンスごとに1行を作成
        num_responses = max(len(scores), len(responses), len(queries))
        for i in range(num_responses):
            row = {
                'training_step': training_step,
                'response_index': i,
                'score': scores[i] if i < len(scores) else None,
                'finish_reason': finish_reasons[i] if i < len(finish_reasons) else None,
                'response': responses[i] if i < len(responses) else '',
                'decoded_response': decoded_responses[i] if i < len(decoded_responses) else '',
                'query': queries[i] if i < len(queries) else '',
                'raw_query': raw_queries[i] if i < len(raw_queries) else '',
                'ground_truth': ground_truths[i] if i < len(ground_truths) else '',
                'dataset': datasets[i] if i < len(datasets) else '',
            }
            
            # reward_metricsを追加
            for key, value in trace.items():
                if key.startswith('reward_'):
                    row[key] = value
            
            rows.append(row)
    
    df = pd.DataFrame(rows)
    if output_file:
        df.to_csv(output_file, index=False, encoding='utf-8')
        print(f"CSV形式で保存しました: {output_file} ({len(rows)}行)")
    else:
        print(df.to_csv(index=False))


def print_statistics(traces: List[Dict[str, Any]]):
    """統計情報を表示"""
    if not traces:
        print("統計情報を表示するデータがありません")
        return
    
    all_scores = []
    all_steps = []
    
    for trace in traces:
        scores = trace.get('scores', [])
        step = trace.get('training_step', 0)
        all_scores.extend(scores)
        all_steps.append(step)
    
    if not all_scores:
        print("スコアデータがありません")
        return
    
    # numpyがある場合はnumpyを使用、ない場合は標準ライブラリを使用
    if HAS_NUMPY:
        mean_score = np.mean(all_scores)
        min_score = np.min(all_scores)
        max_score = np.max(all_scores)
        std_score = np.std(all_scores)
    else:
        mean_score = statistics.mean(all_scores)
        min_score = min(all_scores)
        max_score = max(all_scores)
        std_score = statistics.stdev(all_scores) if len(all_scores) > 1 else 0.0
    
    print("\n=== RLトレース統計情報 ===")
    print(f"総トレーニングステップ数: {len(set(all_steps))}")
    print(f"総レスポンス数: {len(all_scores)}")
    print(f"平均スコア: {mean_score:.4f}")
    print(f"最小スコア: {min_score:.2f}")
    print(f"最大スコア: {max_score:.2f}")
    print(f"スコア標準偏差: {std_score:.4f}")
    print(f"スコア10.0の数: {sum(1 for s in all_scores if s == 10.0)} ({sum(1 for s in all_scores if s == 10.0)/len(all_scores)*100:.2f}%)")
    print(f"スコア0.0の数: {sum(1 for s in all_scores if s == 0.0)} ({sum(1 for s in all_scores if s == 0.0)/len(all_scores)*100:.2f}%)")
    
    # ステップごとの統計
    if len(set(all_steps)) > 1:
        print("\n=== ステップごとの統計 ===")
        step_stats = {}
        for trace in traces:
            step = trace.get('training_step', 0)
            scores = trace.get('scores', [])
            if step not in step_stats:
                step_stats[step] = []
            step_stats[step].extend(scores)
        
        for step in sorted(step_stats.keys()):
            scores = step_stats[step]
            if HAS_NUMPY:
                mean = np.mean(scores)
            else:
                mean = statistics.mean(scores)
            print(f"ステップ {step}: 平均={mean:.4f}, "
                  f"正解率={sum(1 for s in scores if s == 10.0)/len(scores)*100:.2f}%, "
                  f"サンプル数={len(scores)}")


def main():
    parser = argparse.ArgumentParser(
        description='save traceファイルからRLトレースを取得します',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  # 基本的な使用方法（デフォルトでrl_traces.jsonlに保存）
  python scripts/extract_rl_traces.py output/grpo_fast__3__1769303402/traces_grpo_fast__3__1769303402.jsonl

  # 標準出力に簡潔なサマリーを表示（ファイルには保存しない）
  python scripts/extract_rl_traces.py output/grpo_fast__3__1769303402/traces_grpo_fast__3__1769303402.jsonl --stdout

  # JSON形式でファイルに保存
  python scripts/extract_rl_traces.py output/grpo_fast__3__1769303402/traces_grpo_fast__3__1769303402.jsonl -o rl_traces.json -f json

  # CSV形式で保存（pandasがインストールされていない場合は標準ライブラリを使用）
  python scripts/extract_rl_traces.py output/grpo_fast__3__1769303402/traces_grpo_fast__3__1769303402.jsonl -o rl_traces.csv -f csv

  # 特定のトレーニングステップのみ抽出
  python scripts/extract_rl_traces.py output/grpo_fast__3__1769303402/traces_grpo_fast__3__1769303402.jsonl --step 100

  # 最小スコアでフィルタリング（スコア10.0以上のもののみ）
  python scripts/extract_rl_traces.py output/grpo_fast__3__1769303402/traces_grpo_fast__3__1769303402.jsonl --min-score 10.0

  # 統計情報を表示
  python scripts/extract_rl_traces.py output/grpo_fast__3__1769303402/traces_grpo_fast__3__1769303402.jsonl --stats

  # 最初の1000サンプルのみ保存
  python scripts/extract_rl_traces.py output/grpo_fast__3__1769303402/traces_grpo_fast__3__1769303402.jsonl --max-samples 1000

  # 人間が読める形式で保存（トークンIDを除外、ファイル名に_human_readableが追加される）
  python scripts/extract_rl_traces.py output/grpo_fast__3__1769303402/traces_grpo_fast__3__1769303402.jsonl --max-samples 1000 -o rl_traces_1000.json -f json --human-readable
        """
    )
    parser.add_argument(
        'trace_file',
        type=str,
        help='トレースファイルのパス'
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        default=None,
        help='出力ファイルパス（指定しない場合は自動生成: rl_traces.{format}）'
    )
    parser.add_argument(
        '--stdout',
        action='store_true',
        help='標準出力に簡潔なサマリーを表示（ファイルには保存しない）'
    )
    parser.add_argument(
        '-f', '--format',
        type=str,
        choices=['jsonl', 'json', 'csv'],
        default='jsonl',
        help='出力形式 (default: jsonl)'
    )
    parser.add_argument(
        '--step',
        type=int,
        default=None,
        help='特定のトレーニングステップのみ抽出'
    )
    parser.add_argument(
        '--min-score',
        type=float,
        default=None,
        help='最小スコアでフィルタリング'
    )
    parser.add_argument(
        '--max-score',
        type=float,
        default=None,
        help='最大スコアでフィルタリング'
    )
    parser.add_argument(
        '--stats',
        action='store_true',
        help='統計情報を表示'
    )
    parser.add_argument(
        '--max-samples',
        type=int,
        default=None,
        help='最大サンプル数（例: 1000）'
    )
    parser.add_argument(
        '--human-readable',
        action='store_true',
        help='人間が読める形式で保存（トークンIDを除外、デコード済みテキストのみ）'
    )
    
    args = parser.parse_args()
    
    # CSV形式でpandasが必要な場合の警告
    if args.format == 'csv' and not HAS_PANDAS:
        print("注意: pandasがインストールされていません。標準ライブラリのcsvモジュールを使用します。")
    
    # トレースファイルを読み込む
    try:
        traces = load_rl_traces(args.trace_file)
    except FileNotFoundError as e:
        print(f"エラー: {e}")
        return
    except Exception as e:
        print(f"エラー: トレースファイルの読み込みに失敗しました: {e}")
        return
    
    if len(traces) == 0:
        print("警告: トレースファイルにデータがありません")
        return
    
    # RLトレースを抽出
    extracted_traces = extract_rl_traces(
        traces,
        filter_step=args.step,
        min_score=args.min_score,
        max_score=args.max_score,
        max_samples=args.max_samples,
        human_readable=args.human_readable
    )
    
    # サンプル数をカウント
    total_samples = sum(len(trace.get('scores', [])) for trace in extracted_traces)
    print(f"抽出されたトレース数: {len(extracted_traces)}")
    if args.max_samples:
        print(f"抽出されたサンプル数: {total_samples} (最大: {args.max_samples})")
    else:
        print(f"抽出されたサンプル数: {total_samples}")
    
    if args.human_readable:
        print("人間が読める形式で保存します（トークンIDを除外）")
    
    # 統計情報を表示
    if args.stats:
        print_statistics(extracted_traces)
    
    # トレースを保存
    if extracted_traces:
        try:
            save_traces(extracted_traces, args.format, args.output, args.stdout, args.human_readable, args.trace_file)
        except Exception as e:
            print(f"エラー: トレースの保存に失敗しました: {e}")
            return
    else:
        print("警告: 抽出されたトレースがありません。フィルタ条件を緩和してください。")


if __name__ == '__main__':
    main()

