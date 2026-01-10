"""Evaluate predictions on mathematical reasoning tasks.

Usage:
    uv run math-eval <prediction-file> <gold-file> [-k "20,40"] [-o <output-file>]
"""

import json
import dataclasses
from typing import Any, List, Dict, Optional

import typer
from typing_extensions import Annotated
from rich.console import Console
from rich.table import Table
from math_verify import parse, verify
from collections import Counter

app = typer.Typer()

console = Console()
err_console = Console(stderr=True)


@dataclasses.dataclass
class PredictionExample:
    id: str
    problem: str
    solution: str
    category: str
    unit: str
    evaluation_method: str
    output: str
    # JSONデータに含まれる parsed_final_answers を受け取るためのフィールドを追加
    # デフォルト値を設定して、フィールドが存在しない場合のエラーを防ぐ
    parsed_final_answers: List[str] = dataclasses.field(default_factory=list)
    # その他のフィールド（output_sample_Xなど）は **kwargs で無視するか、必要なら追加


@dataclasses.dataclass
class GoldExample:
    id: str
    problem: str
    solution: str
    category: str
    unit: str
    evaluation_method: str


def load_examples(file_path: str, example_cls: type) -> dict[str, Any]:
    """Load examples from a JSONL file."""
    id_example_map = {}
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                item = json.loads(line)
            except json.JSONDecodeError as e:
                err_console.log(f"Error decoding JSON line in '{file_path}': {e}")
                continue
            
            # dataclassのフィールド以外のキーがJSONに含まれている場合に対処するため
            # 必要なフィールドのみを抽出して初期化するロジック、あるいは
            # 単純に try-except で囲む（ここでは安全のためフィールドフィルタリングは行わずクラス生成に任せるが、
            # 実際の運用では **item で渡す際に余計な引数があるとエラーになるため注意が必要。
            # 今回のPredictionExampleは最低限の定義なので、余分なキーを無視する工夫を入れる）
            
            valid_keys = {f.name for f in dataclasses.fields(example_cls)}
            filtered_item = {k: v for k, v in item.items() if k in valid_keys}
            
            try:
                example = example_cls(**filtered_item)
            except TypeError as e:
                err_console.log(f"Error creating {example_cls.__name__} from line in '{file_path}': {e}")
                continue
                
            # parsed_final_answersを手動で補完（もし dataclass 定義外で item にある場合）
            if example_cls is PredictionExample and "parsed_final_answers" in item:
                 example.parsed_final_answers = item["parsed_final_answers"]

            if example.id in id_example_map:
                err_console.log(f"Duplicate example ID '{example.id}' found in '{file_path}'; overwriting previous entry.")
            id_example_map[example.id] = example
    return id_example_map


def check_equivalence(pred_str: str, gold_str: str) -> bool:
    """Helper to parse and verify single prediction against gold."""
    try:
        return verify(parse(pred_str), parse(gold_str))
    except Exception:
        return False


def calculate_cons_k(samples: List[str], gold: str) -> bool:
    """Calculate Consistency@k (Majority Vote)."""
    if not samples:
        print("No samples provided for Consistency@k calculation.")
        return False
    
    # math_verify の verify 関数を使って等価な回答をグループ化する
    # groups: List of {'rep': representative_string, 'count': int}
    groups = []
    
    for sample in samples:
        found_group = False
        # 既存のグループと一致するか確認
        for group in groups:
            if check_equivalence(sample, group['rep']):
                group['count'] += 1
                found_group = True
                break
        
        # 新しいグループを作成
        if not found_group:
            groups.append({'rep': sample, 'count': 1})
    
    # カウントの降順でソート
    if not groups:
        return False
        
    groups.sort(key=lambda x: x['count'], reverse=True)
    
    # 最多回答（多数決）が正解と一致するか判定
    majority_rep = groups[0]['rep']
    return check_equivalence("$$"+majority_rep+"$$", gold)

def calculate_cons_k_counter(samples: List[str], gold: str) -> bool:
    """Calculate Consistency@k (Majority Vote) using exact string match."""
    if not samples:
        return False
    
    # 文字列の完全一致で多数決を行い、最も多い回答を取得
    # most_common(1) は [(element, count)] のリストを返すので [0][0] で要素を取得
    majority_rep = Counter(samples).most_common(1)[0][0]
    
    # 選ばれた回答が正解と一致するか判定（正誤判定自体は verify を使用して柔軟に行う）
    return check_equivalence("$$"+majority_rep+"$$", gold)


def calculate_pass_k(samples: List[str], gold: str) -> bool:
    """Calculate Pass@k (Any Correct)."""
    if not samples:
        print("No samples provided for Pass@k calculation.")
        return False
    # サンプルの中に1つでも正解と一致するものがあれば True
    for sample in samples:
        if check_equivalence("$$"+sample+"$$", gold):
            return True
    return False


def accuracy(results: list[bool]) -> float:
    """Calculate accuracy from a list of boolean results."""
    return sum(results) / len(results) if results else 0.0


@app.command()
def math_eval(
    prediction_file: Annotated[str, typer.Argument(help="Path to the prediction file")],
    gold_file: Annotated[str, typer.Argument(help="Path to the gold file")],
    k_values: Annotated[
        str, typer.Option("--k-values", "-k", help="Comma separated list of k values for cons@k and pass@k (e.g., '20,40')")
    ] = None,
    output_file: Annotated[
        str, typer.Option("--output-file", "-o", help="Path to the output file")
    ] = None,
) -> None:
    """Evaluate predictions on mathematical reasoning tasks."""
    
    # K値のリストをパース
    k_list = []
    if k_values:
        try:
            k_list = [int(k.strip()) for k in k_values.split(",") if k.strip()]
        except ValueError:
            err_console.log(f"Invalid k-values format: '{k_values}'. Ignoring k metrics.")
    
    id_prediction_map = load_examples(prediction_file, PredictionExample)
    id_gold_map = load_examples(gold_file, GoldExample)

    # 結果を格納する辞書
    # 構造: {category: {metric_name: [bool, bool, ...]}}
    # metric_name は 'main', 'cons@20', 'pass@20' など
    category_results: Dict[str, Dict[str, List[bool]]] = {}
    
    # 全体の結果用
    overall_results: Dict[str, List[bool]] = {}

    # 評価ループ
    for id_ in id_gold_map:
        gold = id_gold_map[id_]
        category = gold.category
        
        # カテゴリ初期化
        if category not in category_results:
            category_results[category] = {}
        
        # 予測が存在しない場合
        if id_ not in id_prediction_map:
            err_console.log(f"Missing prediction for example ID '{id_}'; counting as incorrect.")
            # 全てのメトリクスで False を登録
            metrics_to_record = ['main'] + [f'cons@{k}' for k in k_list] + [f'pass@{k}' for k in k_list]
            for m in metrics_to_record:
                category_results[category].setdefault(m, []).append(False)
                overall_results.setdefault(m, []).append(False)
            continue

        prediction = id_prediction_map[id_]
        
        # 1. main (Main Output)
        res_pass1 = check_equivalence(prediction.output, gold.solution)
        # print(prediction.output, ",  ", gold.solution, ",  ", res_pass1)
        category_results[category].setdefault('main', []).append(res_pass1)
        overall_results.setdefault('main', []).append(res_pass1)

        # 2. cons@k, pass@k
        samples = prediction.parsed_final_answers
        for k in k_list:
            # 前方k個を取得
            current_samples = samples[:k] if samples else []
            
            # cons@k
            res_cons_k = calculate_cons_k(current_samples, gold.solution)
            category_results[category].setdefault(f'cons@{k}', []).append(res_cons_k)
            overall_results.setdefault(f'cons@{k}', []).append(res_cons_k)
            
            # pass@k
            res_pass_k = calculate_pass_k(current_samples, gold.solution)
            category_results[category].setdefault(f'pass@{k}', []).append(res_pass_k)
            overall_results.setdefault(f'pass@{k}', []).append(res_pass_k)

    # 集計とテーブル作成
    table = Table(title="Evaluation Results")
    table.add_column("Category", justify="left")
    
    # カラム定義
    metric_names = ['main']
    for k in k_list:
        metric_names.append(f'cons@{k}')
        metric_names.append(f'pass@{k}')
        
    for name in metric_names:
        table.add_column(name, justify="right")

    # データ行の追加 (各カテゴリ)
    final_metrics_data = {"overall": {}, "categories": {}}

    for category, metrics in category_results.items():
        row_data = [category]
        cat_metrics = {}
        for name in metric_names:
            acc = accuracy(metrics.get(name, []))
            row_data.append(f"{acc:.3f}")
            cat_metrics[name] = acc
        table.add_row(*row_data)
        final_metrics_data["categories"][category] = cat_metrics

    # Overall行の追加
    overall_row = ["[bold]Overall[/bold]"]
    for name in metric_names:
        acc = accuracy(overall_results.get(name, []))
        overall_row.append(f"[bold]{acc:.3f}[/bold]")
        final_metrics_data["overall"][name] = acc
    table.add_row(*overall_row)

    console.print(table)

    if output_file:
        with open(output_file, "wt", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "metrics": final_metrics_data,
                        # 詳細なTrue/Falseの結果が必要な場合はここに追加実装可能
                        # "detailed_results": ...
                    },
                    indent=2,
                    ensure_ascii=False,
                )
            )
        err_console.log(f"Results written to '{output_file}'.")

if __name__ == "__main__":
    app()