import logging

import datasets
from datasets import DatasetDict, concatenate_datasets
import json

from configs import DataConfig

logger = logging.getLogger(__name__)

def get_datas_from_config(config: DataConfig, system_prompt: str = None, seed: int = 42, return_messages: bool = False):
    """
    設定オブジェクトに基づき、複数のデータセットをロードして結合する。
    最終的なカラム名は、return_messagesがFalseの場合は 'prompt' と 'completion' に、
    Trueの場合は 'messages' に統一される。

    Args:
        config (DataConfig): ロードするデータセットの情報を含む設定オブジェクト。
        system_prompt (str): システムプロンプト。
        seed (int): シャッフル用のシード。
        return_messages (bool): Trueの場合、'messages' カラム形式で返す。

    Returns:
        DatasetDict: 全てのデータセットを結合し、カラム名を整形した単一のDatasetオブジェクト。
    """
    loaded_datasets = []
    
    print("データセットのロードを開始します...")
    for i, dataset_info in enumerate(config.datasets):
        print(f"  ({i+1}/{len(config.datasets)}) ロード中: {dataset_info.name} ({dataset_info.config})")
        # 1. スライシングを指定してデータセットをロード
        # from_idとto_idが両方指定されているかチェック
        # if dataset_info.from_id is not None and dataset_info.to_id is not None:
        #     # 指定されている場合、スライスしてロード
        #     split_spec = f"{dataset_info.config}[{dataset_info.from_id - 1}:{dataset_info.to_id}]"
        #     print(f"  ({i+1}/{len(config.datasets)}) ロード中: {dataset_info.name}, スライス: {split_spec}")
        # else:
        #     # いずれかがNoneの場合、分割全体をロード
        #     split_spec = dataset_info.config
        #     print(f"  ({i+1}/{len(config.datasets)}) ロード中: {dataset_info.name}, 分割: {split_spec} (全件)")

        # 1. データセットをロード
        # 設定オブジェクトに 'split' 属性があればそれを取得、なければ None
        target_split = getattr(dataset_info, 'split', None)

        if target_split is not None:
            print(f"    -> 指定Split: {target_split} のみをロードします")
            # splitを指定してロード（戻り値は単体の Dataset オブジェクト）
            single_dataset = datasets.load_dataset(
                dataset_info.name, 
                name=dataset_info.config, 
                split=target_split
            )
            # 後続の処理が DatasetDict 形式（dataset['train']）を期待しているため、
            # 単体Datasetを 'train' キーを持つ DatasetDict にラップする
            dataset = DatasetDict({'train': single_dataset})
        else:
            # split指定がない場合は通常通りロード（戻り値は DatasetDict）
            print(f"    -> 全Splitをロードします")
            dataset = datasets.load_dataset(
                dataset_info.name, 
                name=dataset_info.config
            )
        print(dataset)
        print(f"  データセット '{dataset_info.name}' の分割: {list(dataset.keys())}")
        print(f"  データセット '{dataset_info.name}' のカラム: {list(dataset.column_names.values())}")
        print(f"  データセット '{dataset_info.name}' のサンプル数: {len(dataset['train'])} (train)")
        # print(f"  データセット '{dataset_info.name}' の最初のサンプル: {dataset['train'][0]}")

        # 2. 指定されたカラム名を一時的に 'prompt' と 'completion' に統一
        #    これにより、後続の処理を共通化できる
        temp_rename_dict = {}
        # 元の question_field が存在する場合のみリネーム対象に追加
        if dataset_info.question_field in list(dataset.column_names.values())[0]:
            temp_rename_dict[dataset_info.question_field] = 'prompt'
            print(f"  question_field '{dataset_info.question_field}' を 'prompt' にリネームしました。")
        else:
            logger.warning(f"指定された question_field '{dataset_info.question_field}' がデータセットに存在しません。")
        # 元の answer_field が存在する場合のみリネーム対象に追加
        if dataset_info.answer_field in list(dataset.column_names.values())[0]:
            temp_rename_dict[dataset_info.answer_field] = 'completion'
            print(f"  answer_field '{dataset_info.answer_field}' を 'completion' にリネームしました。")
        else:
            logger.warning(f"指定された answer_field '{dataset_info.answer_field}' がデータセットに存在しません。")

        # リネームを実行
        dataset = dataset.rename_columns(temp_rename_dict)

        current_cols = list(dataset.column_names.values())[0]
        
        if 'is_valid' in current_cols:
            print(f"  'is_valid' カラムによるフィルタリングを実行します...")
            # is_valid が 0 ではないデータを残す（つまり 0 をスキップ）
            dataset = dataset.filter(lambda example: example['is_valid'] != 0)

        if return_messages:
            # 3. 'messages' カラムに整形する関数を定義
            def format_to_messages(example):
                """
                'prompt' と 'completion' の内容から 'messages' 形式のリストを作成する。
                """
                raw_prompt = example.get('prompt', "")
                raw_completion = example.get('completion', "")
                
                # system_promptが指定されている場合、テンプレートの{question}を実際のプロンプトで埋める
                if system_prompt:
                    # 文字列型であることを保証してformat
                    final_prompt = system_prompt.format(question=str(raw_prompt))
                else:
                    final_prompt = str(raw_prompt)
                    
                # ChatML形式のmessagesを作成
                # ここではsystem messageは含めず、instructionをuser messageに含める形にしています
                messages = [
                    {"role": "user", "content": final_prompt},
                    {"role": "assistant", "content": str(raw_completion)}
                ]

                return {
                    "messages": messages
                }
            
            formatting_func = format_to_messages
        else:
            # 3. 'text' ('prompt', 'completion') カラムに整形する関数を定義
            def format_to_text_column(example):
                """
                'prompt' と 'completion' の内容から指定のJSON形式の文字列を作成する。
                """
                raw_prompt = example.get('prompt', "")
                raw_completion = example.get('completion', "")
                
                # system_promptが指定されている場合、テンプレートの{question}を実際のプロンプトで埋める
                if system_prompt:
                    # 文字列型であることを保証してformat
                    final_prompt = system_prompt.format(question=str(raw_prompt))
                else:
                    final_prompt = str(raw_prompt)
                    
                return {
                    "prompt": final_prompt,
                    "completion": str(raw_completion)
                }
            
            formatting_func = format_to_text_column

        # 4. map関数を適用して全データセットの各分割に新しいフォーマットを適用
        #    同時に、整形に使った 'prompt', 'completion' やその他不要なカラムをすべて削除
        current_columns = list(list(dataset.column_names.values())[0])
        dataset = dataset.map(formatting_func, remove_columns=current_columns)
        
        # 一番目をprint
        print(f"  データセット '{dataset_info.name}' の最初のサンプル: {dataset['train'][0]}")
        print(f"  データセット '{dataset_info.name}' のカラム: {list(dataset.column_names.values())}")

        loaded_datasets.append(dataset)

    if not loaded_datasets:
        raise ValueError("ロードできるデータセットがありませんでした。")

    print("\n全データセットを結合中...")
    # すべてのDatasetDictからキーの集合を取得
    all_keys = set(k for dd in loaded_datasets for k in dd.keys())
    print(f"結合するキー: {all_keys}")
    print(list(list(loaded_datasets[0].column_names.values())[0]))

    # 辞書内包表記を使って各キーごとにデータセットを結合
    combined_dataset = DatasetDict({
        "train" : concatenate_datasets([data["train"] for data in loaded_datasets]),
    })
    
    

    print("結合が完了しました！")
    # print(combined_dataset['train'][0])  # 最初のサンプルを表示して確認
    
    print("\nデータセットをシャッフル中...")
    # DatasetDict全体をシャッフルする。引数で受け取ったseedを使用する。
    combined_dataset = combined_dataset.shuffle(seed=seed)
    print(f"シャッフルが完了しました！ (シード: {seed})")
    
    print("最終的なデータセットの情報:")
    print(combined_dataset)
    print(combined_dataset['train'])

    # messagesキーは存在しないため、promptとcompletionの内容を表示するように変更
    sample = combined_dataset['train'][0]
    print("  最初のサンプル:", sample)
    if 'prompt' in sample and 'completion' in sample:
        print("  最初のサンプルの合計文字数:", len(sample['prompt']) + len(sample['completion']))
    if 'messages' in sample:
        print("  最初のサンプルの合計文字数:", len(sample['messages']))
    print("  サンプル数:", len(combined_dataset['train']))

    return combined_dataset