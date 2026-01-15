import glob

import pandas as pd


def read_text_file(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as file:
        content = file.read()
    return content


def read_parquet_qa(file_path: str) -> list[dict]:
    """parquetファイルからquestionとanswerを読み込む（is_valid=1のレコードのみ）"""
    df = pd.read_parquet(file_path)
    # is_validカラムが1のレコードのみをフィルタリング
    df_valid = df[df['is_valid'] == 1]
    return df_valid.to_dict("records")


def read_parquet_qa_from_directory(directory: str, pattern: str = "*.parquet") -> tuple[list[dict], list[str]]:
    """ディレクトリ内の全parquetファイルからquestionとanswerを読み込む（is_valid=1のレコードのみ）"""
    parquet_files = sorted(glob.glob(f"{directory}/{pattern}"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {directory} with pattern {pattern}")

    all_records = []
    for file_path in parquet_files:
        df = pd.read_parquet(file_path)
        # is_validカラムが1のレコードのみをフィルタリング
        df_valid = df[df['is_valid'] == 1]
        all_records.extend(df_valid.to_dict("records"))
    return all_records, parquet_files
