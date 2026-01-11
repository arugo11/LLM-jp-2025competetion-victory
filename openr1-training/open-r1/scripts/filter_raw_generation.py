# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "datasets>=4.4.2",
#     "marimo>=0.19.0",
#     "polar>=0.0.127",
#     "polars>=1.37.0",
#     "pyzmq>=27.1.0",
# ]
# ///

import marimo

__generated_with = "0.19.2"
app = marimo.App(width="medium")


@app.cell
def _():
    from huggingface_hub import login
    from datasets import load_dataset, Dataset
    import polars as pl
    import os
    return Dataset, load_dataset, login, os, pl


@app.cell
def _(login, os):

    hf_token = os.getenv("HF_TOKEN")
    login(token=hf_token)
    return (hf_token,)


@app.cell
def _(hf_token, load_dataset, pl):
    repo_name ="HayatoHongoEveryonesAI/qa_verify_1node_test8-TIR"
    dataset = load_dataset(
        repo_name,
        token=hf_token,
        )

    df = pl.from_arrow(dataset["train"].data.table)
    return df, repo_name


@app.cell
def _(Dataset, df, pl):
    clean_df = df.with_columns(
        pl.col("raw_generation")
          .cast(pl.Utf8)
          .str.replace_all(r"</?PYTHON>", "")
          .alias("raw_generation")
    )
    clean_ds = Dataset.from_polars(clean_df)
    return (clean_ds,)


@app.cell
def _(clean_ds, repo_name):
    clean_ds.push_to_hub(
        repo_name,
        config_name="clean",
        set_default=False,
        commit_message="Add clean config",
    )
    return


if __name__ == "__main__":
    app.run()
