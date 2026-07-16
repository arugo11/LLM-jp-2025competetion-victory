from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from .config import ExperimentConfig
from .prompts import PROMPT_REVISION, QUESTION_PROMPT, SOLUTION_PROMPT


def _dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def render_dataset_card(config: ExperimentConfig, *, manifest: dict[str, Any], wandb_artifact: str) -> str:
    dataset_id = config.render(config.publication.dataset_repo)
    return f"""---
pretty_name: {dataset_id}
task_categories:
  - question-answering
language:
  - ja
tags:
  - mathematics
  - synthetic
---

# {dataset_id}

## データの概要

このデータセットは、gpt-oss-20bとgpt-oss-120bに同じ仕様とseedを与えて生成した数学問題のpairedデータです。
学習には、品質検証を通過したgpt-oss-120b由来のSFT splitとGRPO splitだけを使います。

このrepoのlicenseは未指定です。
これは無制限の利用を認める記載ではありません。

## 出典とrevision

- [gpt-oss-20b](https://huggingface.co/{config.models.generator_20b.repo_id}/tree/{config.models.generator_20b.revision})：`{config.models.generator_20b.revision}`
- [gpt-oss-120b](https://huggingface.co/{config.models.generator_120b.repo_id}/tree/{config.models.generator_120b.revision})：`{config.models.generator_120b.revision}`
- W&B Artifact：`{wandb_artifact}`

## 元repoのライセンス表示

- `openai/gpt-oss-20b@{config.models.generator_20b.revision}`：固定revisionのmodel card metadataは`apache-2.0`です。
  repoにApache License 2.0の`LICENSE`があります。
- `openai/gpt-oss-120b@{config.models.generator_120b.revision}`：固定revisionのmodel card metadataは`apache-2.0`です。
  repoにApache License 2.0の`LICENSE`があります。
- `HuggingFaceH4/aime_2024@{config.evaluation.aime_2024.revision}`
  固定revisionのdataset cardにlicense metadataはなく、standalone LICENSE fileもありません。
- `yentinglin/aime_2025@{config.evaluation.aime_2025.revision}`
  固定revisionのdataset cardにlicense metadataはなく、standalone LICENSE fileもありません。

AIME24/25はcontamination検査の参照にだけ使い、その60行をこのdataset repoへ再配布しません。
上記は各固定revisionで確認できた表示の記録であり、本repoへの新しいライセンス付与を意味しません。

## Schemaとsplit

主要schemaは `spec_id, generator_model, generator_revision, category, unit, declared_difficulty`、
`problem, solution_cot, expected_answer, generation_seed, prompt_revision, validation_status`、
`validation_votes, split, content_sha256` です。
splitはstable hashで固定し、失敗後の再配分やtop-upを行っていません。

## 生成prompt

prompt revisionは `{PROMPT_REVISION}` です。

問題生成prompt：

```text
{QUESTION_PROMPT}
```

解答生成prompt：

```text
{SOLUTION_PROMPT}
```

## 品質検証

両生成器を共通validatorとして使い、独立解とreference answerの等価性、構文、LaTeX、単一解、有限出力を検査しました。
重複候補とAIME24/25 contamination候補を抽出し、公開manifestに判定を記録しています。

検証manifest：

```json
{_dump(manifest)}
```

## 制約

検証器も言語モデルであり、数学的妥当性を完全には保証しません。
合成問題の分布は実際のAIME問題の分布と同一ではありません。
"""


def render_model_card(
    config: ExperimentConfig,
    *,
    arm: Literal["thinking", "instruct"],
    dataset_commit: str,
    aime_summary: dict[str, Any],
    merge_summary: dict[str, Any],
    wandb_artifact: str,
) -> str:
    repo_id = config.render(getattr(config.publication, f"{arm}_model_repo"))
    base = getattr(config.models, arm)
    return f"""---
base_model: {base.repo_id}
datasets:
  - {config.render(config.publication.dataset_repo)}
language:
  - ja
tags:
  - mathematics
  - sft
  - grpo
---

# {repo_id}

## モデルの概要

このモデルは、`{base.repo_id}@{base.revision}`へFull SFTを行い、その後LoRA GRPOを適用してmergeしたfull modelです。
学習データは `{config.render(config.publication.dataset_repo)}@{dataset_commit}` です。

このrepoのlicenseは未指定です。
これは無制限の利用を認める記載ではありません。

- [base model](https://huggingface.co/{base.repo_id}/tree/{base.revision})
- [gpt-oss-120b generator](https://huggingface.co/{config.models.generator_120b.repo_id}/tree/{config.models.generator_120b.revision})

## 元repoのライセンス表示

- `{base.repo_id}@{base.revision}`：固定revisionのmodel card metadataは`apache-2.0`です。
  固定revisionのrepoにはstandalone LICENSE fileはありません。
- `openai/gpt-oss-120b@{config.models.generator_120b.revision}`：固定revisionのmodel card metadataは`apache-2.0`です。
  repoにApache License 2.0の`LICENSE`があります。
- 学習dataset repo：このrepoと同様にlicenseは未指定です。

上記は各固定revisionで確認できた表示の記録であり、本repoへの新しいライセンス付与を意味しません。

## 学習設定

SFTはbf16、DeepSpeed ZeRO-3、Full-parameterで実行しました。

```json
{_dump(config.sft.model_dump(mode="json"))}
```

GRPOはoutcome rewardとしてmath-verifyだけを使いました。

```json
{_dump(config.grpo.model_dump(mode="json"))}
```

## Merge検証

```json
{_dump(merge_summary)}
```

## AIME24/25 matched評価

```json
{_dump(aime_summary)}
```

W&B Artifact：`{wandb_artifact}`

信頼区間が0を跨ぐ場合は改善傾向として扱い、統計的に有意とは主張しません。
negative resultの場合も、その結果を省略せず掲載します。

## 制約

AIME24/25は60問の小規模評価です。
本評価だけから、数学能力全般の改善や120B生成データの因果効果を主張できません。
"""


def write_card(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
