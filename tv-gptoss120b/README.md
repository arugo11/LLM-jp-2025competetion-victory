# tv-gptoss120b

Team Victory の小規模再現として、gpt-oss-20b / 120b による paired 数学問題生成、品質監査、難易度評価、llm-jp-4-8B の Full SFT + LoRA GRPO、matched AIME 評価、W&B/Hugging Face lineage を一つの固定 manifest で追跡するパッケージです。

本パッケージはfail-closedです。
正式な4桁実験ID、固定revision、承認参照、品質gateを満たさない実行は停止します。
GPU job内ではdataset tokenizationやHub uploadを行いません。

## 実行順序

1. `generate --smoke`で8 specsのGPU生成を検証します。
2. CPU fixtureとGPU smokeが通った後に、512 specsの`generate`を実行します。
3. `curate --generation-spec ...`で生成時の不変specを再検証してから対称validationとreview queueを作り、人手判定を与えた別のimmutable outputで品質gateを確定します。
4. `difficulty`でpaired holdoutをthinkingとinstructからN=4評価します。
5. CPU jobの`preprocess`で両armのSFT token列とGRPO prompt token列を4つのimmutable datasetへmaterializeし、schema・source/config/model/tokenizer/template・record・payload SHA-256を検証します。
6. `sft`と`grpo`は`--preprocessed`だけを受け取り、検証済みCPU artifactを消費してmerge、parity検査まで適用します。raw curated datasetや`num_proc`はGPU commandへ渡せません。
7. `evaluate`は4モデルmanifestを読み、単一job内でAIME24/25を順番に評価します。
8. `publish`はカード準備、private upload、再download検証、Artifact lineage確定、public化の順に実行します。

正式stageは`--artifact-refs`からexact W&B versionを受け取り、`artifact-receipt.json`を出力します。
W&B通信を使えない場合は`--wandb-mode offline`で論理DAGを記録し、CPU nodeから`publish --offline-journal ...`を実行してcanonical Artifactへreplayします。
`latest`は入力として受け付けません。

SFTとmerged modelにはarm、固定base revision、入力SHA-256を含むlineage fileを埋め込みます。
GRPOとmatched AIMEはこのlineageを再検証し、thinking/instructの取り違えを拒否します。
`preprocess`は`thinking-sft-preprocessed`、`thinking-grpo-preprocessed`、`instruct-sft-preprocessed`、`instruct-grpo-preprocessed`を独立したW&B dataset Artifactとして記録します。
SFTとGRPOは対応するarm/kindのArtifactだけを入力に取り、別armや別kindへの暗黙の差し替えを許しません。

## CLI

```bash
uv run --extra generation --extra tracking tv-gptoss120b generate --help
uv run --extra generation --extra data --extra tracking tv-gptoss120b curate --help
uv run --extra generation --extra data --extra tracking tv-gptoss120b difficulty --help
uv run --extra data --extra training --extra tracking tv-gptoss120b preprocess --help
uv run tv-gptoss120b render-pbs --help
uv run tv-gptoss120b verify-qsub-local --help
uv run --extra training --extra tracking tv-gptoss120b sft --help
uv run --extra training --extra tracking tv-gptoss120b grpo --help
uv run --extra generation --extra data --extra tracking tv-gptoss120b evaluate --help
uv run --extra data tv-gptoss120b summarize-evaluation --help
uv run --extra training --extra data --extra tracking tv-gptoss120b publish --help
```

`configs/experiment.yaml` は承認済みの実験ID `0399` と専用root `/groups/gcg51557/experiments/0399_tv-gptoss120b`を固定します。
既存0399の旧実験rootや成果物は参照せず、変更しません。

全テストをfresh環境で実行する場合は、テストが検証するdataset・weight guard依存も明示してから実行します。

```bash
uv sync --extra data --extra training --extra tracking --extra dev
uv run pytest -q
uv run ruff check src tests
```

## 実験・評価環境

科学条件、固定revision、matched AIME設定、実行profileはすべて`configs/experiment.yaml`を正本とします。
`runtime.stage_profile`は各stageをlocal CPU、ABCI CPU、H200のいずれかへ割り当て、CPU profileはGPU数0、H200 profileはHub upload禁止をschemaで強制します。
Pythonは3.12、package managerは`uv`に固定し、`requires-python >=3.12,<3.13`と`.python-version`の両方で3.13を拒否します。
ABCIの共有環境はYAMLの`runtime.uv_project_environment=envs/tv-gptoss120b-py312`へ分離し、各profileの`uv_extras`とthread環境変数もYAMLから取得します。
生成backendはvLLM 0.18の明示的multi-process data parallel方式を使い、1ノード内で`TP=1 × DP=8`を固定します。
各rankは連続したbalanced shardを処理し、親processが入力順へ再構成します。
20Bと120Bの間では全workerを終了・joinし、前モデルのGPU状態を次モデルへ持ち越しません。

queue、billing mode、resource type、submit account、team approvalは恒久定数ではありません。
これらは`runtime.transient_cluster_fields`として明示し、qsubの60分以内にSlackとlive stateから作るpolicy snapshotでのみ解決します。
`cpu_publication.venue`も`decision_required`のままにし、転送量・所要時間・ファイル数を測る前にlocal/ABCIを決めません。

`coordination`は値の根拠を`user-fixed`、`meeting-verified`、`slack-confirmed`、`unresolved`に分離します。
Slack上ではAIME/LiveCodeBenchに`swallow-evaluation-instruct`を使うことだけが確認され、共通のend-to-end手順は未標準化です。
AIME25 splitはユーザー固定の`train`を維持し、議事録の`test`表記との不一致を未解決事項として残しています。

## 公開条件

新しいHugging Face dataset/modelのYAML metadataには、`license`、`license_name`、`license_link`を記載しません。
カード本文には「このrepoのlicenseは未指定」と記載します。
private commitを別ディレクトリへ再downloadし、checksum、dataset viewerまたはfull model loadを確認するまでpublicへ変更しません。
final publication manifestはrelease auditとともにW&B `publication-manifest:vN`へ格納し、public化直前にremote内容を再downloadしてcanonical digestを照合します。

ABCI job manifestとPBSは、60分以内のpolicy snapshot、ユーザー承認、live stateを共通preflightへ渡して`PASS`を得た場合だけsubmitできます。
本実験には原則としてチーム承認を要求します。
ただし、Slackで明示された「予約ノードが空いていれば非申請者jobも投入可」というbest-effort運用に限り、30分以下・1ノード・単一smoke・preemptible・自動retryなしをmanifestとpolicy snapshotの双方で固定したjobだけを狭い例外とします。
この例外jobは混雑時に予告なくkillされ得るため、killを成功扱いにせず、追加投入もしません。
job manifestは承認済みplan SHA-256へ固定し、過去jobの累積node-hoursは手書き値を受け付けず、保存した`qstat -fx -F json`の実walltimeとnode数から投入前・release時の両方で再計算します。
`queue`、`rate_class`、`resource_type`に加え、`cpus_per_node`と`gpus_per_node`もfresh policy snapshotのscheduler factsへ一致しなければPBSを生成しません。
初回qsub前のstorage判定は`scripts/abci_pre_qsub_storage_audit.sh`による10万inode以下・30秒以内のbounded auditを使い、現在量とjobの出力上限の合計を250 GB制限へ照合します。
このbounded auditをrelease証拠へ昇格させることはできません。各job後とrelease時のlineageは、canonical EXP_DIR内に保存したPBS-side complete deep auditだけからbytesとinodeを取得します。
本CLI自体はqsubを実行しません。
`render-pbs`はmanifestとfresh policyからimmutable PBSを生成するだけで、`verify-qsub-local`のPASS後も共通cluster preflightのPASSが別途必要です。
