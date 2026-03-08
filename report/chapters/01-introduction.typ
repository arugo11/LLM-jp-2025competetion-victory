#import "../lib/components.typ": figure-panel, flow-step, flow-arrow

= はじめに
数理推論タスクでは，中間推論の破綻や計算ミスが最終解答に直結しやすく，大規模言語モデル（LLM）にとって依然として難易度の高い領域である．そのため，良質な Chain of Thought（CoT）データによる教師ありファインチューニング（SFT）と，推論過程をさらに最適化する強化学習の両方が重要となる．

本稿では，「第2回大規模言語モデルのファインチューニング技術と評価（FT-LLM 2026）」チューニングコンペティションの数学タスクに対する我々の取り組みを報告する．対象は，運営から提供された 80 億パラメータのベースモデル `LLM-jp-4-instruct` であり，日本語で出題される中学・高校数学問題に対する推論能力の向上を目指した．

我々は，主力 CoT データセットに対する自動審査とルールベース整形，12B 級モデルへの層拡張を伴う長文 SFT，GRPO（Group Relative Policy Optimization）による正答報酬ベースの最適化，および多数決を用いた推論システムを段階的に構築した．最終提出系では，多数決と文字列正規化を組み合わせたシンプルな推論パイプラインを採用し，補助的な試行は別途分析に留めた．

#figure(
  kind: image,
  supplement: [図],
  caption: [提案手法の全体像],
  align(center)[
    #figure-panel(
      stack(
        dir: ttb,
        spacing: 5pt,
        flow-step(
          [データキュレーション],
          [自動審査とルールベース整形で主力 CoT データを精選する．],
          rgb("F4F8FF"),
          rgb("5B8FF9"),
        ),
        flow-arrow,
        flow-step(
          [長文 SFT + GRPO],
          [12B 級へ拡張したモデルに長文 CoT と正答報酬による GRPO を適用する．],
          rgb("F8F4FF"),
          rgb("8C6FE8"),
        ),
        flow-arrow,
        flow-step(
          [多数決推論],
          [候補解答を正規化して集約し，最終解答を決定する．],
          rgb("F3FBF6"),
          rgb("58A96B"),
        ),
      ),
    )
  ],
) <fig-overview>

提案手法は，データ，学習，推論の各段階を一貫して最適化する構成であり，その全体像を @fig-overview に示す．
