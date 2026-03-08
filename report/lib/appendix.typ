#import "../nlp2026.typ": paragraph

#let appendix-table(size: 8.7pt, body) = block(above: 0.6em, below: 0.75em)[
  #set text(size: size)
  #body
]

#let centered-table(body) = block(width: 100%)[
  #align(center)[#body]
]

#let appendix = [
  #set par(first-line-indent: 0em, leading: 4.9pt, justify: false)

  #heading(level: 1, numbering: none)[付録]

  #columns(2, gutter: 8mm, [
    #paragraph[学習設定の要約][
      本文では省略した学習設定と推論条件のうち，主要なものを表 2 から表 5 にまとめる．
      長文 SFT では，48 層へ拡張した 12B 級モデルに対して最大系列長 `16384` のフルパラメータ学習を行い，
      合成データセットの出力形式に沿うよう，SFT 側で用いた chat template / system prompt に
      GRPO 側の設定を合わせた．
    ]

    #appendix-table(size: 8.7pt)[
      #figure(
        kind: table,
        supplement: [表],
        caption: [長さとバッチサイズの設定],
        centered-table[
          #table(
            columns: 2,
            align: (left, center),
            inset: 2.5pt,
            stroke: (x, y) => if y == 0 { 0.8pt } else { 0.4pt },
            [*項目*], [*値*],
            [SFT 最大系列長], [`16384`],
            [prompt 長上限], [`1024`],
            [response 長], [`7168`],
            [device ごとの batch], [`1`],
            [unique prompts], [`8`],
            [samples / prompt], [`32`],
            [rollout batch], [`256`],
          )
        ],
      )
    ]

    #appendix-table(size: 7.5pt)[
      #figure(
        kind: table,
        supplement: [表],
        caption: [GRPO 実行時の主要設定],
        centered-table[
          #table(
            columns: 2,
            align: (left, left),
            inset: 2.5pt,
            stroke: (x, y) => if y == 0 { 0.8pt } else { 0.4pt },
            [*項目*], [*設定*],
            [verifier], [`qa_10k=math-verify`],
            [ground truths], [`ground_truth`],
            [template], [`math_problem_with_boxed`],
            [DeepSpeed stage], [`3`],
            [epochs], [`1`],
            [learners / node], [`3`],
            [vLLM engines], [`5`],
            [tensor parallel], [`1`],
            [GPU mem util], [`0.65`],
            [beta], [`0.00`],
            [ref policy], [`false`],
            [sync backend], [`nccl`],
            [prefix caching], [on],
            [save traces], [on],
            [eager mode], [on],
            [grad checkpoint], [on],
            [active sampling], [on],
            [zero-std filter], [on],
            [async steps], [`4`],
            [inflight updates], [on],
            [mask truncation], [on],
            [GRPO steps], [`1725`],
          )
        ],
      )
    ]

    #colbreak()

    #paragraph[安定化設定][
      非同期 RL の下では clip higher は発動しない挙動のため設定から外し，
      数値安定化には `truncated_importance_sampling_ratio_cap = 2.0` を有効にした．
      また，`advantage_normalization_type = centered` により平均中心化のみを適用した．
      また，指定していないがデフォルトで `loss_fn = dapo`, `loss_denominator = token` が使われ，
      token-level loss 集約を構成している．
    ]

    #appendix-table(size: 8.7pt)[
      #figure(
        kind: table,
        supplement: [表],
        caption: [代表的な推論条件],
        centered-table[
          #table(
            columns: 2,
            align: (left, center),
            inset: 3pt,
            stroke: (x, y) => if y == 0 { 0.8pt } else { 0.4pt },
            [*項目*], [*値*],
            [sampling temperature], [`0.7`],
            [max generation length], [`16384`],
            [samples], [`40`],
          )
        ],
      )
    ]

    #appendix-table(size: 8.7pt)[
      #figure(
        kind: table,
        supplement: [表],
        caption: [GRPO 学習後半の温度スケジュール],
        centered-table[
          #table(
            columns: 2,
            align: (left, center),
            inset: 4pt,
            stroke: (x, y) => if y == 0 { 0.8pt } else { 0.4pt },
            [*ステップ範囲*], [*温度*],
            [`0`--`999`], [`1.000`],
            [`1000`--`1300`], [`1.010`],
            [`1300`--`1400`], [`1.020`],
            [`1400`--`1500`], [`1.015`],
            [`1500`--`1600`], [`1.020`],
            [`1600`--`1700`], [`1.015`],
            [`1700`--`1725`], [`1.018`],
          )
        ],
      )
    ]

    温度を小数点第 1 位のオーダーまで上げると，truncation mask の急増によってバッチサイズ維持が難しくなり，
    生成長が両極端に割れて学習が不安定化した．そのため，最終的には `1.018` を採用した．
  ])
]
