#let figure-panel(body) = rect(
  width: 100%,
  inset: 7pt,
  radius: 6pt,
  stroke: 0.8pt + rgb("C7D0DA"),
  fill: rgb("FCFDFE"),
)[#body]

#let accent-card(
  title,
  body,
  fill,
  accent,
  title-size: 8.8pt,
  body-size: 7.8pt,
) = rect(
  width: 100%,
  inset: 0pt,
  radius: 5pt,
  stroke: 0.8pt + rgb("B8C3CF"),
  fill: fill,
)[ 
  #stack(
    dir: ttb,
    spacing: 0pt,
    rect(width: 100%, height: 4.5pt, stroke: none, fill: accent),
    block(inset: (x: 7pt, y: 6pt))[
      #align(center)[
        #text(weight: "bold", size: title-size, fill: rgb("243140"))[#title]
      ]
      #v(2pt)
      #set text(size: body-size, fill: rgb("495C6D"))
      #body
    ],
  )
]

#let flow-step(title, detail, fill, accent) = accent-card(
  title,
  detail,
  fill,
  accent,
)

#let flow-arrow = align(center)[#text(size: 11.5pt, fill: rgb("6A7480"))[↓]]

#let stage-detail(label, content, accent) = grid(
  columns: (18pt, 1fr),
  gutter: 5pt,
  align: (left, horizon),
  text(weight: "bold", size: 7.1pt, fill: accent)[#label],
  text(size: 7.6pt, fill: rgb("495C6D"))[#content],
)

#let stage-card(stage, process, goal, fill, accent) = accent-card(
  stage,
  [
    #stage-detail([処理], process, accent)
    #v(2.2pt)
    #rect(width: 100%, height: 0.5pt, stroke: none, fill: rgb("D7DEE6"))
    #v(2.2pt)
    #stage-detail([狙い], goal, accent)
  ],
  fill,
  accent,
  title-size: 8.6pt,
  body-size: 7.7pt,
)
