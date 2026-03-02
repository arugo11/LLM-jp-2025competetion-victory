#let flow-step(title, detail, fill) = rect(
  width: 100%,
  inset: 7pt,
  radius: 4pt,
  stroke: rgb("AAB4C2"),
  fill: fill,
)[
  #set text(weight: "bold", size: 9pt)
  #title
  #v(2pt)
  #set text(weight: "regular", size: 8.5pt)
  #detail
]

#let flow-arrow = align(center)[#text(size: 11pt, fill: rgb("5C6773"))[↓]]

#let stage-card(stage, process, goal, fill) = rect(
  width: 100%,
  inset: 7pt,
  radius: 4pt,
  stroke: rgb("AAB4C2"),
  fill: fill,
)[
  #set text(weight: "bold", size: 9pt)
  #stage
  #v(2pt)
  #set text(size: 8.3pt)
  #text(weight: "bold")[処理:] #process
  #linebreak()
  #text(weight: "bold")[狙い:] #goal
]
