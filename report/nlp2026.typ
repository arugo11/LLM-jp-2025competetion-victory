#let body-fonts = (
  "TeXGyreTermesX",
  "Harano Aji Mincho",
  "HaranoAjiMincho",
  "Noto Serif CJK JP",
  "Noto Serif JP",
  "IPAexMincho",
  "Linux Libertine",
  "New Computer Modern",
)

#let heading-fonts = (
  "TeX Gyre Heros",
  "Harano Aji Gothic",
  "HaranoAjiGothic",
  "Noto Sans CJK JP",
  "Noto Sans JP",
  "IPAexGothic",
  "Linux Libertine",
  "New Computer Modern",
)

#let mono-fonts = (
  "DejaVu Sans Mono",
  "New Computer Modern Mono",
)

#let pkg(body) = text(font: heading-fonts)[#body]
#let code(body) = text(font: mono-fonts)[#body]
#let comment(body) = text(fill: red)[#body]

#let paragraph(title, body) = block(above: 0.7em, below: 0.2em)[
  #set text(font: heading-fonts, size: 10pt, weight: "bold")
  #title
  #linebreak()
  #set text(font: body-fonts, size: 10pt, weight: "regular")
  #body
]

#let abstract-block(lang: "ja", body) = block(above: 0.2em, below: 0.9em)[
  #set text(font: heading-fonts, size: 14pt, weight: "bold")
  #if lang == "en" { [Abstract] } else { [概要] }
  #v(0.35em)
  #set text(font: body-fonts, size: 10pt, weight: "regular")
  #body
]

#let detail-block(body) = block(
  breakable: false,
  above: 0.2em,
  below: 0.6em,
  inset: (left: 1.2em),
)[
  #set par(first-line-indent: 0em, justify: false, leading: 4.2pt)
  #body
]

#let _heading(size, it) = {
  let number = if it.numbering == none {
    []
  } else {
    counter(heading).display(it.numbering)
  }

  block(above: 1.0em, below: 0.45em, breakable: false)[
    #set text(font: heading-fonts, size: size, weight: "bold")
    #number
    #if it.numbering != none { h(0.45em) }
    #it.body
  ]
}

#let nlp2026(
  title: none,
  authors: none,
  abstract: none,
  backmatter: none,
  appendix: none,
  lang: "ja",
  body,
) = {
  set page(
    paper: "a4",
    margin: (top: 30mm, bottom: 30mm, inside: 20mm, outside: 20mm),
    numbering: none,
  )
  set text(font: body-fonts, size: 10pt, lang: lang)
  set par(justify: true, first-line-indent: 1em, leading: 4.9pt)
  set heading(numbering: "1.1.1")
  set math.equation(numbering: "(1)", supplement: [式])
  show emph: it => text(font: heading-fonts, weight: "bold")[#it.body]
  show strong: it => text(weight: "bold")[#it.body]
  show raw.where(block: false): it => text(font: mono-fonts)[#it.text]
  show footnote.entry: set text(size: 9pt)
  show heading.where(level: 1): it => _heading(14pt, it)
  show heading.where(level: 2): it => _heading(12pt, it)
  show heading.where(level: 3): it => _heading(11pt, it)

  align(center)[
    #set text(font: heading-fonts, size: 16pt, weight: "bold")
    #title
    #v(0.9em)
    #set text(font: body-fonts, size: 10.5pt, weight: "regular")
    #authors
  ]

  let main = if abstract != none {
    abstract-block(lang: lang, abstract) + body
  } else {
    body
  }

  columns(2, gutter: 8mm, main)

  if backmatter != none {
    pagebreak()
    columns(2, gutter: 8mm, backmatter)
  }

  if appendix != none {
    pagebreak()
    appendix
  }
}
