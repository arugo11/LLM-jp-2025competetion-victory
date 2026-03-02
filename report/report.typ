#import "nlp2026.typ": nlp2026, paragraph
#import "lib/components.typ": flow-step, flow-arrow, stage-card
#import "lib/metadata.typ": title, authors, abstract
#import "lib/backmatter.typ": backmatter

#show: nlp2026.with(
  title: title,
  authors: authors,
  abstract: abstract,
  backmatter: backmatter,
)

#include "chapters/01-introduction.typ"
#include "chapters/02-related-work.typ"
#include "chapters/03-training-pipeline.typ"
#include "chapters/04-inference-system.typ"
#include "chapters/05-experiments.typ"
#include "chapters/06-conclusion.typ"
