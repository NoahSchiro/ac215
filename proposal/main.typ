
#import "@preview/basic-report:0.5.0": *

#show: it => basic-report(
  doc-category: "",
  doc-title: "Lorem",
  author: "Noah Schiro",
  affiliation: "Harvard",
  logo: "", // we can attach a small image here if we want with image("./path_to_image")
  language: "en",
  compact-mode: true,
  it
)

= Introduction
short problem statement here

== Background
#lorem(100)

== Motivation
#lorem(100)

= Data
source and description here.

== Key attributes / shape of the data
#lorem(100)

== Relevance
#lorem(100)

== Quality
#lorem(100)

= Scope and Design
probably should have some sort of diagram here for the architecture. Talk about scaleability, how many users we want to handle, etc.

== Risks and limitations
What problems might arise?

= Milestones

== Milestone 2
#lorem(10)
== Milestone 3
#lorem(10)
== Milestone 4
#lorem(10)
== Milestone 5
#lorem(10)

= References
any papers / datasets / models we want to ref
