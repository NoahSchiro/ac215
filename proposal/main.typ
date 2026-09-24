#show link: set text(fill: blue)

#let title = "[G/M]aze tracking"
#let authors = "Gabe Gonzalez, Rui Huang, Noah Schiro, Minh-Thy Tyler, Rongzhi Chen"

#set page(paper: "us-letter", margin: (x: 1in, y: 1in, top: 0.5in))
#set document(title: title, author: authors)
#set par(spacing: 0.8em)
#set enum(spacing: 0.55em)
#set list(spacing: 0.55em)
#show heading: set text(size: 11pt)
#show heading: set block(above: 0.75em, below: 0.35em)

#align(center, text(size: 14pt, weight: "bold")[#title])
#align(center, authors)

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
