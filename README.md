# Machine Learning — Sapienza Notes

Personal Obsidian knowledge base for studying Machine Learning at Sapienza.

The goal of this repository is not to reproduce lectures verbatim, but to build a connected set of conceptual notes that supports deep understanding, mathematical rigor, active recall, and exam preparation.

## Study Strategy

Machine Learning is the main path. Mathematical prerequisites are recovered **just in time** when they are needed rather than studied exhaustively in advance.

The working loop is:

```text
Machine Learning topic
        ↓
identify a real prerequisite gap
        ↓
short focused prerequisite detour
        ↓
return immediately to Machine Learning
```

Paper or a tablet is used as working memory for calculations, derivations, and exercises. Obsidian contains the consolidated conceptual knowledge.

## Current Knowledge Map

### Probability Foundations

- [[01 INTRO ML]] — sample space, outcomes, events, probability, intersection, union, conditional probability foundations
- [[Conditional Probability]]
- [[Independence]]
- [[Partition]]
- [[Law of Total Probability]]
- [[Bayes' Theorem]]

### Random Variables and Distributions

- [[Random Variables]]
- [[Probability Distribution]]
- [[Joint Probability]]
- [[Joint Distributions]]
- [[Marginal Probability]]

### Expectation and Dependence

- [[Expectation]]
- [[Variance]]
- [[Covariance]]
- [[Correlation]] — placeholder until required by the course

A dedicated navigation page is available in [[Probability Toolkit]].

## Note Organization

Notes are concept-oriented and connected through Obsidian wikilinks.

Some short files are **index/alias notes**. They point to a canonical explanation already contained in another note instead of duplicating material. This keeps links valid while avoiding divergent copies of the same concept.

For example:

- `Variance.md` points to the variance section in `Expectation.md`;
- `Sample Space.md` points to the foundations in `01 INTRO ML.md`;
- `Marginal Probability.md` points to the marginal-distribution section in `Joint Distributions.md`.

Existing study notes are treated as important source material and should not be deleted or overwritten merely for organizational cleanup.

## Learning Standard

A topic is not considered consolidated simply because it has been read or understood once. The target is to be able to:

- explain the concept without notes;
- interpret the notation precisely;
- reconstruct important formulas from their meaning;
- recognize when a method applies;
- solve exercises and variants;
- detect conceptual and calculation errors;
- connect the concept to the surrounding Machine Learning material.

The preparation target is exam-level mastery, including the ability to handle unfamiliar variants rather than only reproduce known exercises.
