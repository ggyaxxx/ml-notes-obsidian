# Conditional Probability

## Core Idea

Conditional probability answers the question:

> What is the probability of event $A$, knowing that event $B$ has occurred?

It is written as:

$$
P(A\mid B)
$$

and read as:

> Probability of $A$ **given** $B$.

The symbol $\mid$ means **given**, not AND.

---

## Conditioning Changes the Reference Universe

Before receiving any additional information, probabilities are evaluated relative to the original sample space:

$$
\Omega
$$

After learning that $B$ occurred, outcomes outside $B$ are no longer possible.

Therefore, $B$ becomes the new reference universe.

> **Conditioning changes the reference universe from $\Omega$ to $B$.**

We then ask:

> Within $B$, how much also belongs to $A$?

The relevant outcomes are therefore:

$$
A\cap B
$$

---

## Definition

For $P(B)>0$:

$$
\boxed{
P(A\mid B)=\frac{P(A\cap B)}{P(B)}
}
$$

where:

- $P(A\mid B)$ = probability of $A$ given $B$
- $P(A\cap B)$ = probability that both $A$ and $B$ occur
- $P(B)$ = probability of the event we are conditioning on

The denominator $P(B)$ accounts for the fact that $B$ has become our new reference universe.

---

## Example

Consider:

$$
\Omega=\{1,2,3,4,5,6,7,8,9,10\}
$$

Define:

$$
A=\{\text{even}\}=\{2,4,6,8,10\}
$$

and:

$$
B=\{\text{greater than 6}\}=\{7,8,9,10\}
$$

After learning that $B$ occurred, our reference universe becomes:

$$
B=\{7,8,9,10\}
$$

Within $B$, the outcomes that also satisfy $A$ are:

$$
A\cap B=\{8,10\}
$$

Therefore:

$$
P(A\mid B)
=
\frac{P(A\cap B)}{P(B)}
=
\frac{2/10}{4/10}
=
\frac24
=
\frac12
$$

We can also see this directly inside the new reference universe:

$$
P(A\mid B)
=
\frac{|\{8,10\}|}{|\{7,8,9,10\}|}
=
\frac24
$$

---

## AND vs GIVEN

These must not be confused.

### AND

$$
P(A\cap B)
$$

means:

> Probability of $A$ **AND** $B$.

### GIVEN

$$
P(A\mid B)
$$

means:

> Probability of $A$ **GIVEN** $B$.

Therefore:

$$
\boxed{\cap = \text{AND}}
$$

$$
\boxed{\mid = \text{GIVEN}}
$$

---

## Connection to Independence

If learning that $B$ occurred does not change the probability of $A$, then:

$$
P(A\mid B)=P(A)
$$

This is the core idea behind [[Independence]].

---

## What I Should Be Able to Explain

Without looking at the formula:

1. What does $P(A\mid B)$ mean?
2. What is the difference between $P(A\cap B)$ and $P(A\mid B)$?
3. Why does $B$ become the new reference universe?
4. Why does $A\cap B$ appear in the numerator?
5. Why do we divide by $P(B)$?
6. What happens when $P(A\mid B)=P(A)$?

---

## Connections

- [[Probability Toolkit #1 — Foundations]]
- [[Independence]]
- [[Law of Total Probability]]
- [[Bayes' Theorem]]
- [[Joint Probability]]
- [[Random Variables]]