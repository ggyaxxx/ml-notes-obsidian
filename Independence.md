# Independence

## Core Idea

Two events $A$ and $B$ are independent when knowing that one occurred does not change the probability of the other.

$$
P(A\mid B)=P(A)
$$

From [[Conditional Probability]]:

$$
P(A\mid B)=\frac{P(A\cap B)}{P(B)}
$$

Therefore, for independent events:

$$
P(A\cap B)=P(A)P(B)
$$

## Independent vs Mutually Exclusive

These are different concepts.

**Independent:**

Knowing $B$ does not change the probability of $A$.

$$
P(A\mid B)=P(A)
$$

**Mutually exclusive:**

$A$ and $B$ cannot happen together.

$$
A\cap B=\varnothing
$$

If both events have non-zero probability, mutually exclusive events cannot be independent.

## How to Check Independence

Check either:

$$
P(A\mid B)=P(A)
$$

or:

$$
P(A\cap B)=P(A)P(B)
$$

## Connections

- [[Conditional Probability]]
- [[Joint Probability]]
- [[Random Variables]]
- [[Bayes' Theorem]]