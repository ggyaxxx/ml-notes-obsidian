# Law of Total Probability

## Core Idea

The **Law of Total Probability** allows us to compute the overall probability of an event by considering **all the mutually exclusive ways in which that event can occur**.

The key intuition is:

> Divide the sample space into all possible cases, compute how much each case contributes to the event, and sum those contributions.

---

## 1. Partitioning the Sample Space

Suppose the sample space $\Omega$ is divided into events:

$$
G_1,G_2,\ldots,G_n
$$

that form a [[Partition]].

This requires two properties.

### Mutually Exclusive

Different groups cannot occur simultaneously:

$$
G_i\cap G_j=\varnothing
\qquad\text{for }i\neq j
$$

### Collectively Exhaustive

Together, the groups cover the entire sample space:

$$
\bigcup_{i=1}^{n}G_i=\Omega
$$

Therefore, every outcome in $\Omega$ belongs to **exactly one** $G_i$.

As a consequence:

$$
\sum_{i=1}^{n}P(G_i)=1
$$

---

## 2. Decomposing an Event

Suppose we are interested in an event $D$.

Since the $G_i$ form a partition, $D$ can occur through different mutually exclusive "paths":

$$
D\cap G_1,\quad
D\cap G_2,\quad
\ldots,\quad
D\cap G_n
$$

Therefore:

$$
D=
(D\cap G_1)
\cup
(D\cap G_2)
\cup
\cdots
\cup
(D\cap G_n)
$$

Since these events are mutually exclusive:

$$
P(D)
=
\sum_{i=1}^{n}P(D\cap G_i)
$$

---

## 3. Using Conditional Probability

From [[Conditional Probability]]:

$$
P(D\mid G_i)
=
\frac{P(D\cap G_i)}{P(G_i)}
$$

Rearranging:

$$
P(D\cap G_i)
=
P(D\mid G_i)P(G_i)
$$

Therefore each group contributes:

$$
\boxed{
P(D\mid G_i)P(G_i)
}
$$

to the total probability of $D$.

---

## 4. Law of Total Probability

Summing all contributions gives:

$$
\boxed{
P(D)
=
\sum_{i=1}^{n}
P(D\mid G_i)P(G_i)
}
$$

This is the **Law of Total Probability**.

For three groups:

$$
P(D)
=
P(D\mid G_1)P(G_1)
+
P(D\mid G_2)P(G_2)
+
P(D\mid G_3)P(G_3)
$$

---

## 5. Interpretation as Weighted Contributions

Each term contains two different pieces of information:

$$
P(G_i)
$$

tells us **how large group $G_i$ is** relative to the whole population.

While:

$$
P(D\mid G_i)
$$

tells us **how frequent $D$ is inside group $G_i$**.

Their product:

$$
P(D\mid G_i)P(G_i)
$$

tells us how much that group contributes to $D$ in the **entire population**.

Therefore, the Law of Total Probability can be understood as a form of **weighted average** of the conditional probabilities:

$$
P(D)=\sum_i P(G_i)P(D\mid G_i)
$$

where the weights are the probabilities $P(G_i)$.

---

## 6. Example

Suppose a population is partitioned into two groups:

$$
M,\quad F
$$

with:

$$
P(M)=0.40
$$

$$
P(F)=0.60
$$

Suppose:

$$
P(D\mid M)=0.10
$$

and:

$$
P(D\mid F)=0.20
$$

### Contribution from $M$

$$
P(D\cap M)
=
P(D\mid M)P(M)
$$

$$
=0.10\cdot0.40
=0.04
$$

### Contribution from $F$

$$
P(D\cap F)
=
P(D\mid F)P(F)
$$

$$
=0.20\cdot0.60
=0.12
$$

### Total Probability

$$
P(D)
=
0.04+0.12
=
0.16
$$

Therefore:

$$
\boxed{P(D)=0.16}
$$

---

## 7. Why We Cannot Simply Sum Conditional Probabilities

In general:

$$
P(D)
\neq
\sum_i P(D\mid G_i)
$$

because the groups may have different probabilities.

For example:

$$
P(D\mid G_1)=0.8
$$

does not tell us how much $G_1$ contributes to the whole population.

We also need:

$$
P(G_1)
$$

The correct contribution is:

$$
P(D\mid G_1)P(G_1)
$$

A group where $D$ is very frequent may still contribute little to the total if that group itself is very rare.

---

## 8. Why the Partition Matters

We cannot arbitrarily choose overlapping groups.

If:

$$
G_1\cap G_2\neq\varnothing
$$

some outcomes would belong to multiple groups and could be counted more than once.

Likewise, if:

$$
G_1\cup G_2\cup\cdots\cup G_n\neq\Omega
$$

some possible outcomes would not be counted at all.

Therefore, the $G_i$ must form a [[Partition]]:

> mutually exclusive and collectively exhaustive.

---

## 9. Connection to Bayes' Theorem

The Law of Total Probability will allow us to compute quantities such as:

$$
P(D)
$$

even when we only know conditional probabilities such as:

$$
P(D\mid G_i)
$$

This becomes especially important in [[Bayes' Theorem]], where probabilities such as $P(D)$ appear in the denominator.

The Law of Total Probability provides a way to reconstruct that denominator by considering **all possible paths through which $D$ can occur**.

---

## What I Should Be Able to Explain

Without looking at the formula:

1. Why do we divide the sample space into groups?
2. What does it mean for the groups to form a partition?
3. Why must the groups be mutually exclusive?
4. Why must they collectively cover $\Omega$?
5. What does $P(D\mid G_i)$ tell me?
6. What does $P(G_i)$ tell me?
7. Why do I multiply $P(D\mid G_i)$ by $P(G_i)$?
8. Why do I then sum the contributions?
9. Why can't I simply sum the conditional probabilities?
10. How will this help us when deriving Bayes' theorem?

---

## Formula to Understand and Reconstruct

$$
\boxed{
P(D)
=
\sum_{i=1}^{n}
P(D\mid G_i)P(G_i)
}
$$

Rather than memorizing the formula, remember:

> **All possible paths to $D$ → contribution of each path → sum the contributions.**

---

## Connections

- [[Conditional Probability]]
- [[Partition]]
- [[Independence]]
- [[Bayes' Theorem]]
- [[Marginal Probability]]