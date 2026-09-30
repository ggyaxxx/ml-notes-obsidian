# Partition

## Definition

A collection of events

$$
G_1,G_2,\ldots,G_n
$$

forms a **partition** of the sample space $\Omega$ if every outcome in $\Omega$ belongs to **exactly one** of the events.

This requires two conditions.

### 1. Mutually Exclusive

For every $i\neq j$:

$$
G_i\cap G_j=\varnothing
$$

No outcome can belong to two different groups.

### 2. Collectively Exhaustive

$$
\bigcup_{i=1}^{n}G_i=\Omega
$$

Every possible outcome belongs to one of the groups.

Therefore:

$$
\sum_{i=1}^{n}P(G_i)=1
$$

---

## Intuition

A partition divides the entire sample space into **non-overlapping pieces**.

Each outcome belongs to:

- at least one group, because the groups cover $\Omega$;
- at most one group, because the groups do not overlap.

Therefore each outcome belongs to **exactly one group**.

---

## Example

For:

$$
\Omega=\{1,2,3,4,5,6\}
$$

the events

$$
G_1=\{1,2\}
$$

$$
G_2=\{3,4\}
$$

$$
G_3=\{5,6\}
$$

form a partition of $\Omega$.

They do not overlap and:

$$
G_1\cup G_2\cup G_3=\Omega
$$

---

## Why It Matters

Partitions allow us to decompose an event into all the mutually exclusive ways in which it can occur.

This is the key structure behind the [[Law of Total Probability]]:

$$
P(D)=\sum_i P(D\mid G_i)P(G_i)
$$

---

## Connections


- [[Law of Total Probability]]
- [[Conditional Probability]]
- [[Bayes' Theorem]]