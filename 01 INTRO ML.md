# Probability Toolkit #1 — Foundations

## 1. Sample Space

The **sample space** contains all possible outcomes of an experiment.

It is usually denoted by:

$$
\Omega
$$

Example: drawing one number from 1 to 10:

$$
\Omega = \{1,2,3,4,5,6,7,8,9,10\}
$$

A single possible result is called an **outcome**.

For example:

$$
\omega = 4
$$

is one possible outcome belonging to $\Omega$.

---

## 2. Events

An **event** is a set of outcomes from the sample space.

Formally:

$$
A \subseteq \Omega
$$

For example, let $A$ be the event "the number is even":

$$
A = \{2,4,6,8,10\}
$$

Probability is assigned to events:

$$
P(A)
$$

In this example:

$$
P(A)=\frac{5}{10}=0.5
$$

### Important distinction

$A$ is the **event**.

$P(A)$ is the **probability assigned to that event**.

They are not the same object.

---

## 3. Probability as a Function

Probability can be thought of as a function that takes an event and assigns it a number between 0 and 1:

$$
P:\mathcal{F}\rightarrow[0,1]
$$

For now, $\mathcal{F}$ can be informally understood as the collection of events to which probabilities can be assigned.

Conceptually:

$$
\text{Event} \xrightarrow{P} \text{Number between 0 and 1}
$$

For example:

$$
\{2,4,6,8,10\} \xrightarrow{P} 0.5
$$

---

## 4. Complement

The **complement** of an event $A$, written $A^c$, contains all outcomes in $\Omega$ that are **not** in $A$.

For example, if:

$$
\Omega=\{1,2,3,4,5,6\}
$$

and:

$$
A=\{2,4,6\}
$$

then:

$$
A^c=\{1,3,5\}
$$

Since either $A$ or its complement must occur:

$$
P(A)+P(A^c)=1
$$

Therefore:

$$
\boxed{P(A^c)=1-P(A)}
$$

---

## 5. Intersection — AND

The symbol $\cap$ means **intersection**.

For two events $A$ and $B$:

$$
A\cap B
$$

contains the outcomes belonging to **both** $A$ and $B$.

Think:

$$
\boxed{\cap \quad \Longleftrightarrow \quad \text{AND}}
$$

Example:

$$
A=\{2,4,6,8,10\}
$$

$$
B=\{7,8,9,10\}
$$

Then:

$$
A\cap B=\{8,10\}
$$

and:

$$
P(A\cap B)=\frac{2}{10}=0.2
$$

---

## 6. Union — OR

The symbol $\cup$ means **union**.

$$
A\cup B
$$

contains outcomes belonging to $A$, $B$, **or both**.

Think:

$$
\boxed{\cup \quad \Longleftrightarrow \quad \text{OR}}
$$

The probability of the union is:

$$
\boxed{P(A\cup B)=P(A)+P(B)-P(A\cap B)}
$$

We subtract $P(A\cap B)$ because otherwise the outcomes belonging to both events would be counted twice.

---

## 7. Conditional Probability — GIVEN

The vertical bar $\mid$ means **given**.

Therefore:

$$
P(A\mid B)
$$

is read as:

> The probability of $A$ **given** $B$.

It means:

> What is the probability that $A$ occurs, knowing that $B$ has occurred?

The formula is:

$$
\boxed{P(A\mid B)=\frac{P(A\cap B)}{P(B)}}
$$

provided that:

$$
P(B)>0
$$

### Core intuition

**Conditioning changes the universe in which we are reasoning.**

Suppose:

$$
\Omega=\{1,2,3,4,5,6,7,8,9,10\}
$$

with:

$$
A=\{\text{even}\}=\{2,4,6,8,10\}
$$

and:

$$
B=\{\text{greater than 6}\}=\{7,8,9,10\}
$$

Before knowing $B$, our reference universe is $\Omega$.

After learning that $B$ occurred, our new effective universe becomes:

$$
B=\{7,8,9,10\}
$$

Inside this new universe, the outcomes that also satisfy $A$ are:

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
\frac{2}{4}
=
\frac{1}{2}
=
0.5
$$

Once $B$ becomes our reference universe, we can also reason directly:

$$
P(A\mid B)
=
\frac{|\{8,10\}|}{|\{7,8,9,10\}|}
=
\frac{2}{4}
=
\frac{1}{2}
$$

### Key idea

> **Conditioning changes the reference universe.**

The conditional probability formula is the mathematical formalization of this idea.

---

## 8. AND vs GIVEN

This distinction is fundamental.

### AND

For events:

$$
P(A\cap B)
$$

means:

> probability of **A AND B**

Therefore:

$$
\boxed{\cap = \text{AND}}
$$

### GIVEN

$$
P(A\mid B)
$$

means:

> probability of **A GIVEN B**

Therefore:

$$
\boxed{\mid = \text{GIVEN}}
$$

These are different concepts.

---

## 9. Notation with Random Variables

Later we will work with **random variables**, usually denoted by capital letters such as $X$ and $Y$.

The notation:

$$
P(X=x,Y=y)
$$

means:

> the probability that $X=x$ **AND** $Y=y$

Conceptually:

$$
P(X=x,Y=y)
=
P(\{X=x\}\cap\{Y=y\})
$$

This is **not** conditional probability.

Conditional probability is written:

$$
P(X=x\mid Y=y)
$$

and means:

> the probability that $X=x$ **GIVEN that** $Y=y$

### Notation Summary

| Notation | Meaning |
|---|---|
| $A\cap B$ | A **AND** B |
| $A\cup B$ | A **OR** B |
| $P(A\mid B)$ | A **GIVEN** B |
| $P(X=x,Y=y)$ | $X=x$ **AND** $Y=y$ |
| $P(X=x\mid Y=y)$ | $X=x$ **GIVEN** $Y=y$ |

---

## 10. What I Should Be Able to Explain

Without looking at the formulas, I should be able to answer:

1. What is the difference between an **outcome**, a **sample space**, and an **event**?
2. Why is $A$ different from $P(A)$?
3. What does $A\cap B$ mean?
4. What does $A\cup B$ mean?
5. What is the difference between **AND** and **GIVEN**?
6. Why does conditioning on $B$ make $B$ the new reference universe?
7. Why does the following formula make intuitive sense?

$$
P(A\mid B)=\frac{P(A\cap B)}{P(B)}
$$

---

## Symbols to Recognize Automatically

| Symbol | Meaning |
|---|---|
| $\Omega$ | Sample space |
| $\omega$ | A single outcome |
| $\cap$ | Intersection / AND |
| $\cup$ | Union / OR |
| $\mid$ | Given |
| $A^c$ | Complement of $A$ |
| $P(A)$ | Probability of event $A$ |
| $P(A\mid B)$ | Probability of $A$ given $B$ |

---

## Connections

- [[Conditional Probability]]
- [[Independence]]
- [[Law of Total Probability]]
- [[Bayes' Theorem]]
- [[Random Variables]]