# Bayes' Theorem

## Core Idea

[[Bayes' Theorem]] allows us to **update the probability of a hypothesis after observing new evidence**.

The basic idea is:

> Start with what we believe about a hypothesis before observing the evidence, then update that probability according to how compatible the observed evidence is with the hypothesis.

In short:

$$
\text{Prior}
\xrightarrow{\text{observe evidence}}
\text{Posterior}
$$

---

## 1. Derivation from Conditional Probability

From [[Conditional Probability]]:

$$
P(A\mid B)
=
\frac{P(A\cap B)}{P(B)}
$$

Similarly:

$$
P(B\mid A)
=
\frac{P(A\cap B)}{P(A)}
$$

Rearranging both equations:

$$
P(A\cap B)
=
P(A\mid B)P(B)
$$

and:

$$
P(A\cap B)
=
P(B\mid A)P(A)
$$

The joint probability $P(A\cap B)$ is the same in both expressions because:

$$
A\cap B=B\cap A
$$

Therefore:

$$
P(A\mid B)P(B)
=
P(B\mid A)P(A)
$$

Dividing by $P(B)$, assuming $P(B)>0$:

$$
\boxed{
P(A\mid B)
=
\frac{P(B\mid A)P(A)}
{P(B)}
}
$$

This is Bayes' Theorem.

---

## 2. The Four Components

In:

$$
P(A\mid B)
=
\frac{P(B\mid A)P(A)}
{P(B)}
$$

the four quantities have different roles.

### Prior — $P(A)$

$$
\boxed{P(A)}
$$

The **prior** is the probability assigned to the hypothesis $A$ **before observing the current evidence $B$**.

> What did I believe about the hypothesis before seeing this evidence?

---

### Likelihood — $P(B\mid A)$

$$
\boxed{P(B\mid A)}
$$

The **likelihood** tells us how probable the observed evidence $B$ would be **if the hypothesis $A$ were true**.

> Assuming $A$ is true, how likely would I be to observe $B$?

Important:

$$
\boxed{
P(B\mid A)\neq P(A\cap B)
}
$$

The likelihood is a **conditional probability**, not the joint probability.

---

### Evidence — $P(B)$

$$
\boxed{P(B)}
$$

The **evidence** is the overall probability of observing $B$.

It accounts for **all possible ways in which $B$ could occur**.

If $A$ and $A^c$ form a partition:

$$
P(B)
=
P(B\mid A)P(A)
+
P(B\mid A^c)P(A^c)
$$

This follows from the [[Law of Total Probability]].

$P(B)$ also acts as the **normalizing denominator** when we restrict our universe to cases in which $B$ has occurred.

---

### Posterior — $P(A\mid B)$

$$
\boxed{P(A\mid B)}
$$

The **posterior** is the updated probability of the hypothesis $A$ **after observing the evidence $B$**.

> After seeing $B$, how probable do I now consider $A$?

---

## 3. Bayes as an Update

Conceptually:

$$
\boxed{
\text{Prior}
+
\text{Evidence}
\longrightarrow
\text{Posterior}
}
$$

More precisely, the likelihood tells us how the observed evidence should modify the prior:

$$
\boxed{
\text{Posterior}
=
\frac{
\text{Likelihood}\times\text{Prior}
}{
\text{Evidence}
}
}
$$

or:

$$
\boxed{
P(A\mid B)
=
\frac{
P(B\mid A)P(A)
}{
P(B)
}
}
$$

The formula should not replace the underlying reasoning:

> Start from a prior probability, observe new evidence, and update the probability of the hypothesis according to how compatible that evidence is with the hypothesis.

---

## 4. Example

Suppose a population is partitioned into:

$$
M,\qquad F
$$

with:

$$
P(M)=0.40
$$

$$
P(F)=0.60
$$

Let $D$ be some observed characteristic.

Suppose:

$$
P(D\mid M)=0.10
$$

and:

$$
P(D\mid F)=0.20.
$$

We observe $D$ and want:

$$
P(M\mid D).
$$

### Prior

Before observing $D$:

$$
P(M)=0.40.
$$

### Likelihood

If the person belongs to $M$, the probability of observing $D$ is:

$$
P(D\mid M)=0.10.
$$

### Evidence

Using the [[Law of Total Probability]]:

$$
P(D)
=
P(D\mid M)P(M)
+
P(D\mid F)P(F)
$$

$$
=
0.10(0.40)+0.20(0.60)
$$

$$
=
0.04+0.12
=
0.16.
$$

### Posterior

Using Bayes:

$$
P(M\mid D)
=
\frac{P(D\mid M)P(M)}
{P(D)}
$$

$$
=
\frac{0.10\cdot0.40}{0.16}
$$

$$
=
\frac{0.04}{0.16}
=
0.25.
$$

Therefore:

$$
\boxed{
P(M\mid D)=0.25
}
$$

The probability assigned to $M$ changed from:

$$
\underbrace{0.40}_{\text{prior}}
\quad\longrightarrow\quad
\underbrace{0.25}_{\text{posterior}}.
$$

Observing $D$ decreased the probability of $M$ because $D$ is more likely under $F$ than under $M$:

$$
P(D\mid F)>P(D\mid M).
$$

---

## 5. What Determines the Update?

Suppose we are comparing two hypotheses $A$ and $A^c$.

If:

$$
P(B\mid A)>P(B\mid A^c),
$$

then observing $B$ provides evidence in favor of $A$ relative to $A^c$.

If:

$$
P(B\mid A)<P(B\mid A^c),
$$

then observing $B$ provides evidence against $A$ relative to $A^c$.

However, the likelihood alone does **not** determine which hypothesis has the larger posterior.

The posterior also depends on the **prior**.

Therefore:

> Evidence may favor one hypothesis without making that hypothesis the most probable one overall.

---

## 6. Equal Priors

If two hypotheses have equal priors:

$$
P(A)=P(A^c),
$$

then differences in their posterior probabilities are driven directly by their likelihoods.

For example, if:

$$
P(B\mid A)=0.30
$$

and:

$$
P(B\mid A^c)=0.10,
$$

then $B$ is three times more likely under $A$ than under $A^c$.

With equal priors, observing $B$ therefore favors $A$.

Without equal priors, we must also account for how probable the hypotheses were before observing $B$.

---

## 7. Common Mistakes

### Confusing likelihood with joint probability

Likelihood:

$$
P(B\mid A)
$$

Joint probability:

$$
P(A\cap B)
$$

They are related by:

$$
P(A\cap B)
=
P(B\mid A)P(A),
$$

but they are not the same quantity.

### Confusing $P(A\mid B)$ with $P(B\mid A)$

In general:

$$
P(A\mid B)\neq P(B\mid A).
$$

Bayes' Theorem explains exactly how they are related.

### Ignoring the prior

A high likelihood does not automatically imply a high posterior.

The posterior combines the likelihood with the prior.

---

## What I Should Be Able to Explain

Without looking at the formula, I should be able to explain:

1. What problem Bayes' Theorem solves.
2. Why $P(A\mid B)$ and $P(B\mid A)$ are different.
3. How Bayes' Theorem follows from the definition of conditional probability.
4. Why $P(A\cap B)$ is the key quantity in the derivation.
5. What the **prior** represents.
6. What the **likelihood** represents.
7. Why the likelihood is not a joint probability.
8. What the **evidence** represents.
9. What the **posterior** represents.
10. Why observing evidence can increase or decrease the probability of a hypothesis.
11. Why likelihood alone is not sufficient to determine which hypothesis has the largest posterior.

---

## Formula to Reconstruct

Rather than memorizing Bayes' Theorem directly, start from:

$$
P(A\mid B)
=
\frac{P(A\cap B)}{P(B)}
$$

and:

$$
P(B\mid A)
=
\frac{P(A\cap B)}{P(A)}.
$$

Both contain the same joint probability:

$$
P(A\cap B).
$$

From there, Bayes follows:

$$
\boxed{
P(A\mid B)
=
\frac{P(B\mid A)P(A)}
{P(B)}
}
$$

---

## Connections

- [[Conditional Probability]]
- [[Law of Total Probability]]
- [[Partition]]
- [[Independence]]
- [[Random Variables]]