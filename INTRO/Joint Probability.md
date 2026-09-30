# Joint Probability

## Core Idea

The **joint probability** is the probability that two events occur together.

For events $A$ and $B$:

$$
\boxed{
P(A\cap B)
}
$$

means:

> The probability that both $A$ **AND** $B$ occur.

The symbol:

$$
\cap
$$

means **intersection**.

---

## Connection with Conditional Probability

From [[Conditional Probability]]:

$$
P(A\mid B)
=
\frac{P(A\cap B)}{P(B)}
$$

Rearranging:

$$
\boxed{
P(A\cap B)
=
P(A\mid B)P(B)
}
$$

Similarly:

$$
P(B\mid A)
=
\frac{P(A\cap B)}{P(A)}
$$

therefore:

$$
\boxed{
P(A\cap B)
=
P(B\mid A)P(A)
}
$$

Thus:

$$
\boxed{
P(A\cap B)
=
P(A\mid B)P(B)
=
P(B\mid A)P(A)
}
$$

This relationship is sometimes called the **product rule**.

---

## Why the Joint Probability Is the Same in Both Directions

Intersection is commutative:

$$
A\cap B=B\cap A
$$

Therefore:

$$
P(A\cap B)=P(B\cap A).
$$

This fact is the key step used to derive [[Bayes' Theorem]].

---

## Example

Suppose:

$$
P(M)=0.40
$$

and:

$$
P(D\mid M)=0.10.
$$

Then:

$$
P(D\cap M)
=
P(D\mid M)P(M)
$$

$$
=
0.10\cdot0.40
=
0.04.
$$

Therefore:

$$
\boxed{
P(D\cap M)=0.04
}
$$

means that **4% of the entire population belongs to $M$ AND has $D$**.

---

## Joint Probability vs Conditional Probability

These represent different questions.

### Joint Probability

$$
P(A\cap B)
$$

asks:

> What is the probability that $A$ and $B$ both occur?

The reference universe is still the original sample space $\Omega$.

### Conditional Probability

$$
P(A\mid B)
$$

asks:

> Given that $B$ occurred, what is the probability that $A$ also occurs?

Conditioning changes the reference universe from:

$$
\Omega
$$

to:

$$
B.
$$

Therefore:

$$
\boxed{
P(A\cap B)\neq P(A\mid B)
}
$$

in general.

---

## Special Case: Independent Events

If $A$ and $B$ are [[Independence|independent]], then:

$$
P(A\mid B)=P(A).
$$

Therefore the product rule becomes:

$$
\boxed{
P(A\cap B)=P(A)P(B)
}
$$

Important:

$$
P(A\cap B)=P(A)P(B)
$$

is **not** the general rule for joint probability.

It is valid when $A$ and $B$ are independent.

The general rule is:

$$
P(A\cap B)=P(A\mid B)P(B).
$$

---

## Event Notation vs Random Variable Notation

For events:

$$
P(A\cap B)
$$

means $A$ AND $B$.

Later, with [[Random Variables]], we will encounter notation such as:

$$
P(X=x,Y=y).
$$

This also represents a **joint probability**:

> $X=x$ AND $Y=y$.

The comma does **not** mean "given".

Conditional probability instead uses:

$$
P(X=x\mid Y=y).
$$

where:

$$
\mid
$$

means **given**.

This distinction will become important when studying [[Joint Distributions]].

---

## What I Should Be Able to Explain

1. What does $P(A\cap B)$ mean?
2. Why is it called a joint probability?
3. How is it related to conditional probability?
4. Why is $P(A\cap B)=P(B\cap A)$?
5. What is the difference between $P(A\cap B)$ and $P(A\mid B)$?
6. When can I write $P(A\cap B)=P(A)P(B)$?
7. Why was joint probability the key step in deriving Bayes' theorem?

---

## Connections

- [[Probability Toolkit #1 — Foundations]]
- [[Conditional Probability]]
- [[Independence]]
- [[Law of Total Probability]]
- [[Bayes' Theorem]]
- [[Random Variables]]
- [[Joint Distributions]]