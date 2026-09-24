# Joint Distributions

## Core Idea

A **joint probability distribution** describes how two random variables behave **together**.

For two discrete random variables $X$ and $Y$, the joint probability mass function is:

$$
\boxed{
p_{X,Y}(x,y)=P(X=x,Y=y)
}
$$

It assigns a probability to **every possible pair** $(x,y)$.

The comma means **AND**:

$$
P(X=x,Y=y)
$$

means:

> the probability that $X=x$ AND $Y=y$.

It must not be confused with conditioning:

$$
P(X=x\mid Y=y),
$$

where $|$ means **GIVEN**.

---

# Joint Probability vs Joint Distribution

A **joint probability** is one particular value:

$$
P(X=x,Y=y).
$$

A **joint distribution** is the collection of joint probabilities for **all possible combinations** of $X$ and $Y$.

Think:

$$
\boxed{
\text{Joint probability}=\text{one cell}
}
$$

$$
\boxed{
\text{Joint distribution}=\text{the entire table}
}
$$

See also [[Joint Probability]].

---

# Example

Consider:

| | $Y=0$ | $Y=1$ |
|---|---:|---:|
| $X=0$ | $0.40$ | $0.10$ |
| $X=1$ | $0.10$ | $0.40$ |

For example:

$$
P(X=0,Y=0)=0.40
$$

and:

$$
P(X=1,Y=0)=0.10.
$$

Each individual cell is a **joint probability**.

The complete table is the **joint distribution** of $X$ and $Y$.

Because it describes all possible pairs:

$$
\boxed{
\sum_x\sum_y p_{X,Y}(x,y)=1
}
$$

For the example:

$$
0.40+0.10+0.10+0.40=1.
$$

---

# Marginal Distributions

Suppose we have the complete joint distribution but are interested only in $X$.

We can obtain the distribution of $X$ by summing over every possible value of $Y$:

$$
\boxed{
p_X(x)
=
\sum_y p_{X,Y}(x,y)
}
$$

This operation is called **marginalization**.

Conceptually:

> Fix $X=x$ and sum over all possible values of $Y$.

We say that we **sum out $Y$**.

Similarly:

$$
\boxed{
p_Y(y)
=
\sum_x p_{X,Y}(x,y)
}
$$

sums out $X$.

---

## Example

Using:

| | $Y=0$ | $Y=1$ |
|---|---:|---:|
| $X=0$ | $0.40$ | $0.10$ |
| $X=1$ | $0.10$ | $0.40$ |

we obtain:

$$
P(X=0)
=
P(X=0,Y=0)+P(X=0,Y=1)
$$

$$
=0.40+0.10
=0.50.
$$

Similarly:

$$
P(X=1)=0.10+0.40=0.50.
$$

Therefore the marginal distribution of $X$ is:

$$
P(X=0)=0.50,
\qquad
P(X=1)=0.50.
$$

The same operation can be performed for $Y$.

---

# Conditional Distributions

From the joint distribution we can also obtain a **conditional distribution**.

For discrete random variables:

$$
\boxed{
p_{X|Y}(x|y)
=
\frac{p_{X,Y}(x,y)}{p_Y(y)}
}
$$

provided:

$$
p_Y(y)>0.
$$

This is simply the random-variable version of [[Conditional Probability]]:

$$
P(A\mid B)
=
\frac{P(A\cap B)}{P(B)}.
$$

---

## Interpretation

When conditioning on:

$$
Y=y,
$$

we restrict the reference universe to cases where $Y=y$.

The probabilities in the joint distribution were measured relative to the **original universe**.

We therefore divide by:

$$
P(Y=y)
$$

to **renormalize** the restricted universe so that its total probability becomes $1$.

---

## Example

Suppose:

| | $Y=0$ | $Y=1$ |
|---|---:|---:|
| $X=0$ | $0.30$ | $0.20$ |
| $X=1$ | $0.40$ | $0.10$ |

First:

$$
P(Y=1)=0.20+0.10=0.30.
$$

The joint probability:

$$
P(X=0,Y=1)=0.20
$$

is relative to the original population.

But:

$$
P(X=0\mid Y=1)
$$

uses $Y=1$ as the new reference universe.

Therefore:

$$
P(X=0\mid Y=1)
=
\frac{P(X=0,Y=1)}{P(Y=1)}
$$

$$
=
\frac{0.20}{0.30}
=
\frac23.
$$

So:

$$
\boxed{
P(X=0,Y=1)\neq P(X=0\mid Y=1)
}
$$

in general.

---

# Joint → Marginal → Conditional

The important conceptual map is:

$$
\boxed{
p_{X,Y}(x,y)
\quad\text{Joint Distribution}
}
$$

From the joint distribution we can obtain a marginal by **summing out** a variable:

$$
\boxed{
p_X(x)=\sum_y p_{X,Y}(x,y)
}
$$

and a conditional distribution by **restricting and renormalizing**:

$$
\boxed{
p_{X|Y}(x|y)
=
\frac{p_{X,Y}(x,y)}{p_Y(y)}
}
$$

Therefore:

$$
\boxed{
\text{Joint}
\begin{cases}
\xrightarrow{\text{sum out}} \text{Marginal}\\
\xrightarrow{\text{condition + normalize}} \text{Conditional}
\end{cases}
}
$$

---

# Product Rule

Rearranging the conditional probability formula:

$$
p_{X|Y}(x|y)
=
\frac{p_{X,Y}(x,y)}{p_Y(y)}
$$

gives:

$$
\boxed{
p_{X,Y}(x,y)
=
p_{X|Y}(x|y)p_Y(y)
}
$$

Similarly:

$$
\boxed{
p_{X,Y}(x,y)
=
p_{Y|X}(y|x)p_X(x)
}
$$

Therefore:

$$
\boxed{
p_{X,Y}(x,y)
=
p_{X|Y}(x|y)p_Y(y)
=
p_{Y|X}(y|x)p_X(x)
}
$$

This is the same [[Joint Probability|product rule]] already seen for events.

---

# Independence

If $X$ and $Y$ are independent, their joint distribution factorizes:

$$
\boxed{
p_{X,Y}(x,y)
=
p_X(x)p_Y(y)
}
$$

for every possible pair $(x,y)$.

Equivalently, knowing $X$ does not change the distribution of $Y$:

$$
p_{Y|X}(y|x)=p_Y(y).
$$

See [[Independence]].

---

# Why Joint Distributions Matter for Expectation and Covariance

For a function of two random variables:

$$
g(X,Y),
$$

its expected value in the discrete case is:

$$
\boxed{
\mathbb E[g(X,Y)]
=
\sum_x\sum_y
g(x,y)p_{X,Y}(x,y)
}
$$

The joint distribution tells us how much probability weight to assign to each pair $(x,y)$.

For example:

$$
\boxed{
\mathbb E[XY]
=
\sum_x\sum_y
xy\,p_{X,Y}(x,y)
}
$$

This is why the joint distribution appears naturally in [[Covariance]]:

$$
\operatorname{Cov}(X,Y)
=
\mathbb E[XY]
-
\mathbb E[X]\mathbb E[Y].
$$

---

# Machine Learning Connection

In supervised learning we often have:

$$
X=\text{input/features}
$$

and:

$$
Y=\text{target/label}.
$$

The joint distribution:

$$
p(X,Y)
$$

describes how inputs and targets occur together.

The conditional distribution:

$$
p(Y\mid X)
$$

describes the distribution of the target given the observed input.

This distinction appears repeatedly throughout Machine Learning.

---

# What I Need to Know

## Understand deeply

- A joint distribution describes multiple random variables together.
- A joint probability is one entry of the joint distribution.
- The comma in $P(X=x,Y=y)$ means AND.
- Marginalization removes a variable by summing over its possible values.
- Conditioning changes the reference universe and requires normalization.
- Joint, marginal, and conditional distributions are different objects.

## Be able to apply

From:

$$
p_{X,Y}(x,y)
$$

derive a marginal:

$$
p_X(x)=\sum_y p_{X,Y}(x,y)
$$

and a conditional:

$$
p_{X|Y}(x|y)
=
\frac{p_{X,Y}(x,y)}{p_Y(y)}.
$$

## Recognize

Product rule:

$$
p_{X,Y}(x,y)
=
p_{X|Y}(x|y)p_Y(y).
$$

Independence:

$$
p_{X,Y}(x,y)
=
p_X(x)p_Y(y).
$$

---

# Connections

- [[Random Variables]]
- [[Probability Distribution]]
- [[Joint Probability]]
- [[Conditional Probability]]
- [[Independence]]
- [[Expectation and Variance]]
- [[Covariance]]
- [[Bayes' Theorem]]