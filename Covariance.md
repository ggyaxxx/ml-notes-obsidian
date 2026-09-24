# Covariance

## Core Idea

The [[Variance]] of a random variable measures how much that variable tends to deviate from its own mean.

For two random variables $X$ and $Y$, **covariance** measures whether their deviations from their respective means tend to have the same or opposite signs.

Let:

$$
\mu_X = \mathbb E[X]
$$

and:

$$
\mu_Y = \mathbb E[Y].
$$

The covariance is defined as:

$$
\boxed{
\operatorname{Cov}(X,Y)
=
\mathbb E[(X-\mu_X)(Y-\mu_Y)]
}
$$

or equivalently:

$$
\boxed{
\operatorname{Cov}(X,Y)
=
\mathbb E[
(X-\mathbb E[X])
(Y-\mathbb E[Y])
]
}
$$

---

# Intuition

Consider the product:

$$
(X-\mu_X)(Y-\mu_Y).
$$

There are four possibilities:

| $X-\mu_X$ | $Y-\mu_Y$ | Product |
|---|---|---|
| $+$ | $+$ | $+$ |
| $-$ | $-$ | $+$ |
| $+$ | $-$ | $-$ |
| $-$ | $+$ | $-$ |

Therefore:

- if $X$ and $Y$ tend to deviate from their means in the **same direction**, covariance tends to be positive;
- if they tend to deviate in **opposite directions**, covariance tends to be negative.

The expectation averages these products over the probability distribution.

---

# Interpretation

## Positive covariance

$$
\operatorname{Cov}(X,Y)>0
$$

means that $X$ and $Y$ tend to deviate from their respective means in the same direction.

Informally:

$$
X\uparrow \quad Y\uparrow
$$

and:

$$
X\downarrow \quad Y\downarrow.
$$

---

## Negative covariance

$$
\operatorname{Cov}(X,Y)<0
$$

means that they tend to deviate in opposite directions.

Informally:

$$
X\uparrow \quad Y\downarrow
$$

or:

$$
X\downarrow \quad Y\uparrow.
$$

---

## Zero covariance

$$
\operatorname{Cov}(X,Y)=0
$$

means that the positive and negative co-deviation contributions cancel on average.

It indicates absence of linear co-variation as measured by covariance.

It does **not** necessarily imply independence.

---

# Covariance and the Joint Distribution

Covariance concerns two random variables simultaneously.

Therefore, for discrete random variables, its calculation requires their [[Joint Distributions|joint distribution]]:

$$
p_{X,Y}(x,y)
=
P(X=x,Y=y).
$$

Expanding the expectation:

$$
\boxed{
\operatorname{Cov}(X,Y)
=
\sum_x\sum_y
(x-\mu_X)(y-\mu_Y)
P(X=x,Y=y)
}
$$

For every possible pair $(x,y)$:

1. compute $x-\mu_X$;
2. compute $y-\mu_Y$;
3. multiply the deviations;
4. multiply by the joint probability $P(X=x,Y=y)$;
5. sum over all possible pairs.

Conceptually:

$$
(X,Y)
\rightarrow
(X-\mu_X)(Y-\mu_Y)
\rightarrow
\mathbb E[(X-\mu_X)(Y-\mu_Y)].
$$

---

# Computational Formula

A fundamental equivalent formula is:

$$
\boxed{
\operatorname{Cov}(X,Y)
=
\mathbb E[XY]
-
\mathbb E[X]\mathbb E[Y]
}
$$

This is often more convenient for calculations.

It is **not a separate definition**.

It follows directly from:

$$
\operatorname{Cov}(X,Y)
=
\mathbb E[(X-\mu_X)(Y-\mu_Y)].
$$

---

# Derivation of the Computational Formula

Start from the definition:

$$
\operatorname{Cov}(X,Y)
=
\mathbb E[(X-\mu_X)(Y-\mu_Y)].
$$

Expand the product:

$$
(X-\mu_X)(Y-\mu_Y)
$$

$$
=
XY-X\mu_Y-\mu_XY+\mu_X\mu_Y.
$$

Therefore:

$$
\operatorname{Cov}(X,Y)
=
\mathbb E[
XY-X\mu_Y-\mu_XY+\mu_X\mu_Y
].
$$

Using the **linearity of expectation**:

$$
\mathbb E[A+B]
=
\mathbb E[A]+\mathbb E[B],
$$

we obtain:

$$
\operatorname{Cov}(X,Y)
=
\mathbb E[XY]
-
\mathbb E[X\mu_Y]
-
\mathbb E[\mu_XY]
+
\mathbb E[\mu_X\mu_Y].
$$

Now $\mu_X$ and $\mu_Y$ are constants, because:

$$
\mu_X=\mathbb E[X]
$$

and:

$$
\mu_Y=\mathbb E[Y].
$$

Constants can be moved outside the expectation:

$$
\mathbb E[cX]
=
c\mathbb E[X].
$$

Therefore:

$$
\mathbb E[X\mu_Y]
=
\mu_Y\mathbb E[X],
$$

$$
\mathbb E[\mu_XY]
=
\mu_X\mathbb E[Y],
$$

and:

$$
\mathbb E[\mu_X\mu_Y]
=
\mu_X\mu_Y.
$$

Substituting:

$$
\operatorname{Cov}(X,Y)
=
\mathbb E[XY]
-
\mu_Y\mathbb E[X]
-
\mu_X\mathbb E[Y]
+
\mu_X\mu_Y.
$$

But:

$$
\mathbb E[X]=\mu_X
$$

and:

$$
\mathbb E[Y]=\mu_Y.
$$

Therefore:

$$
\operatorname{Cov}(X,Y)
=
\mathbb E[XY]
-
\mu_X\mu_Y
-
\mu_X\mu_Y
+
\mu_X\mu_Y.
$$

The last three terms simplify:

$$
-\mu_X\mu_Y
-\mu_X\mu_Y
+\mu_X\mu_Y
=
-\mu_X\mu_Y.
$$

Hence:

$$
\boxed{
\operatorname{Cov}(X,Y)
=
\mathbb E[XY]
-
\mu_X\mu_Y
}
$$

and since:

$$
\mu_X=\mathbb E[X],
\qquad
\mu_Y=\mathbb E[Y],
$$

we finally obtain:

$$
\boxed{
\operatorname{Cov}(X,Y)
=
\mathbb E[XY]
-
\mathbb E[X]\mathbb E[Y]
}
$$

---

# Understanding $\mathbb E[XY]$

The quantity:

$$
\mathbb E[XY]
$$

means:

> first multiply $X$ and $Y$, then calculate the expectation of the resulting random variable.

For discrete $X$ and $Y$:

$$
\boxed{
\mathbb E[XY]
=
\sum_x\sum_y
xyP(X=x,Y=y)
}
$$

Again, the [[Joint Distributions|joint distribution]] is required because we need the probability associated with every pair $(x,y)$.

Do not confuse:

$$
\mathbb E[XY]
$$

with:

$$
\mathbb E[X]\mathbb E[Y].
$$

In general:

$$
\boxed{
\mathbb E[XY]
\neq
\mathbb E[X]\mathbb E[Y]
}
$$

---

# Covariance and Variance

Variance is a special case of covariance.

Set:

$$
Y=X.
$$

Then:

$$
\operatorname{Cov}(X,X)
=
\mathbb E[(X-\mu_X)(X-\mu_X)].
$$

Therefore:

$$
\operatorname{Cov}(X,X)
=
\mathbb E[(X-\mu_X)^2].
$$

But this is precisely the definition of variance:

$$
\boxed{
\operatorname{Cov}(X,X)
=
\operatorname{Var}(X)
}
$$

This provides an important conceptual connection:

- variance measures how one variable varies with itself;
- covariance measures how two variables vary together.

---

# Covariance and Independence

If $X$ and $Y$ are independent:

$$
P(X=x,Y=y)
=
P(X=x)P(Y=y).
$$

For independent random variables:

$$
\mathbb E[XY]
=
\mathbb E[X]\mathbb E[Y].
$$

Using:

$$
\operatorname{Cov}(X,Y)
=
\mathbb E[XY]
-
\mathbb E[X]\mathbb E[Y],
$$

we obtain:

$$
\operatorname{Cov}(X,Y)=0.
$$

Therefore:

$$
\boxed{
X\perp Y
\quad\Rightarrow\quad
\operatorname{Cov}(X,Y)=0
}
$$

assuming the relevant expectations exist.

---

# Zero Covariance Does NOT Imply Independence

The converse is not generally true:

$$
\boxed{
\operatorname{Cov}(X,Y)=0
\quad\not\Rightarrow\quad
X\perp Y
}
$$

Zero covariance only tells us that covariance detects no net linear co-variation.

Two variables may still have a strong nonlinear dependence.

---

## Counterexample

Suppose:

$$
P(X=-1)=P(X=0)=P(X=1)=\frac13
$$

and define:

$$
Y=X^2.
$$

Then $Y$ is completely determined by $X$, so $X$ and $Y$ are certainly not independent.

However:

$$
\mathbb E[X]=0.
$$

Since:

$$
XY=X^3,
$$

we have:

$$
\mathbb E[XY]
=
\mathbb E[X^3].
$$

Therefore:

$$
\mathbb E[XY]
=
(-1)^3\frac13
+
0^3\frac13
+
1^3\frac13
$$

$$
=
-\frac13+\frac13
=
0.
$$

Thus:

$$
\operatorname{Cov}(X,Y)
=
\mathbb E[XY]
-
\mathbb E[X]\mathbb E[Y]
$$

$$
=
0-0\cdot\mathbb E[Y]
=
0.
$$

So:

$$
\boxed{
\operatorname{Cov}(X,Y)=0
}
$$

despite the deterministic relationship:

$$
\boxed{
Y=X^2.
}
$$

This demonstrates that covariance does not detect every possible type of dependence.

---

# Independence vs Zero Covariance

Keep these two ideas separate:

### Independence

A property of the entire joint distribution:

$$
P(X,Y)=P(X)P(Y).
$$

Conceptually:

> Knowing one variable provides no probabilistic information about the other.

### Zero covariance

A property of one numerical summary:

$$
\mathbb E[(X-\mu_X)(Y-\mu_Y)]=0.
$$

Conceptually:

> The covariance detects no net linear co-variation.

Therefore independence is the stronger condition:

$$
\boxed{
\text{Independence}
\Rightarrow
\text{Zero covariance}
}
$$

but not generally:

$$
\boxed{
\text{Zero covariance}
\Rightarrow
\text{Independence}.
}
$$

---

# Calculation Checklist

Given a discrete joint distribution:

1. Check that:

$$
\sum_x\sum_yP(X=x,Y=y)=1.
$$

2. Obtain the marginals:

$$
P(X=x)=\sum_yP(X=x,Y=y)
$$

and:

$$
P(Y=y)=\sum_xP(X=x,Y=y).
$$

3. Calculate:

$$
\mu_X=\mathbb E[X]
$$

and:

$$
\mu_Y=\mathbb E[Y].
$$

4. Either use the definition:

$$
\operatorname{Cov}(X,Y)
=
\mathbb E[(X-\mu_X)(Y-\mu_Y)]
$$

or the computational identity:

$$
\operatorname{Cov}(X,Y)
=
\mathbb E[XY]
-
\mathbb E[X]\mathbb E[Y].
$$

---

# What I Need to Know

## Understand deeply

- Covariance measures whether two random variables tend to deviate from their respective means in the same or opposite directions.
- Why the product of deviations determines the sign.
- Why covariance requires information about the joint distribution.
- Why zero covariance does not necessarily imply independence.
- The relationship between variance and covariance.

## Be able to derive

Starting from:

$$
\operatorname{Cov}(X,Y)
=
\mathbb E[(X-\mu_X)(Y-\mu_Y)],
$$

derive:

$$
\boxed{
\operatorname{Cov}(X,Y)
=
\mathbb E[XY]
-
\mathbb E[X]\mathbb E[Y].
}
$$

## Be able to apply

For discrete random variables:

$$
\operatorname{Cov}(X,Y)
=
\sum_x\sum_y
(x-\mu_X)(y-\mu_Y)
P(X=x,Y=y).
$$

Also be able to use:

$$
\operatorname{Cov}(X,Y)
=
\mathbb E[XY]
-
\mathbb E[X]\mathbb E[Y].
$$

## Remember

$$
\boxed{
\operatorname{Cov}(X,X)=\operatorname{Var}(X)
}
$$

and:

$$
\boxed{
X\perp Y
\Rightarrow
\operatorname{Cov}(X,Y)=0
}
$$

but:

$$
\boxed{
\operatorname{Cov}(X,Y)=0
\not\Rightarrow
X\perp Y.
}
$$

---

# Connections

- [[Random Variables]]
- [[Probability Distribution]]
- [[Joint Distributions]]
- [[Expectation and Variance]]
- [[Independence]]
- [[Correlation]]
