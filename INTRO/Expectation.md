# Expectation and Variance

## Expectation

The **expected value** (or **expectation**) of a random variable describes the mean of its probability distribution.

For a discrete random variable:

$$
\boxed{
\mathbb E[X]
=
\sum_x x\,P(X=x)
}
$$

or, using the PMF:

$$
\boxed{
\mathbb E[X]
=
\sum_x x\,p_X(x)
}
$$

The expectation is therefore a **probability-weighted average**.

---

## Expected Value = Mean of the Distribution

For a random variable, the terms:

- expected value;
- expectation;
- mean of the distribution;

refer to the same quantity.

The mean is often denoted by:

$$
\boxed{
\mu=\mathbb E[X]
}
$$

It is a weighted mean because different values of $X$ may have different probabilities.

---

## Example

Suppose:

$$
X=
\begin{cases}
1 & P(X=1)=0.25\\
3 & P(X=3)=0.50\\
5 & P(X=5)=0.25
\end{cases}
$$

Then:

$$
\mathbb E[X]
=
1(0.25)+3(0.50)+5(0.25)
$$

$$
=0.25+1.50+1.25
$$

$$
\boxed{\mathbb E[X]=3}
$$

---

## Expectation Does Not Have to Be a Possible Value

Consider:

$$
X=
\begin{cases}
0 & P=0.5\\
1 & P=0.5
\end{cases}
$$

Then:

$$
\mathbb E[X]
=
0(0.5)+1(0.5)
=
0.5.
$$

But $X$ can only assume:

$$
X\in\{0,1\}.
$$

Therefore $0.5$ is not a possible value of $X$.

So:

$$
\boxed{
\mathbb E[X]
\text{ does not have to be a possible value of }X
}
$$

Expectation represents the theoretical mean of the probability distribution.

---

# Expectation of a Function of a Random Variable

This is an important point.

If we transform $X$ using some function $g$, then:

$$
g(X)
$$

is itself a random variable.

For a discrete random variable:

$$
\boxed{
\mathbb E[g(X)]
=
\sum_x g(x)P(X=x)
}
$$

The correct procedure is:

$$
\boxed{
X
\rightarrow
g(X)
\rightarrow
\mathbb E[g(X)]
}
$$

In words:

1. determine the transformed value $g(x)$ for every possible $x$;
2. keep the corresponding probability $P(X=x)$;
3. multiply each transformed value by its probability;
4. sum the contributions.

---

## Example: $\mathbb E[X^2]$

Suppose:

$$
X=
\begin{cases}
2 & P=0.9\\
10 & P=0.1
\end{cases}
$$

First transform $X$:

$$
2^2=4
$$

$$
10^2=100.
$$

Therefore:

$$
X^2=
\begin{cases}
4 & P=0.9\\
100 & P=0.1
\end{cases}
$$

Now calculate the expectation:

$$
\mathbb E[X^2]
=
4(0.9)+100(0.1)
$$

$$
=13.6.
$$

### Important

In general:

$$
\boxed{
\mathbb E[X^2]\neq(\mathbb E[X])^2
}
$$

These expressions perform the operations in a different order.

### $\mathbb E[X^2]$

First square, then take the expectation:

$$
X
\rightarrow
X^2
\rightarrow
\mathbb E[X^2].
$$

### $(\mathbb E[X])^2$

First take the expectation, then square:

$$
X
\rightarrow
\mathbb E[X]
\rightarrow
(\mathbb E[X])^2.
$$

---

# Variance

Expectation tells us about the **center** of a probability distribution.

However, two random variables can have the same expectation while having very different spreads.

Variance measures how much the values of a random variable tend to spread around their mean.

Let:

$$
\mu=\mathbb E[X].
$$

The deviation of $X$ from its mean is:

$$
X-\mu.
$$

Because deviations can be positive or negative, simply averaging them would cause cancellation.

We therefore square them:

$$
(X-\mu)^2.
$$

Variance is the expectation of these squared deviations:

$$
\boxed{
\operatorname{Var}(X)
=
\mathbb E[(X-\mathbb E[X])^2]
}
$$

or equivalently:

$$
\boxed{
\operatorname{Var}(X)
=
\mathbb E[(X-\mu)^2]
}
$$

### Meaning

> **Variance is the expected squared deviation from the mean.**

Equivalently:

> Variance is the probability-weighted average of the squared distances from the mean.

---

# Computing Variance for a Discrete Random Variable

Expanding the expectation gives:

$$
\boxed{
\operatorname{Var}(X)
=
\sum_x
(x-\mu)^2P(X=x)
}
$$

where:

$$
\mu=\mathbb E[X].
$$

The procedure is:

1. calculate $\mu=\mathbb E[X]$;
2. calculate $x-\mu$ for every possible value $x$;
3. square each deviation;
4. multiply each squared deviation by $P(X=x)$;
5. sum the contributions.

---

## Example

Suppose:

$$
X=
\begin{cases}
0 & P(X=0)=0.2\\
2 & P(X=2)=0.5\\
4 & P(X=4)=0.3
\end{cases}
$$

First check:

$$
0.2+0.5+0.3=1.
$$

### Step 1 — Expectation

$$
\mathbb E[X]
=
0(0.2)+2(0.5)+4(0.3)
$$

$$
=0+1+1.2
$$

$$
\boxed{\mu=2.2}
$$

### Step 2 — Deviations from the Mean

$$
0-2.2=-2.2
$$

$$
2-2.2=-0.2
$$

$$
4-2.2=1.8.
$$

### Step 3 — Squared Deviations

$$
(-2.2)^2=4.84
$$

$$
(-0.2)^2=0.04
$$

$$
(1.8)^2=3.24.
$$

### Step 4 — Expectation of the Squared Deviations

$$
\operatorname{Var}(X)
=
4.84(0.2)+0.04(0.5)+3.24(0.3)
$$

$$
=0.968+0.020+0.972
$$

$$
\boxed{\operatorname{Var}(X)=1.96}
$$

---

# Alternative Variance Formula

Variance can also be calculated using:

$$
\boxed{
\operatorname{Var}(X)
=
\mathbb E[X^2]-(\mathbb E[X])^2
}
$$

For the previous example:

$$
\mathbb E[X^2]
=
0^2(0.2)+2^2(0.5)+4^2(0.3)
$$

$$
=0+2+4.8
$$

$$
=6.8.
$$

Also:

$$
(\mathbb E[X])^2
=
2.2^2
=
4.84.
$$

Therefore:

$$
\operatorname{Var}(X)
=
6.8-4.84
=
\boxed{1.96}.
$$

The two formulas give the same result.

---

# Important Distinction: What Is Inside $\mathbb E[\cdot]$?

Always inspect the expression **inside the expectation operator before doing any calculation**.

For example:

$$
\mathbb E[X]
$$

means:

> compute the probability-weighted average of $X$.

---

$$
\mathbb E[X^2]
$$

means:

> square the possible values of $X$, then compute their probability-weighted average.

---

$$
\mathbb E[(X-3)^2]
$$

means:

> subtract $3$ from each possible value of $X$, square the result, then compute the probability-weighted average.

---

$$
\mathbb E[(X-\mathbb E[X])^2]
$$

means:

> compute the mean of $X$, subtract it from every possible value of $X$, square the deviations, and finally compute their probability-weighted average.

This last quantity is exactly:

$$
\operatorname{Var}(X).
$$

---

# Error-Control Rule

When an expression contains several nested operations, do not compress the calculation mentally.

Read and execute it **from the inside outward**.

For example:

$$
\mathbb E[(X-\mu)^2]
$$

should be read as:

$$
\boxed{
X
\rightarrow
X-\mu
\rightarrow
(X-\mu)^2
\rightarrow
\mathbb E[(X-\mu)^2]
}
$$

Only after understanding this sequence should numerical calculation begin.

For a discrete expectation, remember:

$$
\boxed{
\mathbb E[g(X)]
=
\sum_x g(x)P(X=x)
}
$$

The probability weights are applied when computing the expectation; they are **not part of the transformation $g(x)$**.

---

# Quick Sanity Checks

Before accepting a calculation:

1. Do the probabilities sum to $1$?
2. Did I identify exactly what is inside $\mathbb E[\cdot]$?
3. Did I perform the transformation before applying the probability weights?
4. Did every transformed value keep the probability associated with its original value of $X$?
5. Is the variance non-negative?

Since variance is an average of squared quantities:

$$
\boxed{
\operatorname{Var}(X)\geq0
}
$$

---

# Machine Learning Connection

Expectation appears throughout Machine Learning because we often care about the average behavior of a quantity over a probability distribution.

For example, if:

- $X$ represents the input;
- $Y$ represents the target;
- $f(X)$ represents a model prediction;
- $\ell(f(X),Y)$ represents the prediction loss;

then:

$$
\mathbb E[\ell(f(X),Y)]
$$

represents the expected loss over the underlying data distribution.

The same rule applies:

$$
(X,Y)
\rightarrow
\ell(f(X),Y)
\rightarrow
\mathbb E[\ell(f(X),Y)].
$$

---

# What I Need to Know

## Understand deeply

- Expectation is the probability-weighted mean of a random variable.
- $\mathbb E[\cdot]$ is an operator that can be applied to a random variable or a function of random variables.
- Variance measures spread around the mean.
- Variance is the expected squared deviation from the mean.
- The order of operations inside and outside $\mathbb E[\cdot]$ matters.

## Be able to apply

For a discrete random variable:

$$
\mathbb E[X]
=
\sum_x xP(X=x)
$$

and:

$$
\operatorname{Var}(X)
=
\mathbb E[(X-\mathbb E[X])^2].
$$

## Recognize

$$
\operatorname{Var}(X)
=
\mathbb E[X^2]-(\mathbb E[X])^2.
$$

## Do not confuse

$$
\boxed{
\mathbb E[X^2]\neq(\mathbb E[X])^2
}
$$

in general.

---

# Active Recall

Without looking at the formulas, I should be able to explain:

1. Why is expectation a weighted average?
2. Why can $\mathbb E[X]$ be a value that $X$ never assumes?
3. What does $\mathbb E[g(X)]$ mean?
4. What is the difference between $\mathbb E[X^2]$ and $(\mathbb E[X])^2$?
5. Why do we square deviations when defining variance?
6. What does $\mathbb E[(X-\mathbb E[X])^2]$ mean step by step?
7. What information does variance provide that expectation does not?

---

# Connections

- [[Random Variables]]
- [[Probability Distribution]]
- [[Joint Distributions]]
- [[Covariance]]
