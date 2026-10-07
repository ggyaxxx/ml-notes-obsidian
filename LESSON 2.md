---
title: "Lecture 2 - Decision Theory, Bayes Predictor, Empirical Risk and Overfitting"
date: 2026-09-29
course: "Machine Learning - Sapienza"
status: "studied"
tags:
  - machine-learning
  - decision-theory
  - bayes-predictor
  - risk
  - overfitting
---

# Lecture 2 — Decision Theory, Bayes Predictor, Empirical Risk and Overfitting

## Conceptual map

This lecture develops the decision-theoretic view of supervised learning:

$$
\text{distribution } \mathcal D
\rightarrow
\text{loss } \ell
\rightarrow
\text{expected risk}
\rightarrow
\text{Bayes predictor}
\rightarrow
\text{Bayes risk}
\rightarrow
\text{empirical risk}
\rightarrow
\text{generalization / overfitting}.
$$

The central question is:

> Given uncertainty about the label $Y$ once an input $X=x$ is observed, which prediction should we make in order to minimize expected loss?

---

## 1. Supervised-learning setup

We assume a joint distribution

$$
\mathcal D
$$

over

$$
\mathcal X \times \mathcal Y.
$$

The random pair

$$
(X,Y)\sim\mathcal D
$$

represents a generic example drawn from the data-generating process.

Keep the object types distinct:

- $\mathcal X$: input / feature space;
- $\mathcal Y$: output / label space;
- $X,Y$: random variables;
- $x,y$: realizations of those random variables;
- $f:\mathcal X\to\mathcal Y$: predictor;
- $f(x)$: prediction produced at a specific input $x$.

A training sample is

$$
S=\{(x_1,y_1),\ldots,(x_m,y_m)\},
$$

or, probabilistically,

$$
(X_1,Y_1),\ldots,(X_m,Y_m)\overset{\mathrm{i.i.d.}}{\sim}\mathcal D.
$$

> **Pitfall:** $S$ is not a subset of $\mathcal D$.  
> $\mathcal D$ is a probability distribution; $S$ is a finite sample of realizations drawn according to it.

---

## 2. Loss and expected risk

A loss function quantifies how costly a prediction is:

$$
\ell:\mathcal Y\times\mathcal Y\to\mathbb R.
$$

### Convention used in `Week 0.pdf`

The official handout uses

$$
\ell(\text{prediction},\text{true label}).
$$

Therefore:

$$
\ell(f(X),Y).
$$

The expected risk of a predictor $f$ is

$$
\boxed{
R_{\mathcal D}(f)
=
\mathbb E[\ell(f(X),Y)]
}
$$

where the expectation is taken with respect to the underlying distribution $\mathcal D$.

Interpretation:

> $R_{\mathcal D}(f)$ is the average loss we would obtain when using $f$ on future examples generated according to $\mathcal D$.

### Important source convention

`Exercises 0.pdf` declares the opposite loss-argument convention:

$$
\ell(\text{true label},\text{prediction}),
$$

and consequently writes

$$
R(f)=\mathbb E[\ell(Y,f(X))].
$$

Always follow the convention explicitly declared by the source being used.

---

## 3. Conditioning on the input

Using the law of total expectation,

$$
R_{\mathcal D}(f)
=
\mathbb E_X
\left[
\mathbb E[\ell(f(X),Y)\mid X]
\right].
$$

At a fixed input $x$,

$$
\mathbb E[\ell(f(x),Y)\mid X=x]
$$

is the **conditional expected loss**.

This separates two levels:

- **local / pointwise problem:** which prediction is best at a given $x$?
- **global problem:** how good is the predictor on average over all possible $X$?

---

## 4. Bayes predictor

For a fixed input $x$, consider a candidate prediction

$$
z\in\mathcal Y.
$$

The Bayes predictor chooses a prediction minimizing conditional expected loss:

$$
\boxed{
f^\star(x)
\in
\arg\min_{z\in\mathcal Y}
\mathbb E[\ell(z,Y)\mid X=x]
}
$$

### Meaning of each symbol

- $x$: observed input;
- $z$: candidate prediction;
- $Y$: still-unknown random label;
- $\ell(z,Y)$: loss if we choose $z$;
- $\mathbb E[\ell(z,Y)\mid X=x]$: expected loss of choosing $z$ given the observed input;
- $\arg\min$: the set of values of $z$ attaining the minimum;
- $f^\star(x)$: one optimal prediction at $x$.

### Parameter during expectation, variable during optimization

A subtle but important point is that the candidate prediction $z$ does **not** have one absolute role. Its role depends on the operation being performed.

For a fixed input $x$, define the conditional risk

$
r_x(z)
=
\mathbb E[\ell(z,Y)\mid X=x].
$

While computing the expectation with respect to the still-random label $Y$, the candidate prediction $z$ is held fixed. In that sense, $z$ is a **parameter** of the expectation. For example,

$
\mathbb E[zY\mid X=x]
=
z\,\mathbb E[Y\mid X=x].
$

However, after the expectation has been evaluated or algebraically simplified, $r_x(z)$ is a function of $z$. At that stage, $z$ becomes the **optimization variable**:

$
f^\star(x)
\in
\arg\min_z r_x(z).
$

So the correct mental model is:

$
\boxed{
\text{fix }z
\;\longrightarrow\;
\text{compute the expected loss for that choice}
\;\longrightarrow\;
\text{vary }z\text{ to find the best choice}
}
$

An analogous calculus example is

$
g(a)=\int_0^{10}(t-a)^2\,dt.
$

During the integration, $t$ is the **variable of integration** and $a$ is treated as a fixed parameter. Once the integral is evaluated, $t$ disappears and the result is a function $g(a)$. Then $a$ is the variable with respect to which we can differentiate and optimize.

> **Key question:** whenever something is called a “constant” or a “variable”, ask: **with respect to which operation?**

This distinction is especially useful for squared-loss exercises, where a candidate prediction is constant with respect to the expectation over $Y$ but variable with respect to the outer minimization.

---

### Why $\in\arg\min$, not $=\min$?

`min` returns the **minimum value of the objective**.

`argmin` returns the **argument(s) at which that minimum is attained**.

Example:

$$
g(0)=0.8,\qquad g(1)=0.2.
$$

Then

$$
\min_z g(z)=0.2,
$$

whereas

$$
\arg\min_z g(z)=\{1\}.
$$

A predictor must output a label/action, not a risk value.

---

## 5. Why pointwise minimization gives the global optimum

For discrete $X,Y$,

$$
R_{\mathcal D}(f)
=
\sum_x\sum_y
\ell(f(x),y)
P(X=x,Y=y).
$$

Using

$$
P(X=x,Y=y)
=
P(Y=y\mid X=x)P(X=x),
$$

we obtain

$$
R_{\mathcal D}(f)
=
\sum_x
P(X=x)
\left[
\sum_y
\ell(f(x),y)
P(Y=y\mid X=x)
\right].
$$

The inner term is

$$
\mathbb E[\ell(f(x),Y)\mid X=x].
$$

Therefore

$$
R_{\mathcal D}(f)
=
\sum_x
P(X=x)
\mathbb E[\ell(f(x),Y)\mid X=x].
$$

The choice $f(x_1)$ affects only the term associated with $x_1$.  
Hence each conditional term can be minimized separately.

This yields

$$
f^\star(x)
\in
\arg\min_z
\mathbb E[\ell(z,Y)\mid X=x].
$$

---

## 6. Bayes risk

The expected risk of a Bayes-optimal predictor is called the **Bayes risk**:

$$
\boxed{
R_{\mathcal D}^\star
=
R_{\mathcal D}(f^\star)
}
$$

Equivalently,

$$
\boxed{
R_{\mathcal D}^\star
=
\mathbb E_X
\left[
\min_z
\mathbb E[\ell(z,Y)\mid X]
\right].
}
$$

Interpretation:

> The Bayes risk is the lowest expected risk theoretically achievable for the given distribution $\mathcal D$ and loss $\ell$.

It is a theoretical lower bound on prediction risk.

### Important distinction

$$
f^\star
$$

is the theoretically optimal predictor, whereas

$$
f_S
$$

denotes a predictor learned from the finite training sample $S$.

In general,

$$
f_S\neq f^\star.
$$

---

## 7. The three risks to keep distinct

These three quantities play different roles.

### Bayes risk

$$
\boxed{R_{\mathcal D}^\star}
$$

**Official name:** Bayes risk.

Meaning:

> The minimum expected risk theoretically achievable on $\mathcal D$.

It tells us **what is theoretically achievable**.

---

### Empirical risk

$$
\boxed{
R_S(f_S)
=
\frac1m
\sum_{i=1}^m
\ell(f_S(x_i),y_i)
}
$$

**Official name:** empirical risk.

Meaning:

> The average loss of the learned predictor on the observed training sample $S$.

It tells us **how well the learned predictor fits the observed sample**.

---

### Expected risk of the learned predictor

$$
\boxed{
R_{\mathcal D}(f_S)
=
\mathbb E[\ell(f_S(X),Y)]
}
$$

**Official name:** expected risk (also true risk) of the learned predictor.

Meaning:

> The expected loss of the specific predictor $f_S$ on new examples generated according to $\mathcal D$.

It tells us **how well our learned predictor really performs on the underlying distribution**.

---

### Two different comparisons

#### Generalization / overfitting

Compare

$$
\boxed{
R_S(f_S)
\quad\text{with}\quad
R_{\mathcal D}(f_S)
}
$$

A large gap suggests poor generalization / overfitting.

#### Distance from the theoretical optimum

Compare

$$
\boxed{
R_{\mathcal D}^\star
\quad\text{with}\quad
R_{\mathcal D}(f_S)
}
$$

This tells us how far the learned predictor is from the theoretically best achievable performance.

> **Do not use the Bayes risk to define overfitting.**

---

## 8. Bayes predictor under 0-1 loss

For classification,

$$
\ell(z,y)
=
\mathbf 1_{\{z\neq y\}}.
$$

For a fixed $x$,

$$
\mathbb E[\ell(z,Y)\mid X=x]
=
\sum_{y\in\mathcal Y}
\mathbf 1_{\{z\neq y\}}
P(Y=y\mid X=x).
$$

Terms with $y=z$ vanish because their loss is $0$. Therefore

$$
=
\sum_{y\neq z}
P(Y=y\mid X=x).
$$

Because the probabilities of all possible labels sum to $1$,

$$
\boxed{
\sum_{y\neq z}
P(Y=y\mid X=x)
=
1-P(Y=z\mid X=x).
}
$$

### Complement explanation

The two events

$$
\{Y=z\}
\qquad\text{and}\qquad
\{Y\neq z\}
$$

are complementary.

Hence

$$
P(Y\neq z\mid X=x)
=
1-P(Y=z\mid X=x).
$$

Example with

$$
\mathcal Y=\{0,1,2,3\},
\qquad z=2:
$$

$$
\sum_{y\neq2}P(Y=y\mid X=x)
=
P(Y=0\mid X=x)
+
P(Y=1\mid X=x)
+
P(Y=3\mid X=x),
$$

which equals

$$
1-P(Y=2\mid X=x).
$$

Therefore

$$
\arg\min_z
\left[
1-P(Y=z\mid X=x)
\right]
=
\arg\max_z
P(Y=z\mid X=x).
$$

So under 0-1 loss,

$$
\boxed{
f^\star(x)
\in
\arg\max_{y\in\mathcal Y}
P(Y=y\mid X=x)
}
$$

The Bayes predictor chooses a most probable conditional label.

> **Important:** choosing the most probable class is not the universal Bayes rule. It follows specifically from 0-1 loss.

---

## 9. Non-unique Bayes predictors

Suppose

$$
P(Y=0\mid X=x)
=
P(Y=1\mid X=x)
=
0.5.
$$

Then under 0-1 loss,

$$
\mathbb E[\ell(0,Y)\mid X=x]
=
0.5
$$

and

$$
\mathbb E[\ell(1,Y)\mid X=x]
=
0.5.
$$

Therefore

$$
\arg\min_{z\in\{0,1\}}
\mathbb E[\ell(z,Y)\mid X=x]
=
\{0,1\}.
$$

Thus

$$
f^\star(x)\in\{0,1\}.
$$

The Bayes predictor need not be unique, but every Bayes predictor attains the same Bayes risk.

---

## 10. Optimal does not mean perfect

Example:

$$
P(Y=1\mid X=x)=0.8,
\qquad
P(Y=0\mid X=x)=0.2.
$$

Under 0-1 loss:

If we predict $z=1$,

$$
\mathbb E[\ell(1,Y)\mid X=x]
=
0.8\cdot0+0.2\cdot1
=
0.2.
$$

If we predict $z=0$,

$$
\mathbb E[\ell(0,Y)\mid X=x]
=
0.8\cdot1+0.2\cdot0
=
0.8.
$$

Therefore

$$
f^\star(x)=1,
$$

but the minimum conditional risk is still

$$
0.2.
$$

Three different objects:

$$
P(Y=1\mid X=x)=0.8
$$

describes uncertainty,

$$
f^\star(x)=1
$$

is the decision,

and

$$
\mathbb E[\ell(f^\star(x),Y)\mid X=x]=0.2
$$

is the expected cost of that decision.

---

## 11. Continuous form

When $Y$ is continuous, sums are replaced by integrals.

For a fixed $x$,

$$
\mathbb E[\ell(z,Y)\mid X=x]
=
\int_{\mathcal Y}
\ell(z,y)
p_{Y\mid X=x}(y)\,dy.
$$

Therefore

$$
\boxed{
f^\star(x)
\in
\arg\min_{z\in\mathcal Y}
\int_{\mathcal Y}
\ell(z,y)
p_{Y\mid X=x}(y)\,dy
}
$$

The conceptual structure is unchanged:

> aggregate the losses associated with every possible $y$, weighted by how plausible each $y$ is given $X=x$.

---

## 12. Expected risk vs empirical risk

In practice, $\mathcal D$ is generally unknown.

We observe only a finite sample

$$
S=\{(x_1,y_1),\ldots,(x_m,y_m)\}.
$$

The **expected risk**

$$
R_{\mathcal D}(f)
=
\mathbb E[\ell(f(X),Y)]
$$

is the quantity we really care about, but cannot normally compute directly.

We instead observe the **empirical risk**

$$
\boxed{
R_S(f)
=
\frac1m
\sum_{i=1}^m
\ell(f(x_i),y_i)
}
$$

which is the average loss on the finite sample.

For 0-1 loss, if a classifier makes 7 errors on 100 training examples,

$$
R_S(f)=\frac7{100}=0.07.
$$

But

$$
R_S(f)=0.07
$$

does **not imply**

$$
R_{\mathcal D}(f)=0.07.
$$

The two values could happen to be equal, but equality cannot be inferred from the training sample alone.

---

## 13. Overfitting

A predictor overfits when it performs very well on the training sample but significantly worse on the underlying distribution.

Typical pattern:

$$
\boxed{
R_S(f_S)\ll R_{\mathcal D}(f_S).
}
$$

This is a **generalization problem**.

It must not be confused with Bayes / irreducible risk.

- Bayes risk asks: *How well could an optimal predictor theoretically do?*
- Overfitting asks: *How differently does my learned predictor perform on the training sample versus the underlying distribution?*

---

## 14. Geometric overfitting example

Let

$$
\mathcal X=[-1,1]^2
$$

and assume

$$
X\sim\operatorname{Unif}([-1,1]^2).
$$

Define the central square

$$
Q_\delta=[-\delta,\delta]^2.
$$

Let labels be deterministic:

$$
Y=
\begin{cases}
+1 & \text{if }X\in Q_\delta,\\
-1 & \text{if }X\notin Q_\delta.
\end{cases}
$$

### Bayes predictor

The optimal predictor is

$$
f^\star(x)
=
\begin{cases}
+1 & x\in Q_\delta,\\
-1 & x\notin Q_\delta.
\end{cases}
$$

It never makes a mistake, so

$$
\boxed{
R_{\mathcal D}^\star=0.
}
$$

The problem has no irreducible classification error.

---

### Memorizing predictor

Given

$$
S=\{(x_1,y_1),\ldots,(x_m,y_m)\},
$$

consider the predictor

$$
f_S(x)
=
\begin{cases}
+1 &
\text{if }x=x_i
\text{ for some positive training point},\\
-1 & \text{otherwise}.
\end{cases}
$$

It memorizes every positive training point and predicts $-1$ elsewhere.

It fits the training data perfectly:

$$
\boxed{
R_S(f_S)=0.
}
$$

However, $X$ is continuous.

For every individual training point $x_i$,

$$
P(X=x_i)=0.
$$

Hence a future input almost surely does not coincide exactly with one of the memorized points.

Inside $Q_\delta$, the true label is $+1$, but the memorizing predictor almost surely outputs $-1$.

Outside $Q_\delta$, both the true label and the predictor output are $-1$.

Therefore the classifier essentially makes an error exactly when

$$
X\in Q_\delta.
$$

So

$$
R_{\mathcal D}(f_S)
=
P(X\in Q_\delta).
$$

---

## 15. Law of total probability view of the error

Let

$$
A=\{f_S(X)\neq Y\}.
$$

Split the input space into

$$
B=\{X\in Q_\delta\},
\qquad
B^c=\{X\notin Q_\delta\}.
$$

By the law of total probability,

$$
P(A)
=
P(A\mid B)P(B)
+
P(A\mid B^c)P(B^c).
$$

For the memorizing classifier,

$$
P(A\mid B)=1
$$

and

$$
P(A\mid B^c)=0.
$$

Therefore

$$
P(A)
=
1\cdot P(X\in Q_\delta)
+
0\cdot P(X\notin Q_\delta),
$$

hence

$$
\boxed{
R_{\mathcal D}(f_S)
=
P(X\in Q_\delta).
}
$$

Interpretation of the weighting:

> conditional error in a region  
> $\times$  
> probability of being in that region.

---

## 16. Computing $P(X\in Q_\delta)$

The full square

$$
[-1,1]^2
$$

has side length

$$
2
$$

and area

$$
4.
$$

The central square

$$
Q_\delta=[-\delta,\delta]^2
$$

has side length

$$
2\delta
$$

and area

$$
(2\delta)^2=4\delta^2.
$$

Because $X$ is uniform,

$$
P(X\in Q_\delta)
=
\frac{\operatorname{Area}(Q_\delta)}
{\operatorname{Area}([-1,1]^2)}.
$$

Therefore

$$
P(X\in Q_\delta)
=
\frac{4\delta^2}{4}
=
\boxed{\delta^2}.
$$

Thus

$$
\boxed{
R_{\mathcal D}(f_S)=\delta^2.
}
$$

The three key quantities in the example are therefore

$$
\boxed{
R_{\mathcal D}^\star=0,
\qquad
R_S(f_S)=0,
\qquad
R_{\mathcal D}(f_S)=\delta^2.
}
$$

Interpretation:

- $R_{\mathcal D}^\star=0$: a perfect predictor exists;
- $R_S(f_S)=0$: the memorizing predictor fits the training sample perfectly;
- $R_{\mathcal D}(f_S)=\delta^2$: the same predictor performs worse on new data.

This is overfitting in a problem whose Bayes risk is zero.

---

## 17. Generalization gap vs distance from Bayes risk

Suppose

$$
R_{\mathcal D}^\star=0.10,
\qquad
R_S(f_S)=0.02,
\qquad
R_{\mathcal D}(f_S)=0.25.
$$

### Generalization / overfitting comparison

$$
R_S(f_S)=0.02
$$

versus

$$
R_{\mathcal D}(f_S)=0.25.
$$

The large gap suggests overfitting.

### Optimality comparison

$$
R_{\mathcal D}^\star=0.10
$$

versus

$$
R_{\mathcal D}(f_S)=0.25.
$$

The learned predictor is also substantially worse than the theoretical optimum.

These are two distinct questions.

---

## 18. Percentage points vs relative percentage

If

$$
R_{\mathcal D}^\star=0.10
$$

and

$$
R_{\mathcal D}(f_S)=0.13,
$$

the **absolute difference** is

$$
0.13-0.10=0.03,
$$

i.e. **3 percentage points**.

The relative increase with respect to the Bayes risk is

$$
\frac{0.13-0.10}{0.10}=0.30,
$$

i.e. **30%**.

Do not confuse percentage points with relative percentages.

---

# Key formulas

### Expected risk

$$
R_{\mathcal D}(f)
=
\mathbb E[\ell(f(X),Y)].
$$

### Conditional decomposition

$$
R_{\mathcal D}(f)
=
\mathbb E_X
\left[
\mathbb E[\ell(f(X),Y)\mid X]
\right].
$$

### Bayes predictor

$$
f^\star(x)
\in
\arg\min_{z\in\mathcal Y}
\mathbb E[\ell(z,Y)\mid X=x].
$$

### Bayes risk

$$
R_{\mathcal D}^\star
=
R_{\mathcal D}(f^\star)
=
\mathbb E_X
\left[
\min_z
\mathbb E[\ell(z,Y)\mid X]
\right].
$$

### Bayes predictor under 0-1 loss

$$
f^\star(x)
\in
\arg\max_{y\in\mathcal Y}
P(Y=y\mid X=x).
$$

### Empirical risk

$$
R_S(f)
=
\frac1m
\sum_{i=1}^m
\ell(f(x_i),y_i).
$$

### Geometric overfitting example

$$
R_{\mathcal D}^\star=0,
\qquad
R_S(f_S)=0,
\qquad
R_{\mathcal D}(f_S)=\delta^2.
$$

---

# Common pitfalls

1. **Confusing a distribution with a dataset**
   - Wrong: “$S$ is a subset of $\mathcal D$.”
   - Correct: $S$ is a finite sample drawn according to $\mathcal D$.

2. **Confusing a probability with a prediction**
   - $P(Y=1\mid X=x)=0.8$ is a probability.
   - $f^\star(x)=1$ is a prediction.

3. **Confusing `min` and `argmin`**
   - `min` returns the minimum loss value.
   - `argmin` returns the prediction(s) attaining it.

4. **Writing $P(Y\mid X)$ when a numerical event probability is intended**
   - $P(Y\mid X)$: conditional distribution.
   - $P(Y=y\mid X=x)$: probability of a specific label given a specific input.

5. **Using $x\neq y$ to express classification error**
   - $x$ is an input and $y$ is a label.
   - The correct error event is
     $$
     f(X)\neq Y.
     $$

6. **Confusing Bayes risk with overfitting**
   - Overfitting concerns
     $$
     R_S(f_S)
     \quad\text{vs}\quad
     R_{\mathcal D}(f_S).
     $$
   - Bayes optimality concerns
     $$
     R_{\mathcal D}^\star
     \quad\text{vs}\quad
     R_{\mathcal D}(f_S).
     $$

7. **Saying “the model learned perfectly” when $R_S=0$**
   - Better:
     > the model fits the training sample perfectly.

8. **Assuming $R_S(f)=a\Rightarrow R_{\mathcal D}(f)=a$**
   - Empirical and expected risk may differ.

---

# Connections

- [[LESSON 1]]
- Conditional probability
- Law of total probability
- Law of total expectation
- Bayes decision theory
- Generalization
- Overfitting
- Empirical Risk Minimization

---

# Active recall checklist

Without looking at the formulas, be able to explain:

- [ ] What is the difference between $\mathcal D$ and $S$?
- [ ] What does $R_{\mathcal D}(f)$ measure?
- [ ] What does $R_S(f)$ measure?
- [ ] What problem does the Bayes predictor solve?
- [ ] Why does the Bayes rule minimize risk pointwise in $x$?
- [ ] What is the difference between $f^\star$ and $f_S$?
- [ ] What is the Bayes risk $R_{\mathcal D}^\star$?
- [ ] Why can $R_{\mathcal D}^\star>0$?
- [ ] Why can a Bayes predictor be non-unique?
- [ ] Why does 0-1 loss lead to the conditional mode?
- [ ] Why is
  $$
  \sum_{y\neq z}P(Y=y\mid X=x)
  =
  1-P(Y=z\mid X=x)?
  $$
- [ ] Why does $R_S(f_S)=0$ not imply good generalization?
- [ ] Which comparison diagnoses overfitting?
- [ ] Which comparison tells us how far we are from the theoretical optimum?
- [ ] In the geometric example, why is $R_{\mathcal D}(f_S)=\delta^2$?

---

# Mastery target

For exam-level mastery, be able to:

1. derive the discrete Bayes predictor from expected risk;
2. derive the 0-1-loss Bayes rule;
3. distinguish conditional expected loss from global expected risk;
4. explain Bayes risk and irreducible error;
5. distinguish empirical risk from expected risk;
6. identify overfitting from risk values;
7. reconstruct the geometric memorization example and compute its true risk;
8. use precise notation for random variables, realizations, probabilities, predictors, and risks.
