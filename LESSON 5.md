---
title: "Lecture 5 — Linear Classifiers, VC Dimension, and Perceptron"
aliases:
  - "Lecture 5 - 09-10-2026"
date: 2026-10-09
course: "Machine Learning — Sapienza"
academic_year: "2026/2027"
type: lecture-note
status: "Content covered; full unaided proofs and delayed review pending"
tags: [machine-learning, linear-classifiers, vc-dimension, perceptron, linear-algebra, convergence]
---

# Lecture 5 — Linear Classifiers and the Perceptron

> [!NOTE] Learning state (29 October 2026)
> We covered the lecture's main contents with guided reasoning. The linear-VC-dimension argument was also tested on a new numerical example, with a sign-precision correction. The Perceptron convergence proof was understood with guidance but **not reproduced independently in full**. Do not interpret these notes as proof of exam readiness.

**Course sources (2026/27 confirmed):**
- [Handwritten Lecture 5 — 9 October 2026](https://drive.google.com/file/d/1zdPHTD2z3FDlUc7fI06fBtRR3TZNp0qE/view): primary narrative, VC dimension, Perceptron, bound, closing affine observation.
- [Week 2 — Linear Classifiers I / The Perceptron Algorithm, version 2](https://drive.google.com/file/d/1FhP2Vy1u108UjZF8PI0ynV21cHQJrzeZ/view): §1 pp. 1–3; §2 pp. 3–5; formal notation and proof.
- **Study reconstruction:** explanations, step-by-step algebra, numerical exercise, mistakes and corrections are our study work, **not direct quotations from the professor**. Later affine material continues in [[LESSON 6]].

**Prerequisites:** [[Shattering and VC Dimension]], [[LESSON 4]], scalar products, linear dependence, Euclidean norms. In this lecture, labels are $\{-1,+1\}$.

## 1. Linear (homogeneous) classifiers

For $x,w\in\mathbb R^d$ define

$$
h_w(x)=\operatorname{sign}(\langle x,w\rangle),
\qquad
\langle x,w\rangle=\sum_{j=1}^{d}x_jw_j.
$$

The course handout uses $\operatorname{sign}(z)=+1$ if $z\ge0$, and $-1$ otherwise. This convention matters in the upper-bound proof. For $w\neq 0$, the decision boundary is the hyperplane

$$H_w=\{x\in\mathbb R^d:\langle w,x\rangle=0\}.$$

The vector $w$ is **normal (perpendicular)** to the boundary. Its direction determines its orientation, and its sign determines which side receives label $+1$. Multiplying $w$ by a **positive** constant changes the signed scores but **not** the classifications or decision boundary; multiplication by a negative constant reverses the non-boundary class assignments.

Geometric identity:

$$
\langle w,x\rangle=\|w\|_2\|x\|_2\cos\theta.
$$

Thus a signed score is not only an angle: it also depends on vector lengths. See the subsequent treatment of geometric distance in [[LESSON 6]].

## 2. Why VCdim of homogeneous linear classifiers in $\mathbb R^d$ equals $d$

**Theorem (Week 2, 1.2):**

$$\boxed{\operatorname{VCdim}(\mathcal H_{\mathrm{lin}})=d.}$$

As with every exact VC-dimension proof, establish a **lower bound by constructing one shattered set**, then an **upper bound by ruling out every set with $d+1$ points**. Recall the quantified definition in [[Shattering and VC Dimension]].

### Lower bound: construct $d$ points

Use the canonical basis $S=\{e_1,\ldots,e_d\}$, where $e_i$ is $1$ in coordinate $i$ and $0$ elsewhere.

Fix **any** desired labels $y_1,\ldots,y_d\in\{-1,+1\}$. Choose $w=(y_1,\ldots,y_d)$. For each $i$,

$$
\langle w,e_i\rangle=w_i=y_i,\qquad
h_w(e_i)=\operatorname{sign}(y_i)=y_i.
$$

For every assignment there is such a $w$. Hence $S$ is shattered and $\operatorname{VCdim}(\mathcal H_{\mathrm{lin}})\ge d$.

> [!TIP] Recovered prerequisite
> The canonical basis works because dotting with $e_i$ **selects the $i$th coordinate**. Setting $w_i=y_i$ is a construction for this particular basis, not a rule for arbitrary points.

### Upper bound: an arbitrary set of $d+1$ points

Take **any** $d+1$ vectors $v_0,\ldots,v_d\in\mathbb R^d$. They are linearly dependent, so some $\lambda_0,\ldots,\lambda_d$, not all zero, satisfy

$$
\sum_{i=0}^{d}\lambda_i v_i=0.
$$

We may negate all coefficients, so assume **at least one** $\lambda_i<0$. Assign label $y_i=-1$ whenever $\lambda_i<0$, and $y_i=+1$ otherwise. Suppose a classifier $h_w$ realizes these labels. With the specified sign convention:

- If $\lambda_i<0$, then $\langle w,v_i\rangle<0$ (strictly), so $\lambda_i\langle w,v_i\rangle>0$.
- If $\lambda_i\ge0$, then $\langle w,v_i\rangle\ge0$, so $\lambda_i\langle w,v_i\rangle\ge0$.

Consequently at least one term is **strictly positive**, all others nonnegative:

$$
\sum_{i=0}^{d}\lambda_i\langle w,v_i\rangle>0.
$$

But by linearity,

$$
\sum_{i=0}^{d}\lambda_i\langle w,v_i\rangle
=\left\langle w,\sum_{i=0}^{d}\lambda_i v_i\right\rangle
=\langle w,0\rangle=0.
$$

Contradiction. So the chosen labeling is impossible for **every** set of $d+1$ points, and VCdim is at most $d$. Combine both bounds.

> [!WARNING] Actual study difficulties
> Do not confuse **some coefficients not zero** with **every coefficient nonzero**. Keep $>0$ and $\ge0$ distinct; a sum of nonnegative terms is strictly positive only if at least one term is strictly positive. The upper bound is universal: one specific non-shattered configuration is insufficient.

### Independent exercise actually attempted

For $v_1=(1,2)$, $v_2=(2,-1)$, $v_3=(2,7)$, the student found independently

$$
-\frac{16}{5}v_1+\frac{3}{5}v_2+v_3=0.
$$

The labels $(-1,+1,+1)$ would force the corresponding three products $\lambda_i\langle w,v_i\rangle$ to be strictly positive, nonnegative, nonnegative — a contradiction. Conversely $(+1,-1,-1)$ gives a strictly negative sum, also impossible. The logical choices were correct; written sign inequalities needed correction. **Status:** application to this concrete example operational after correction; generic proof still needs a blank-page check.

## 3. Statistical meaning and computational question

Finite VC dimension allows statistical uniform-convergence/generalization guarantees under the conditions of the earlier theorems (see [[LESSON 4]]). Its dependence on $d$ matters: high-dimensional models can require many samples. **But a sample-complexity guarantee does not itself give an efficient algorithm** for finding an ERM. Lecture 5 therefore asks how to compute a separating linear classifier.

## 4. The Perceptron: what it solves

The algorithm is used on a **finite strictly linearly separable** sample:

$$
\exists u\in\mathbb R^d:\quad y_i\langle u,x_i\rangle>0
\quad\forall i.
$$

This is stronger than the mere ability to classify zero-score boundary points as $+1$ under the chosen sign convention. With finitely many examples, the minimum signed score

$$m=\min_i y_i\langle u,x_i\rangle>0$$

is positive. Rescale **theoretical separator** $u$ to $u/m$ to obtain

$$y_i\langle u/m,x_i\rangle\ge1\quad\forall i.$$

This does **not** improve classification and is **not an algorithmic operation**; it gives a uniform unit of progress for the proof. A different positive constant could be used, but $1$ simplifies arithmetic.

**Algorithm (Week 2 §2).** Initialize $w^{(1)}=0$. As long as there is an example $i_t$ with nonpositive signed score $y_{i_t}\langle w^{(t)},x_{i_t}\rangle\le0$, update

$$\boxed{w^{(t+1)}=w^{(t)}+y_{i_t}x_{i_t}.}$$

Return the current $w$ when no such example remains.

The signed score of **the selected point** changes by

$$
y_{i_t}\langle x_{i_t},w^{(t+1)}\rangle
=y_{i_t}\langle x_{i_t},w^{(t)}\rangle+\|x_{i_t}\|_2^2.
$$

This is a **nonnegative improvement** for that point, not a guarantee it is already correctly classified after a single update. Scores on other points can worsen. Convergence therefore needs a global argument.

## 5. Novikoff's update bound (Week 2, Theorem 2.3)

Define

$$
R=\max_i\|x_i\|_2,\qquad
B=\min\{\|w\|_2:y_i\langle x_i,w\rangle\ge1\;\forall i\}.
$$

Choose a minimum-norm feasible vector $w^\star$ with $\|w^\star\|_2=B$. Let $T$ be the **number of updates**, not the number of complete loop checks. Then

$$\boxed{T\le R^2B^2.}$$

**Proof map:** (i) Each update increases the projection against a *fixed* valid reference separator by at least $1$. (ii) The squared length of the changing Perceptron weight can grow by at most $R^2$ per update. (iii) Cauchy–Schwarz combines these otherwise incomparable statements.

### (i) Lower bound on progress toward a reference

By the update rule and normalization,

$$
\begin{aligned}
\langle w^\star,w^{(t+1)}\rangle
-\langle w^\star,w^{(t)}\rangle
&=\langle w^\star,y_{i_t}x_{i_t}\rangle\\
&=y_{i_t}\langle w^\star,x_{i_t}\rangle\\
&\ge1.
\end{aligned}
$$

Summing these differences **telescopes** (intermediate terms cancel). Since $w^{(1)}=0$,

$$
\boxed{\langle w^\star,w^{(T+1)}\rangle\ge T.}
$$

A large dot product alone does **not** prove the angle is getting smaller: norms can change too. It is a lower bound on a projection-like quantity, not a monotone angle/distance statement.

### (ii) Upper bound on squared norm growth

Use $\|a+b\|_2^2=\|a\|_2^2+2\langle a,b\rangle+\|b\|_2^2$:

$$
\begin{aligned}
\|w^{(t+1)}\|_2^2
&=\|w^{(t)}+y_{i_t}x_{i_t}\|_2^2\\
&=\|w^{(t)}\|_2^2
 +2y_{i_t}\langle w^{(t)},x_{i_t}\rangle
 +\|x_{i_t}\|_2^2\\
&\le\|w^{(t)}\|_2^2+R^2.
\end{aligned}
$$

The inequality follows from the **update condition** (the middle term is $\le0$) and the definition of $R$. Telescope again from the zero initialization:

$$\boxed{\|w^{(T+1)}\|_2^2\le TR^2,\qquad
\|w^{(T+1)}\|_2\le R\sqrt T.}$$

### (iii) Combine with Cauchy–Schwarz

$$
T\le\langle w^\star,w^{(T+1)}\rangle
\le\|w^\star\|_2\|w^{(T+1)}\|_2
\le BR\sqrt T.
$$

If $T>0$, divide by $\sqrt T$ and square to obtain $T\le B^2R^2$; for $T=0$ the claim is immediate. The bound applies to any prefix, ruling out more updates and establishing termination under strict separability.

**Important scope:** It bounds updates, **not directly machine instructions**. A naive scan for a violating point costs $O(nd)$, giving $O(nd(T+1))$ for the handout implementation. It does **not** solve empirical 0–1-loss minimization on arbitrary nonseparable datasets.

> [!WARNING] Personal misconceptions corrected
> We used the dot product with $w^\star$ to quantify guaranteed progress, **not** because Perceptron directly reduces the distance to an optimal separator at every step. The two growth bounds must be combined with a *product of norms*, not a sum. In one numerical checkpoint, $R=2$ and $B=3$ produce $R^2B^2=36$, not $13$. The student caught this attention error after correction. Another checkpoint was $R=3,T=4\Rightarrow\|w^{(T+1)}\|\le6$.

## 6. Closing lecture observation: affine classifiers

The hand-written lecture ends by noting that the affine family

$$h_{w,b}(x)=\operatorname{sign}(\langle w,x\rangle+b)$$

can be treated as linear in augmented space using $\phi(x)=(x,1)$ and $\widetilde w=(w,b)$:

$$\langle\widetilde w,\phi(x)\rangle=\langle w,x\rangle+b.$$

In $\mathbb R^d$, the homogeneous class has VCdim $d$, whereas the **affine class has VCdim $d+1$**. This relation was introduced, **not independently tested**. The corresponding material begins [[LESSON 6]].

## Review / active recall (do these without the notes)

1. Prove the exact VC dimension using **one** shattered canonical-basis set and **arbitrary** $d+1$ points. Justify the strict sign in the contradiction.
2. For a fresh 3-vector set in $\mathbb R^2$, find a linear dependence and derive one impossible labeling.
3. Explain why scaling $w^\star$ to make all signed scores $\ge1$ does not change the classifier.
4. Reconstruct the three stages of the Perceptron update-bound proof on paper, explaining precisely where strict separability, update condition, telescoping and Cauchy–Schwarz enter.
5. Explain why improved score of the chosen example does not imply fewer total classification mistakes after each update.
6. Show the feature augmentation identity and distinguish affine from homogeneous separators.

**Next target:** independently reconstruct the VC proof and Perceptron bound after a delay, then continue [[LESSON 6]]. Avoid treating a guided explanation as a completed exam simulation.
