---
title: "Lecture 6 — Affine Classifiers, Margins, and Support Vector Machines"
aliases:
  - "Lecture 6 - 13-10-2026"
date: 2026-10-13
course: "Machine Learning — Sapienza"
academic_year: "2026/2027"
type: lecture-note
status: "Full lecture reference (pp. 1–11); student guided study completed through geometric margin on p. 5 only"
tags: [machine-learning, affine-classifiers, vc-dimension, margin, svm, hard-svm, soft-svm, hinge-loss, optimization]
---

# Lecture 6 — Affine Classifiers, Margins, and Support Vector Machines

> [!IMPORTANT] Source vs learning progress
> This document covers **all 11 pages** of the professor's handwritten Lecture 6. It also supplies **tutor explanations and proofs**, especially for the early topics that were not explained in our study dialogue. **Reference coverage is NOT evidence of mastery.** As of 3 November 2026, the student has worked with the tutor through the **geometric margin of a classifier (p. 5)**, and has only *introduced* the optimal margin. **Hard-SVM, support-vector optimality, soft-SVM, and generalization results are not yet independently studied or verified.** Page 3 (agnostic ERM hardness) has been recovered in English notes but not tested independently.

**Authoritative source and order:** [Lezione 6 — 13/10/2026, handwritten pages 1–11](https://drive.google.com/file/d/1t0ElkUB94qu45AypUvmvv0gBrHFgpvtH/view).  
**Secondary formal reference:** *Week 3 — Linear Classifiers II / Support Vector Machines*, by Federico Fusco, especially §§1–3. The later textbook excerpt on handwritten p. 11 is from *Understanding Machine Learning*, Chapter 15. When notation differs, the **handwritten lecture is identified first** and alternative normalizations are labeled explicitly.  
**Related notes:** [[LESSON 5]] · [[Shattering and VC Dimension]]

## 1. Affine versus homogeneous linear classifiers — handwritten pp. 1–2

### 1.1 Definitions and geometry [Source: p. 1; tutor explanation added]

For an input $x\in\mathbb R^d$, a **homogeneous linear classifier** is

$$
h_w(x)=\operatorname{sign}(\langle w,x\rangle),\qquad w\in\mathbb R^d.
$$

For a nonzero $w$, its decision boundary is $H_w=\{x:\langle w,x\rangle=0\}$. Since $\langle w,0\rangle=0$, **it necessarily passes through the origin**.

An **affine classifier** includes a bias $b\in\mathbb R$:

$$
\boxed{h_{w,b}(x)=\operatorname{sign}(\langle w,x\rangle+b).}
$$

Its boundary is $\{x:\langle w,x\rangle+b=0\}$. The bias lets us translate the hyperplane without changing its normal direction $w$. For instance, in $\mathbb R^2$, $w=(1,0)$ and $b=-2$ produce the vertical boundary $x_1=2$, whereas $b=0$ gives $x_1=0$.

**Convention used in the course:** $\operatorname{sign}(0)=+1$. Always be careful with strict versus non-strict inequalities at the boundary.

### 1.2 Feature augmentation: why an affine classifier becomes linear in one extra dimension [Source: pp. 1–2; algebra explained]

Define the feature map and the augmented parameter vector:

$$
\phi:\mathbb R^d\longrightarrow\mathbb R^{d+1},\quad
\phi(x)=(x_1,\ldots,x_d,1),\qquad
\widetilde w=(w_1,\ldots,w_d,b).
$$

Their inner product is

$$
\begin{aligned}
\langle\widetilde w,\phi(x)\rangle
 &=\sum_{j=1}^{d}w_jx_j+b\cdot1\\
 &=\langle w,x\rangle+b.
\end{aligned}
$$

Therefore,

$$
\boxed{h_{w,b}(x)=h_{\widetilde w}(\phi(x)).}
$$

**Interpretation:** We have not changed the predictions. We have changed the *representation*: the intercept $b$ becomes another weight, multiplying a new constant feature $1$. An affine classifier on $\mathbb R^d$ is equivalent **as a decision rule** to a homogeneous classifier applied to these specially augmented features in $\mathbb R^{d+1}$.

> [!NOTE] Geometric nuance (tutor clarification)
> Equivalence of predictions under feature augmentation does **not** mean that Euclidean distances or norm penalties automatically remain equivalent. For an affine boundary measured in the *original* $\mathbb R^d$, distance uses $\|w\|_2$, not $\|(\,w,b\,)\|_2$. This matters later for affine SVM.

### 1.3 VC dimension of affine classifiers [Source: p. 2; explanatory proof added]

The handwritten lecture states

$$
\boxed{\operatorname{VCdim}(\mathcal H_{\mathrm{affine}})=d+1.}
$$

Compare this with homogeneous linear classifiers in $\mathbb R^d$, whose VC dimension is $d$ (see [[LESSON 5]]). **The extra bias adds one degree of shattering capacity.**

A concise way to see both bounds:

**Lower bound ($\ge d+1$).** Choose the $d+1$ points $0,e_1,\ldots,e_d$, where the $e_j$ are the canonical basis vectors. Given *any* labels $y_0,y_1,\ldots,y_d\in\{-1,+1\}$, set $b=y_0$ and $w_j=y_j-b$. Then

$$
\langle w,0\rangle+b=y_0,\qquad
\langle w,e_j\rangle+b=y_j.
$$

All labelings are realized; this set is shattered.

**Upper bound ($\le d+1$).** Any $d+2$ augmented vectors $(x_i,1)\in\mathbb R^{d+1}$ are linearly dependent. There are coefficients $\alpha_i$, not all zero, such that

$$
\sum_i\alpha_i x_i=0,\qquad \sum_i\alpha_i=0.
$$

In particular, there must be both positive and negative nonzero coefficients. Label the point with $\alpha_i>0$ as $+1$, and each with $\alpha_i<0$ as $-1$ (zero coefficients can receive any label). If an affine classifier realized those labels, its scores $s_i=\langle w,x_i\rangle+b$ would obey $s_i\ge0$ for positive coefficients and $s_i<0$ for negative coefficients. Thus $\sum_i\alpha_i s_i>0$; yet

$$
\sum_i\alpha_i s_i
=\Big\langle w,\sum_i\alpha_i x_i\Big\rangle+b\sum_i\alpha_i=0,
$$

a contradiction. Hence no $d+2$-point set is shattered.

**Status:** statement is in the handwritten lecture, but this proof is a **tutor-added reconstruction**. Autonomous proof remains to be checked.

### 1.4 The geometric meaning of $B$ from the Perceptron bound [Source: bottom of p. 2; tutor connection]

For a strictly separable dataset $S=\{(x_i,y_i)\}_{i=1}^{n}$, the quantity used in the Perceptron analysis can be written

$$
\boxed{
B=\min_{w}\Big\{\|w\|_2:\ y_i\langle w,x_i\rangle\ge1\quad\forall i\Big\}.
}
$$

This is the **smallest norm of a weight vector satisfying a fixed functional-margin normalization**, *not* the Euclidean distance of a data point from a boundary.

Why impose the threshold $1$? A strict separator can be multiplied by a sufficiently large **positive** constant until all functional scores $y_i\langle w,x_i\rangle$ reach at least $1$. The normalization removes freedom to shrink or expand the same classifier arbitrarily.

We will prove on p. 6 that, under strict separability,

$$
\boxed{B=\frac{1}{\gamma^\star}.}
$$

Thus $B$ is the reciprocal of the *optimal geometric margin*, not a separate geometric length.

## 2. Complexity of ERM in the agnostic setting — handwritten p. 3

### 2.1 The optimization problem [Source: p. 3; formulation expanded]

In the **agnostic setting**, we do not assume that a perfect linear separator exists. Under the $0$–$1$ loss, empirical risk is

$$
\widehat R_S(h_w)=\frac1n\sum_{i=1}^{n}
\mathbf 1\{h_w(x_i)\ne y_i\}.
$$

An exact ERM solution is any

$$
\boxed{w^\star\in\arg\min_{w\in\mathbb R^d}\widehat R_S(h_w).}
$$

It minimizes the number of training classification errors among the available linear classifiers; there is no assumption that this minimum is zero.

### 2.2 The professor's complexity statement [Source: p. 3]

**Exact 0–1 ERM for linear classifiers in the agnostic setting is NP-hard when dimension $d$ is part of the input.** The lecture also cites strong NP-hardness-of-approximation results for halfspaces with noise.

- Höffgen, Simon & Van Horn (1995), *Robust Trainability of Single Neurons*.
- Guruswami & Raghavendra (2009), *Hardness of Learning Halfspaces with Noise*.

The lecture presents the results and citations, **not a hardness reduction to prove**.

**Meaning of the terms (tutor explanation):** A decision problem belongs to **P** if some deterministic algorithm solves it in polynomial time. A decision problem belongs to **NP** if a proposed yes-certificate can be verified in polynomial time. A problem is **NP-hard** if every problem in NP can be polynomial-time reduced **to that problem**. NP-hard does not automatically mean NP-complete, nor does it prove there cannot be a polynomial algorithm unless P $\ne$ NP. Approximation hardness means that specified types/qualities of approximation are also computationally difficult; it does **not** mean every possible approximate algorithm is impossible.

### 2.3 Statistical vs computational difficulty [Tutor explanation connecting p. 2 and p. 3]

A finite VC dimension provides *statistical* learnability/generalization guarantees for ERM; it does **not** provide an efficient procedure to *compute* the ERM solution.

- **Statistical question:** How many i.i.d. examples are enough to estimate risks and generalize?
- **Computational question:** How much algorithmic work is needed to find a predictor minimizing empirical risk?

A class can be statistically learnable while exact 0–1 ERM is computationally difficult. This motivates changing the objective to one that can be optimized efficiently, rather than treating 0–1 ERM as an implementable black box.

> [!NOTE] Progress
> The student noticed that this entire p. 3 section was missing from personal notes. English notes were supplied and the point is now documented, but independent recall is not yet verified.

## 3. Finding the best separating hyperplane — handwritten pp. 4–5

### 3.1 Hyperplane and projection [Source: p. 4; derivation explained]

For nonzero $w\in\mathbb R^d$, define the homogeneous hyperplane

$$
H_w=\{z\in\mathbb R^d:\langle w,z\rangle=0\}.
$$

The vector $w$ is *normal* (perpendicular) to $H_w$. By the definition of inner product,

$$
\langle w,x\rangle=\|w\|_2\|x\|_2\cos\theta.
$$

The component of $x$ along the **unit normal** $w/\|w\|_2$ is its scalar projection:

$$
\underbrace{\frac{\langle w,x\rangle}{\|w\|_2}}_{\text{signed distance along the normal}}.
$$

Consequently, the **ordinary Euclidean distance** from $x$ to the hyperplane is

$$
\boxed{\operatorname{dist}(x,H_w)=\frac{|\langle w,x\rangle|}{\|w\|_2}.}
$$

The absolute value makes ordinary distance nonnegative. Remove it to retain orientation relative to $w$. In the special case $\|w\|_2=1$, the signed distance simplifies to $\langle w,x\rangle$.

**Optional rigorous projection argument (tutor addition).** The orthogonal projection onto $H_w$ is

$$
v=x-\frac{\langle w,x\rangle}{\|w\|_2^2}w.
$$

Since $\langle w,v\rangle=0$, $v\in H_w$, and

$$
\|x-v\|_2=\frac{|\langle w,x\rangle|}{\|w\|_2}.
$$

For any $u\in H_w$, Cauchy–Schwarz gives $|\langle w,x\rangle|=|\langle w,x-u\rangle|\le\|w\|_2\|x-u\|_2$, so no other point on the hyperplane is closer.

### 3.2 Scaling ambiguity [Discussed with tutor; p. 4 geometry]

The classifier $h_w(x)=\operatorname{sign}(\langle w,x\rangle)$ has **scaling ambiguity**. For $c>0$,

$$
\operatorname{sign}(\langle cw,x\rangle)
=\operatorname{sign}(c\langle w,x\rangle)
=\operatorname{sign}(\langle w,x\rangle).
$$

Thus $cw$ represents the **same classifier and the same decision boundary**, but the functional signed score is multiplied by $c$:

$$
r_i(w)=y_i\langle w,x_i\rangle,\qquad r_i(cw)=c\,r_i(w).
$$

Hence $r_i(w)$ **cannot itself be a geometric distance**: the distance from a fixed point to an unchanged boundary cannot vary under positive rescaling of the parameter vector. Normalize by $\|w\|_2$:

$$
\frac{\langle cw,x_i\rangle}{\|cw\|_2}
=\frac{\langle w,x_i\rangle}{\|w\|_2}\quad (c>0).
$$

For negative $c$ the classifier orientation flips, so positive rescaling is the relevant invariance for a *labeled* margin.

### 3.3 Ordinary distance, signed distance, and labeled geometric margin [Discussed with tutor]

Distinguish these **three different** quantities:

$$
\begin{aligned}
\text{Ordinary distance:}\quad&
\frac{|\langle w,x_i\rangle|}{\|w\|_2};\\[2pt]
\text{Signed distance along }w:\quad&
\frac{\langle w,x_i\rangle}{\|w\|_2};\\[2pt]
\text{Labeled geometric margin:}\quad&
\boxed{m_i(w)=\frac{y_i\langle w,x_i\rangle}{\|w\|_2}}.
\end{aligned}
$$

**The label $y_i$ multiplies the signed distance, NOT the absolute distance.** Example: if $y_i=-1$, $\langle w,x_i\rangle=-2$ and $\|w\|_2=1$, the correctly labeled margin is $(-1)(-2)=+2$. Multiplying $-1$ by the *absolute* distance $2$ would falsely give a negative value.

Away from the boundary, $m_i>0$ means the point lies on the correct side, while $m_i<0$ means it lies on the wrong side; $m_i=0$ means on the boundary, where predicted label depends on $\operatorname{sign}(0)$.

### 3.4 Geometric margin of the entire sample [Source: p. 5; discussed with tutor]

Given $S=\{(x_1,y_1),\ldots,(x_n,y_n)\}$, define the **geometric margin of classifier $w$ over the dataset**:

$$
\boxed{
\gamma(w)=\min_{1\le i\le n}
\frac{y_i\langle w,x_i\rangle}{\|w\|_2}.
}
$$

The **minimum over $i$** selects the most critical *oriented* training margin.

- $\gamma(w)>0$: every point has a positive margin; $w$ strictly separates the dataset.
- $\gamma(w)=0$: no margin is negative and at least one point lies on the boundary; actual tie classification depends on the sign convention.
- $\gamma(w)<0$: at least one point has negative signed margin and is misclassified.

**Important subtlety:** $\gamma(w)$ equals the *ordinary distance to the closest example* **only when the sample is strictly separated by $w$**. With misclassified points, the minimum chooses the most negative *oriented* margin, which need **not** be the Euclidean-nearest point.

**Checked example from study dialogue:**

$$
m_i\in\{1.2,\ 0.7,\ -0.3,\ 2.1\}
\quad\Longrightarrow\quad
\boxed{\gamma(w)=-0.3}.
$$

The student supplied this result correctly and emphasized the oriented distance. This confirms a worked application, not an independently proved SVM theorem.

### 3.5 Optimal margin: why a MAX of a MIN? [Source: p. 5; explanation for next study session]

Different vectors $w$ define different candidate separating hyperplanes. **For each fixed $w$**, compute the worst (minimum) margin among data points. **Across all $w$**, select the hyperplane whose worst margin is as large as possible:

$$
\boxed{
\gamma^\star
=\max_{w\ne0}\gamma(w)
=\max_{w\ne0}\min_i
\frac{y_i\langle w,x_i\rangle}{\|w\|_2}.
}
$$

This is a **maximin** optimization: choose the separator that improves the weakest geometric margin. When $S$ is strictly linearly separable, $\gamma^\star>0$.

> [!IMPORTANT] Exact study frontier
> The student has **just been introduced** to $\gamma^\star$; the maximization and its relationship with the Hard-SVM objective have **not yet been worked through together**. Resume here before using the derivation below.

## 4. Hard-SVM in the strictly separable case — handwritten pp. 5–6

### 4.1 Formulation [Source: bottom p. 5]

In the homogeneous case, the professor defines

$$
\boxed{
\begin{aligned}
\min_{w\in\mathbb R^d}\quad &\frac12\|w\|_2^2\\
\text{subject to}\quad &y_i\langle w,x_i\rangle\ge1,
\quad i=1,\ldots,n.
\end{aligned}
}
$$

The label constraints ensure a **strict separator** and normalize the *functional* scores. The quadratic objective is convex; the constraints are linear in $w$. Strict separability makes the problem feasible by positive rescaling.

### 4.2 Why minimizing $\|w\|_2$ maximizes geometric margin [Source: theorem and proof on p. 6; detailed explanation from Week 3]

Suppose $w$ is feasible. Then for every $i$, $y_i\langle w,x_i\rangle\ge1$, so

$$
\gamma(w)=\min_i\frac{y_i\langle w,x_i\rangle}{\|w\|_2}
\ge\frac1{\|w\|_2}.
$$

Now choose a **unit** optimal-margin normal vector $\widetilde w$ such that

$$
\|\widetilde w\|_2=1,\qquad
\min_i y_i\langle\widetilde w,x_i\rangle=\gamma^\star>0.
$$

Rescale it:

$$
\overline w=\frac{\widetilde w}{\gamma^\star}.
$$

Because $\min_i y_i\langle\overline w,x_i\rangle=1$, $\overline w$ is Hard-SVM feasible. Its norm is

$$
\|\overline w\|_2=\frac1{\gamma^\star}.
$$

Let $w^\star$ be a minimum-norm feasible vector. Then $\|w^\star\|_2\le\|\overline w\|_2$, and therefore

$$
\gamma^\star
\ge\gamma(w^\star)
\ge\frac1{\|w^\star\|_2}
\ge\frac1{\|\overline w\|_2}
=\gamma^\star.
$$

All inequalities must be equalities. Thus

$$
\boxed{\gamma(w^\star)=\gamma^\star=\frac1{\|w^\star\|_2}.}
$$

**Interpretation:** under the fixed functional-score normalization, smaller parameter norm means a larger *geometric* margin. The factor $1/2$ in the objective simplifies derivatives and does not change the optimizer.

### 4.3 Connecting $B$ to the Perceptron [Source: end of p. 6]

The previously defined $B$ solves the same minimum-norm constrained problem (without squaring the norm). Hence

$$
\boxed{B=\|w^\star\|_2=\frac1{\gamma^\star}.}
$$

In the Perceptron update bound $T\le R^2B^2$ with $R=\max_i\|x_i\|_2$, this gives

$$
T\le \left(\frac{R}{\gamma^\star}\right)^2.
$$

**Do not confuse the algorithms:** the Perceptron finds a strict separator when one exists; Hard-SVM specifically chooses the **maximum-margin** separator.

## 5. Support vectors and optimality conditions — handwritten p. 7

### 5.1 Why only some examples determine the boundary [Source: p. 7]

In Hard-SVM, define the set of **active margin constraints**:

$$
I=\{i:\ y_i\langle w^\star,x_i\rangle=1\}.
$$

These examples lie on the margin boundary at geometric distance $1/\|w^\star\|_2$ from $H_{w^\star}$. They are the geometric **support vectors** in the handwritten discussion. Points with $y_i\langle w^\star,x_i\rangle>1$ lie farther away and have inactive constraints.

At least one constraint is tight: otherwise one could scale $w^\star$ down slightly, keep every constraint feasible, and reduce its norm, contradicting optimality.

### 5.2 Stationarity derivation [Source: Fritz John lemma pasted on p. 7; clarified with formal Week 3 treatment]

Write the objective and inequality constraints:

$$
f(w)=\frac12\|w\|_2^2,\qquad
g_i(w)=1-y_i\langle w,x_i\rangle\le0.
$$

Their gradients are

$$
\boxed{\nabla f(w)=w,\qquad\nabla g_i(w)=-y_ix_i.}
$$

The **minus sign** in $\nabla g_i$ is essential. The handwritten page writes a normalized stationarity relation; the fully general Fritz John conditions initially allow an objective multiplier $\mu_0\ge0$. In strictly feasible Hard-SVM the normalized/KKT form is justified. We obtain multipliers $\alpha_i\ge0$ satisfying

$$
w^\star-\sum_{i=1}^n\alpha_i y_ix_i=0,
$$

together with complementary slackness

$$
\alpha_i\left(1-y_i\langle w^\star,x_i\rangle\right)=0.
$$

Therefore $\alpha_i=0$ whenever the constraint is inactive, and

$$
\boxed{w^\star=\sum_{i\in I}\alpha_i y_ix_i.}
$$

**Meaning:** the optimal separating normal can be expressed as a weighted sum of the labeled examples on the margin. An example *on* the margin may have $\alpha_i=0$ in degenerate configurations; the implementation-oriented definition often reserves “support vector” for $\alpha_i>0$.

> [!NOTE] Source distinction
> The handwritten p. 7 uses a pasted optimality lemma and gives the support-vector formula; the multiplier/constraint-qualification nuance and the explicit complementary-slackness proof here are **formal clarification from Week 3**, not material independently derived by the student.

## 6. Large margins and generalization — handwritten p. 8 (first half)

### 6.1 $(\gamma,\rho)$-separability [Source: p. 8, textbook excerpt]

A distribution $D$ over $(X,Y)$ is separable with margin $\gamma>0$ and input bound $\rho>0$ if there is a unit vector $u$ such that (almost surely)

$$
\|X\|_2\le\rho,\qquad Y\langle u,X\rangle\ge\gamma,
\qquad \|u\|_2=1.
$$

The handwritten page shows the input-radius and signed-margin conditions; the **unit-norm condition** is spelled out in the formal handout. This matters because otherwise multiplying $u$ by a positive constant would change the numerical margin without changing the classifier.

### 6.2 Hard-SVM generalization bound [Source: p. 8, Theorem 15.4 quoted; stated without proof]

Under the distributional margin/radius assumption, for an i.i.d. sample of size $m$, with probability at least $1-\delta$ the 0–1 true risk of the Hard-SVM output $\widehat w$ satisfies

$$
\boxed{
R_D(h_{\widehat w})
\le
\sqrt{\frac{4(\rho/\gamma)^2}{m}}
+
\sqrt{\frac{2\log(2/\delta)}{m}}.
}
$$

**Interpretation (tutor):** the bound decreases with sample size, worsens when the ratio $\rho/\gamma$ is large, and has no *explicit* factor depending on dimension $d$. The theorem assumes the **distribution** has the indicated bounded radius and margin; observing a large margin on one training sample alone does not prove that assumption. The professor cites the theorem; its proof is not given in the handwritten lecture.

## 7. Soft-SVM for nonseparable data — handwritten pp. 8–9

### 7.1 Why Hard-SVM is insufficient [Source: p. 8; tutor explanation]

For nonseparable data there is no $w$ for which every score $y_i\langle w,x_i\rangle\ge1$: the Hard-SVM constraints are **infeasible**. Soft-SVM introduces nonnegative **slack variables** $\xi_i$ that permit violations of these constraints while penalizing them.

### 7.2 Professor's Soft-SVM formulation [Source: p. 8]

The handwritten page uses the form

$$
\boxed{
\begin{aligned}
\min_{w,\xi}\quad&
\frac{\lambda}{2}\|w\|_2^2+\sum_{i=1}^{n}\xi_i\\
\text{subject to}\quad&
y_i\langle w,x_i\rangle\ge1-\xi_i,\quad i=1,\ldots,n,\\
&\xi_i\ge0.
\end{aligned}
}
$$

Here $\lambda>0$ is a hyperparameter balancing norm regularization against total constraint violation. **Larger $\lambda$ penalizes the parameter norm more strongly relative to slack.**

> [!NOTE] Do not silently swap conventions
> The handwritten lecture writes **$\lambda\|w\|_2^2/2+\sum_i\xi_i$**. The separate formal *Week 3* note uses **$\lambda\|w\|_2^2/2+(1/n)\sum_i\xi_i$**; that changes the relative parameter scaling unless $\lambda$ is adjusted. Some textbooks instead use $\|w\|_2^2/2+C\sum_i\xi_i$. These parameterizations are related but not numerically identical at fixed hyperparameter values.

### 7.3 Eliminating slack variables [Source: p. 9; proof explained]

For a fixed $w$, the two inequalities imply

$$
\xi_i\ge0,\qquad
\xi_i\ge1-y_i\langle w,x_i\rangle.
$$

As slack has a positive coefficient in the objective, choose the **smallest allowed** slack:

$$
\boxed{\xi_i^\star=\max\{0,\,1-y_i\langle w,x_i\rangle\}
=(1-y_i\langle w,x_i\rangle)_+.}
$$

There are two cases:

- If $y_i\langle w,x_i\rangle\ge1$, no violation: $\xi_i^\star=0$.
- If $y_i\langle w,x_i\rangle<1$, the minimum slack needed is $1-y_i\langle w,x_i\rangle>0$.

Substituting these optimal slacks into the professor's objective gives the unconstrained equivalent problem

$$
\boxed{
\min_{w\in\mathbb R^d}
\left[
\frac{\lambda}{2}\|w\|_2^2+
\sum_{i=1}^{n}(1-y_i\langle w,x_i\rangle)_+
\right].
}
$$

### 7.4 The hinge loss [Source: pp. 9–10]

Define the **hinge loss** using the signed functional score $r=y\langle w,x\rangle$:

$$
\boxed{\ell_{\mathrm{hinge}}(w;(x,y))=(1-y\langle w,x\rangle)_+
=\max\{0,1-r\}.}
$$

The hinge loss measures *margin violation*, **not just classification error**:

| Signed score $r$ | Optimal slack / hinge loss | Meaning |
|---|---|---|
| $r\ge1$ | $0$ | Correct side, on or outside the margin |
| $0<r<1$ | $1-r\in(0,1)$ | Correctly classified, but inside the margin |
| $r=0$ | $1$ | On the decision boundary; sign tie convention matters |
| $r<0$ | $1-r>1$ | Misclassified; wrong side of the boundary |

For example, $r=0.4$ means a correctly classified point **still incurs hinge loss $0.6$** because its margin is too small.

### 7.5 Why use hinge instead of 0–1 loss? [Source: pp. 9–10; tutor clarification]

For labeled scores, the empirical 0–1 error is bounded above by a function of the signed score:

$$
\mathbf1\{h_w(x)\ne y\}
\le\mathbf1\{y\langle w,x\rangle\le0\}
\le(1-y\langle w,x\rangle)_+.
$$

The first inequality accounts for the course's $\operatorname{sign}(0)=+1$ convention. The second follows because any nonpositive score incurs hinge loss at least $1$.

In addition,

$$
\ell_{\mathrm{hinge}}(w;(x_i,y_i))
=\max\big\{0,\,1-y_i\langle w,x_i\rangle\big\}
$$

is a **convex** function of $w$: the maximum of two affine functions is convex. With $\lambda>0$, the regularized objective has a convex quadratic part and is computationally tractable.

**Terminology nuance:** p. 10 informally calls hinge loss a “convexification” of the 0–1 loss. More precisely, it is a **convex surrogate upper bound**. Minimizing it is *not* in general the same objective as minimizing the number of errors.

## 8. Observations and soft-margin support vectors — handwritten p. 10

The professor stresses three observations:

1. Hinge loss is a convex surrogate for classification error.
2. Unlike direct agnostic 0–1 ERM, hinge-loss optimization is computationally tractable.
3. The hyperparameter $\lambda$ balances the hinge-loss term and norm regularization.

The figure illustrates that a Soft-SVM may tolerate points on the wrong side or inside the margin. The geometric idea of **support vectors extends to Soft-SVM**, but their set can include examples *inside* the margin and even misclassified examples. The handwritten page remarks on this without deriving full soft-margin multiplier conditions.

**Additional formal detail from Week 3 (not proved on handwritten p. 10):** with the formal handout's normalized penalty $(1/n)\sum_i\xi_i$ and parameter $\lambda$, multipliers may be chosen so that

$$
\lambda\widehat w=\sum_i\alpha_i y_ix_i,\qquad
0\le\alpha_i\le\frac1n.
$$

At the optimum, $\alpha_i=0$ for scores $r_i>1$, while $r_i<1$ implies $\alpha_i=1/n$. Points at $r_i=1$ may have a multiplier anywhere in $[0,1/n]$. **Do not transplant these numerical bounds unchanged into the handwritten unnormalized-sum convention.**

## 9. Soft-SVM generalization — handwritten p. 11

The professor attaches a textbook excerpt: *Understanding Machine Learning*, **Corollary 15.7**, stated **without proof**. The following formulas track the **book's own convention**, not the exact scale of $\lambda$ in the handwritten Soft-SVM objective on p. 8.

Let $\|X\|_2\le\rho$ almost surely, $S\sim D^m$, and $A(S)$ be the Soft-SVM output. For every fixed comparator $u$, the book states an expected hinge-risk bound

$$
\boxed{
\mathbb E_S\!\left[L_D^{\mathrm{hinge}}(A(S))\right]
\le
L_D^{\mathrm{hinge}}(u)
+\lambda\|u\|_2^2
+\frac{2\rho^2}{\lambda m}.
}
$$

Since hinge loss upper-bounds 0–1 classification loss,

$$
\mathbb E_S\!\left[L_D^{0\text{–}1}(A(S))\right]
\le
L_D^{\mathrm{hinge}}(u)
+\lambda\|u\|_2^2
+\frac{2\rho^2}{\lambda m}.
$$

If a comparator is restricted to $\|u\|_2\le B$ and we choose

$$
\lambda=\sqrt{\frac{2\rho^2}{B^2m}},
$$

the two remainder terms balance, giving the printed comparison

$$
\boxed{
\mathbb E_S[L_D^{0\text{–}1}(A(S))]
\le
\mathbb E_S[L_D^{\mathrm{hinge}}(A(S))]
\le
\min_{\|u\|_2\le B}L_D^{\mathrm{hinge}}(u)
+\sqrt{\frac{8\rho^2B^2}{m}}.
}
$$

**Interpretation (tutor):** expected risk of the learned classifier is compared to the best **bounded-norm hinge-risk** comparator, with an additional term decreasing like $1/\sqrt m$. This is an **expectation-over-training-samples** result, not a high-probability bound for every realized sample. The lecturer's excerpt uses labels $\{0,1\}$ while the lecture's classifier formulas use $\{-1,+1\}$; translate label conventions carefully.

## 10. Concept map, source audit, and current study checkpoint

**Conceptual progression, following the professor's pages:**

$$
\begin{gathered}
\text{Affine vs homogeneous (pp. 1–2)}
\longrightarrow
\text{VCdim}=d+1\text{ and }B\text{ (p. 2)}\\
\longrightarrow
\text{Agnostic 0–1 ERM is NP-hard (p. 3)}\\
\longrightarrow
\text{Hyperplane distance and scaling ambiguity (p. 4)}\\
\longrightarrow
\text{Geometric and optimal margin (p. 5)}\\
\longrightarrow
\text{Hard-SVM and max-margin proof (pp. 5–6)}\\
\longrightarrow
\text{Support vectors / optimality conditions (p. 7)}\\
\longrightarrow
\text{Margin generalization (p. 8)}\\
\longrightarrow
\text{Soft-SVM / slack / hinge (pp. 8–10)}\\
\longrightarrow
\text{Soft-SVM generalization (p. 11).}
\end{gathered}
$$

| Handwritten pages | What the professor covers | Actual learning status as of 2026-11-03 |
|---|---|---|
| 1–2 | Affine classifiers, augmentation, VC dimension, $B$ | Introduced/reviewed; autonomous checks pending |
| 3 | Agnostic ERM NP-hardness and approximation hardness | Missing notes recovered; recall not yet checked |
| 4 | Hyperplane distance, sign, scaling ambiguity | Explained and conceptually understood with tutor |
| 5 (first part) | Geometric margin $\gamma(w)$ | Explained; signed example correctly solved |
| 5 (second part) | Optimal margin $\gamma^\star$, Hard-SVM statement | Introduced only |
| 6–7 | Max-margin proof and support vectors | Reference notes included, **not yet studied together** |
| 8–11 | Generalization, Soft-SVM, hinge loss, observations | Reference notes included, **not yet studied together** |

### Resume here in the next guided session

**Stop point: handwritten p. 5**, immediately after defining $\gamma(w)$. First explain **why the optimal margin is a max over $w$ of a min over points**, using one geometric example if needed. Then introduce the bottom-of-p. 5 Hard-SVM optimization; derive the equivalence carefully using the p. 6 proof. Only afterward proceed to multipliers/support vectors and soft-margin.

**Targeted later recall:** (a) recreate the affine-feature-map identity and $d+1$ VC dimension argument, (b) explain $B$ and its later relation to $\gamma^\star$, (c) explain why finite VC dimension says nothing by itself about the runtime of 0–1 ERM, (d) distinguish unsigned distance, signed projection, functional score, and labeled geometric margin.

> [!TIP] Notation and study workflow
> Notes are in **English**, with Obsidian-friendly Markdown and $LaTeX$. For interactive tutoring, always give mathematical notes as **normal chat messages with rendered LaTeX**, never a WritingBlock. Keep the student in control of the pace, and never mark a theorem as independently mastered merely because it is present in the notes.
