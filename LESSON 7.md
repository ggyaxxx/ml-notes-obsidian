---
title: "Lecture 7 — Soft SVM, Hinge Loss, Subgradients, and the Descent Lemma"
aliases:
  - "Lecture 7"
  - "LESSON 7"
course: "Machine Learning — Sapienza"
academic_year: "2026/2027"
type: lecture-note
status: "Handwritten pp. 1–5 explained with guidance; independent recall and exam-style exercises pending"
tags: [machine-learning, soft-svm, hinge-loss, convexity, gradient-descent, subgradient, optimization, learning-rate]
---

# Lecture 7 — Soft SVM, Hinge Loss, Subgradients, and the Descent Lemma

> [!IMPORTANT] Scope and source discipline
> **2026/27 confirmed by the five handwritten pages shared in our study conversation (November 2026):** Soft SVM and hinge loss (p. 1); gradient descent and its motivation (p. 2); the definition and graphical interpretation of a subgradient (p. 3); Lemma 14.1 and the first part of its proof (p. 4); norm bounds, selection of the learning rate, and conclusion of the lemma (p. 5).
>
> The **2026/27 handwritten lecture** is the primary source here. This note is a reconstruction from the five supplied images, not a verbatim transcript. Sections labeled *Tutor clarification* include additional reasoning, prerequisite recovery, or examples that were **not established as statements by the lecturer**.
>
> **Study status:** The rationale for gradient descent, lateral derivatives and non-differentiability, the meaning of the support inequality, and the algebraic structure of Lemma 14.1 were discussed with guidance. The student independently recognized the purpose of the bounds involving \(B,\rho,T\). Independent derivation under exam conditions remains to be checked.

**Prerequisites:** [[LESSON 6]] (hard/soft SVM and margin); [[LESSON 5]] (linear classifiers and Perceptron).  
**Connected concepts:** [[Shattering and VC Dimension]] (different issue: statistical capacity versus numerical optimization).  
**Notation:** The handwritten pages use \(w\) for weights, \(\theta\) for generic gradient descent parameters, \(v_t\) for an update direction, \(\eta\) for the step size in the lemma, \(B\) for the comparator norm bound and \(\rho\) for the update-vector norm bound. The ordinary gradient-descent sketch uses \(\gamma\) for its learning rate.

---

## 1. Handwritten p. 1 — From hard SVM to soft SVM

### 1.1 Why the problem changes [2026/27 confirmed, p. 1]

The hard-margin problem, in the homogeneous notation on this page, is

$$
\min_{w\in\mathbb R^d}\ \frac12\|w\|^2
\quad\text{subject to}\quad
y_i\langle x_i,w\rangle\ge1,\quad i=1,\ldots,n.
$$

Here \(x_i\in\mathbb R^d\) is a training feature vector, \(y_i\in\{-1,+1\}\) is its label and \(y_i\langle x_i,w\rangle\) is its **signed score**. Hard SVM requires every sample to satisfy the margin constraint. If the dataset is not linearly separable, these requirements can be infeasible.

Soft SVM permits violations by introducing a nonnegative **slack variable** \(\xi_i\) for each example:

$$
\min_{w,\xi_1,\ldots,\xi_n}
\left[
\frac{\lambda}{2}\|w\|^2+\frac1n\sum_{i=1}^n\xi_i
\right]
$$

subject to

$$
y_i\langle x_i,w\rangle\ge1-\xi_i,\qquad \xi_i\ge0.
$$

We trade off weight size (related to margin) against violations of the margin. In this convention, **larger \(\lambda\)** gives greater relative weight to the norm penalty. Equivalent SVM conventions may place a parameter \(C\) in front of the slack penalty; do not treat \(C\) and \(\lambda\) as numerically identical.

### 1.2 Eliminate the slack variables [Tutor clarification of the p. 1 arrow]

Starting with the first constraint,

$$
y_i\langle x_i,w\rangle\ge 1-\xi_i,
$$

add \(\xi_i\) to both sides, then subtract the signed score:

$$
\xi_i\ge1-y_i\langle x_i,w\rangle.
$$

Together with \(\xi_i\ge0\), this means

$$
\xi_i\ge\max\{0,1-y_i\langle x_i,w\rangle\}.
$$

**For fixed \(w\)**, the objective increases whenever \(\xi_i\) increases because its coefficient is positive. Therefore the optimal feasible choice is the smallest one:

$$
\boxed{\xi_i^{\mathrm{opt}}(w)=\max\{0,1-y_i\langle x_i,w\rangle\}
=(1-y_i\langle x_i,w\rangle)_+.}
$$

This equality holds for the **optimal slack at fixed \(w\)**, not for every feasible slack assignment. Substitution gives the unconstrained-in-\(\xi\) objective

$$
\boxed{
F(w)=\frac{\lambda}{2}\|w\|^2+
\frac1n\sum_{i=1}^n(1-y_i\langle x_i,w\rangle)_+,
\qquad \min_{w\in\mathbb R^d}F(w).
}
$$

### 1.3 Hinge loss versus 0–1 loss [2026/27 confirmed, p. 1; examples tutor-added]

With signed score \(s=y_i\langle x_i,w\rangle\),

$$
\ell_{\mathrm{hinge}}(s)=(1-s)_+=\max\{0,1-s\}.
$$

For a nonzero signed score, the 0–1 classification loss is \(1\) when \(s<0\) and \(0\) when \(s>0\). The value at \(s=0\) depends on the convention used to turn a zero score into a label (in our prior notes, \(\mathrm{sign}(0)=+1\)). The inequality between 0–1 and hinge loss still holds under that convention.

| Signed score \(s\) | Classification / margin situation | Hinge loss |
| --- | --- | --- |
| \(s=2\) | Correct, margin respected | \(0\) |
| \(s=0.3\) | Correct but margin violated | \(0.7\) |
| \(s=0\) | Decision boundary / tie convention relevant | \(1\) |
| \(s=-2\) | Misclassified | \(3\) |

**Crucial distinction:** A correct classification need **not** have zero hinge loss. Hinge loss also penalizes correctly classified points with insufficient functional margin. It is a **convex surrogate** of 0–1 loss, and it upper-bounds 0–1 loss, but the two losses are not equal.

A minimal nonnegative slack has the following interpretations: \(\xi_i=0\) means the margin is respected; \(0<\xi_i<1\) means a correct classification within the margin; \(\xi_i=1\) means zero signed score; \(\xi_i>1\) means a classification error.

---

## 2. Handwritten p. 2 — Why gradient descent appears here

### 2.1 Connection with the preceding page [2026/27 confirmed, p. 2; tutor explanation]

We have **defined what to minimize**, \(F(w)\), but need a way to **compute** a good minimizing weight vector. Solving a large optimization problem in closed form may be impractical. Gradient descent is a general iterative method that uses local slope information to update parameters:

$$
\boxed{\theta^{(t+1)}=\theta^{(t)}-\gamma\nabla F(\theta^{(t)}),\qquad \gamma>0.}
$$

The lecturer's sketch starts from an initial \(\theta^{(0)}\); \(\theta\) denotes generic parameters and can be identified with \(w\) in this SVM discussion. For differentiable \(F:\mathbb R^d\to\mathbb R\),

$$
\nabla F(\theta)=
\begin{pmatrix}
\frac{\partial F}{\partial\theta_1}\\
\vdots\\
\frac{\partial F}{\partial\theta_d}
\end{pmatrix}.
$$

At a point with nonzero gradient, \(-\nabla F\) is the direction of **steepest local decrease** with respect to Euclidean distance, **not necessarily a line pointing directly at the minimum**. A sufficiently large step can increase the function even when it is convex.

**Two different scale issues:** \(d\) counts model parameters, while \(n\) counts training examples. A full gradient calculation can be expensive when the sum runs over many examples, motivating stochastic methods later; this is additional context, **not the lecturer's stated reason on p. 2**.

### 2.2 The two printed figures [2026/27 confirmed, p. 2]

**Left, contour map:** the axes are two *parameter coordinates*, not two data features. Every ellipse consists of parameter vectors with the same objective value. The zigzagging path moves through these parameter configurations, generally towards lower level curves and the central minimizer.

**Right, 3D surface:** the same kind of objective is shown as height \(F(\theta_1,\theta_2)\) above the parameter plane. The update trajectory descends across the surface towards the bottom. The path zigzags because the local steepest descent direction need not point along the valley towards the minimizer.

These printed diagrams illustrate optimization geometry. They are **not literal plots of the Soft SVM hinge objective**, which may have non-smooth parts.

### 2.3 Why introduce subgradients next? [2026/27 confirmed, p. 2]

The hinge function is non-differentiable at \(s=1\). Therefore the ordinary gradient-descent formula is not available at every parameter vector. The lecturer notes that a **subgradient** is needed.

> [!CAUTION] Tutor mathematical precision
> The lecturer's informal phrase about using a direction in which \(F\) decreases should not be taken as the **definition** of a subgradient. A subgradient obeys the supporting inequality given on p. 3; its negative is not guaranteed to decrease the function for every step size.

---

## 3. Handwritten p. 3 — Definition and geometry of the subgradient

### 3.1 The definition [2026/27 confirmed, p. 3]

Let \(f:S\to\mathbb R\) with \(S\subseteq\mathbb R^d\) convex. A vector \(v\in\mathbb R^d\) is a **subgradient of \(f\) at \(u\in S\)** if

$$
\boxed{
f(z)\ge f(u)+\langle z-u,v\rangle
\qquad\text{for every }z\in S.
}
$$

- \(u\): the point where we construct a supporting affine function;
- \(z\): **any** comparison point in \(S\);
- \(v\): candidate supporting slope/vector;
- \(\langle z-u,v\rangle\): inner product of displacement and supporting vector;
- \(\partial f(u)\): **subdifferential**, i.e. the set of all subgradients at \(u\).

The inequality means that the affine function

$$
L(z)=f(u)+\langle z-u,v\rangle
$$

passes through the function value at \(u\) and lies **below or on** \(f\) for every allowed \(z\). The quantifier \(\forall z\in S\) is essential: a locally plausible slope is not enough.

### 3.2 Printed figure on the left: a smooth point

The left figure uses \(f(u)=\tfrac12u^2\) at \(u=1\). Since

$$
f'(u)=u,\qquad f'(1)=1,
$$

the unique supporting slope is \(v=1\), and the supporting line is

$$
L(z)=f(1)+1(z-1)=\frac12+z-1=z-\frac12.
$$

Check explicitly:

$$
f(z)-L(z)
=\frac12z^2-z+\frac12
=\frac12(z-1)^2\ge0.
$$

Thus \(1\in\partial f(1)\); because this convex function is differentiable there,

$$
\boxed{\partial f(1)=\{1\}.}
$$

### 3.3 Printed figure on the right: a corner

For \(f(u)=|u|\) at \(u=0\), the subgradient definition becomes

$$
|z|\ge gz\qquad\forall z\in\mathbb R.
$$

If \(z>0\), then \(|z|=z\), so \(z\ge gz\), hence **\(g\le1\)**.

If \(z<0\), then \(|z|=-z\), so \(-z\ge gz\). Dividing by the **negative** \(z\) reverses the inequality: **\(g\ge-1\)**.

If \(z=0\), both sides are zero and impose no new restriction. Therefore

$$
\boxed{\partial|0|=[-1,1].}
$$

The printed slopes \(g=-0.7,0,0.7\) all generate supporting lines, which explains the multiple dashed lines in the right figure.

### 3.4 Targeted prerequisite recovery: derivative and lateral limits [Tutor clarification]

The **derivative** of a one-variable function at \(a\) is defined by the limit of the difference quotient:

$$
f'(a)=\lim_{h\to0}\frac{f(a+h)-f(a)}{h}.
$$

Here \(h\) is a **signed displacement**: \(h<0\) approaches from the left and \(h>0\) from the right. Importantly, \(f(a+h)\) means **replace the input \(x\) by \(a+h\)**, not \(f(a)+h\).

For instance, when \(f(x)=x^2\) and \(a=2\),

$$
\begin{aligned}
f(2+h)&=(2+h)^2=4+4h+h^2,\\
\frac{f(2+h)-f(2)}h
&=\frac{4+4h+h^2-4}{h}
=4+h\quad(h\ne0),\\
f'(2)&=\lim_{h\to0}(4+h)=4.
\end{aligned}
$$

For the hinge function \(h(s)=\max(0,1-s)\), at \(s=1\), note first that \(h(1)=0\). If a displacement \(\delta<0\), then \(1+\delta<1\), and the first branch applies:

$$
h(1+\delta)=1-(1+\delta)=-\delta.
$$

Consequently the left difference quotient is

$$
\frac{h(1+\delta)-h(1)}{\delta}
=\frac{-\delta}{\delta}=-1.
$$

If \(\delta>0\), then \(1+\delta>1\) and the second branch gives \(h(1+\delta)=0\); the right quotient is \(0\). The lateral derivatives **\(-1\) and \(0\)** differ, even though the function itself is continuous at \(1\). Thus the ordinary derivative does not exist there.

The subdifferential instead is

$$
\boxed{\partial h(1)=[-1,0].}
$$

**What to remember:** Continuity compares function values; differentiability compares limits of the difference quotient. A convex kink can be continuous yet non-differentiable.

---

## 4. Handwritten p. 4 — Lemma 14.1: the generic update bound

### 4.1 Statement [2026/27 confirmed, printed Lemma 14.1]

Let \(v_1,\ldots,v_T\in\mathbb R^d\) be **arbitrary vectors**. Initialize \(w^{(1)}=0\) and update by

$$
\boxed{w^{(t+1)}=w^{(t)}-\eta v_t,\qquad \eta>0.}
$$

For **every comparator** \(w^*\in\mathbb R^d\),

$$
\boxed{
\sum_{t=1}^T\langle w^{(t)}-w^*,v_t\rangle
\le
\frac{\|w^*\|^2}{2\eta}
+\frac{\eta}{2}\sum_{t=1}^T\|v_t\|^2.
}
\tag{14.5}
$$

**Do not introduce extra assumptions prematurely.** At this stage \(v_t\) need **not** be a subgradient; \(w^*\) is a reference vector, not necessarily an optimum.

### 4.2 The proof strategy

The handwritten proof starts from the norm identity

$$
\|a-b\|^2=\|a\|^2+\|b\|^2-2\langle a,b\rangle,
$$

which, after rearrangement, gives

$$
\langle a,b\rangle
=\frac12\big(\|a\|^2+\|b\|^2-\|a-b\|^2\big).
$$

**Key substitutions:** \(a=w^{(t)}-w^*\), \(b=\eta v_t\). The update rule identifies

$$
a-b=w^{(t)}-w^*-\eta v_t=w^{(t+1)}-w^*.
$$

Expanding the squared distance after the step gives

$$
\begin{aligned}
\|w^{(t+1)}-w^*\|^2
&=\|w^{(t)}-w^*-\eta v_t\|^2\\
&=\|w^{(t)}-w^*\|^2
+\eta^2\|v_t\|^2
-2\eta\langle w^{(t)}-w^*,v_t\rangle.
\end{aligned}
$$

Isolating the inner product,

$$
\boxed{
\langle w^{(t)}-w^*,v_t\rangle
=\frac{\|w^{(t)}-w^*\|^2-\|w^{(t+1)}-w^*\|^2}{2\eta}
+\frac{\eta}{2}\|v_t\|^2.
}
$$

Summing from \(t=1\) to \(T\), the squared-distance differences **telescope**:

$$
\begin{aligned}
&\sum_{t=1}^T
\big(\|w^{(t)}-w^*\|^2-\|w^{(t+1)}-w^*\|^2\big)\\
&\hspace{12mm}=
\|w^{(1)}-w^*\|^2-\|w^{(T+1)}-w^*\|^2.
\end{aligned}
$$

Since \(w^{(1)}=0\),

$$
\|w^{(1)}-w^*\|^2=\|w^*\|^2.
$$

Since a squared norm is nonnegative, discarding its **negative** contribution produces an upper bound:

$$
\boxed{
\sum_{t=1}^T\langle w^{(t)}-w^*,v_t\rangle
\le
\frac{\|w^*\|^2}{2\eta}
+\frac{\eta}{2}\sum_{t=1}^T\|v_t\|^2.
}
$$

This completes the **first part** of the lemma. The choice of learning rate is developed on **p. 5**, not here.

---

## 5. Handwritten p. 5 — Bounding the norms and choosing the learning rate

The five-step mapping below follows the lecturer's colored handwritten annotations in **top-to-bottom order**.

### 5.1 Top red assumptions and first inequality

The page opens with

$$
\boxed{\|w^*\|\le B,\qquad \|v_t\|\le\rho\quad\forall t.}
$$

Assume \(B,\rho>0\). From p. 4,

$$
\begin{aligned}
\sum_{t=1}^T\langle w^{(t)}-w^*,v_t\rangle
&\le\frac{\|w^*\|^2}{2\eta}
+\frac{\eta}{2}\sum_{t=1}^T\|v_t\|^2\\
&\le \boxed{\frac{B^2}{2\eta}+\frac{\eta T\rho^2}{2}},
\end{aligned}
$$

because \(\|w^*\|^2\le B^2\), and because

$$
\sum_{t=1}^T\|v_t\|^2\le\sum_{t=1}^T\rho^2=T\rho^2.
$$

**Study checkpoint:** This bounding step was understood and explicitly marked as clear before continuing to the rest of this page.

### 5.2 Blue note: "We can set η to minimize the sum"

The right-hand side depends on the adjustable learning rate. Define

$$
G(\eta)=\frac{B^2}{2\eta}+\frac{\eta T\rho^2}{2},\qquad \eta>0.
$$

We are **not** minimizing the SVM objective \(F(w)\) here. We are minimizing a **proven upper bound**, to make the resulting guarantee as strong as this calculation permits.

### 5.3 Central graph and red note: "The sum is minimal here, when the two terms are ="

The **orange decreasing curve** is \(B^2/(2\eta)\). The **black increasing straight line** is \(\eta T\rho^2/2\). The lecturer circles their intersection.

For this particular function \(G(\eta)=a/\eta+b\eta\), with \(a,b>0\), the minimum occurs where its two positive terms balance. A derivation (tutor expansion of the graph) confirms:

$$
G'(\eta)=-\frac{B^2}{2\eta^2}+\frac{T\rho^2}{2}.
$$

Setting \(G'(\eta)=0\) yields

$$
\boxed{\frac{B^2}{2\eta}=\frac{\eta T\rho^2}{2}.}
$$

This *equal-terms rule* is **not true for arbitrary sums of functions**; it works for this reciprocal-plus-linear form. Also, the orange/black intersection is where the **sum** is minimized, not where either individual function is minimized.

### 5.4 Equation in blue beneath the graph: solve for η

$$
\begin{aligned}
\frac{B^2}{2\eta}&=\frac{\eta T\rho^2}{2}\\
B^2&=\eta^2T\rho^2\\
\eta^2&=\frac{B^2}{T\rho^2}\\
\boxed{\eta&=\frac{B}{\rho\sqrt T}}.
\end{aligned}
$$

We select the positive root because the learning rate is positive. This value depends on the chosen horizon \(T\).

### 5.5 Bottom blue note: "Plugging this η into the RHS gives", then red "Dividing everything by T concludes the proof"

The first upper-bound term becomes

$$
\frac{B^2}{2\eta}
=\frac{B^2}{2B/(\rho\sqrt T)}
=\frac{B\rho\sqrt T}{2},
$$

and the second becomes

$$
\frac{\eta T\rho^2}{2}
=\frac{B}{\rho\sqrt T}\frac{T\rho^2}{2}
=\frac{B\rho\sqrt T}{2}.
$$

Thus the handwritten **starred inequality** is

$$
\boxed{
\sum_{t=1}^T\langle w^{(t)}-w^*,v_t\rangle
\le B\rho\sqrt T.
}
$$

Finally divide by \(T>0\):

$$
\boxed{
\frac1T\sum_{t=1}^T\langle w^{(t)}-w^*,v_t\rangle
\le\frac{B\rho}{\sqrt T}.
}
$$

This concludes the **second part** of Lemma 14.1.

### 5.6 What this does — and does not — prove [Tutor clarification]

The derived **upper bound** on the averaged inner products tends to zero at rate \(O(T^{-1/2})\). The inner-product average is not necessarily nonnegative, so this upper bound alone does **not** prove that the average itself converges to zero, nor that each iterate converges to \(w^*\).

For an application to convex optimization, we must now **add** the assumption that \(v_t\in\partial F(w^{(t)})\). From the subgradient inequality on p. 3, setting \(u=w^{(t)}\), \(z=w^*\),

$$
F(w^*)\ge F(w^{(t)})+\langle w^*-w^{(t)},v_t\rangle,
$$

so

$$
\boxed{
F(w^{(t)})-F(w^*)
\le\langle w^{(t)}-w^*,v_t\rangle.
}
$$

**This final connection is a tutor preview of a subsequent application, not part of the completed handwritten p. 5 proof.** Under appropriate assumptions it leads to a convergence guarantee on averaged objective suboptimality or an averaged iterate. Do not attribute that extension to the five pages already studied.

---

## 6. Compact reconstruction for examination

The conceptual chain in the current lecture is:

1. **Soft SVM:** permit margin violations, then eliminate optimal slack variables to obtain a regularized hinge-loss objective.
2. **Gradient descent:** an iterative mechanism to optimize a differentiable objective; \(-\nabla F\) is the local steepest-descent direction.
3. **Subgradient:** generalized supporting slope defined by \(f(z)\ge f(u)+\langle z-u,v\rangle\) for all \(z\).
4. **Lemma 14.1:** use a squared-distance expansion and a telescoping sum to bound the cumulative comparator inner products.
5. **Optimize the bound:** impose \(\|w^*\|\le B\), \(\|v_t\|\le\rho\); balance the two terms in \(\eta\) to obtain the \(B\rho/\sqrt T\) average bound.
6. **Later application, not yet shown:** connect the bound to loss differences through the subgradient inequality.

**Reconstruct rather than memorize:** why the squared distance is expanded, which terms cancel in the telescoping sum, where nonnegativity is used, and why the optimal learning rate contains \(\sqrt T\).

## 7. Common traps and personal learning checkpoints

- **Confusing \(n\), \(d\), and \(T\):** \(n\) = training examples; \(d\) = parameter dimension; \(T\) = number of optimization steps.
- **Confusing the errors:** 0–1 loss counts misclassifications, while hinge loss measures a margin violation and can be positive even for a correct prediction.
- **Treating optimal slack as an arbitrary feasible slack:** equality with hinge loss comes from minimizing a positively weighted slack, at fixed \(w\).
- **Confusing \(f(a+h)\) with \(f(a)+h\):** always substitute the **full argument** into the formula defining \(f\).
- **Confusing continuity with differentiability:** equality of function limits does not imply equality of lateral difference quotients.
- **Treating a subgradient as any descent direction:** it is defined by a global supporting inequality.
- **Thinking the negative gradient points geometrically straight to the minimum:** only local steepest decrease is guaranteed.
- **Attributing the entire lemma to p. 4:** p. 4 develops the generic inequality; p. 5 selects \(\eta\) and concludes the bound.
- **Assuming the lemma alone proves SVM convergence:** further use of the subgradient inequality and additional conditions is needed.

### Active recall / exercises (do not consult solutions yet)

1. For signed score \(s=0.25\), compute the smallest admissible slack and hinge loss. Is the prediction correct?
2. For \(f(x)=|x|\), verify directly from the definition that \(g=0.5\) is a subgradient at \(0\), and that \(g=2\) is not.
3. Starting from \(f(x)=\max(0,1-x)\), derive the two lateral derivatives at \(x=1\), explaining why \(f(1+\delta)=-\delta\) for \(\delta<0\).
4. Reconstruct the squared-distance identity for the update \(w^{(t+1)}=w^{(t)}-\eta v_t\), then show which two adjacent terms cancel when summing over \(t\).
5. Show using differentiation, without relying on the graph, that \(G(\eta)=B^2/(2\eta)+\eta T\rho^2/2\) is minimized at \(\eta=B/(\rho\sqrt T)\).
6. **Exam-level distinction:** Explain why an upper bound \(A_T\le B\rho/\sqrt T\) does not by itself imply \(A_T\to0\), but can be useful for proving bounds on nonnegative objective gaps.

**Next session:** Check independent recall of the lemma proof and the interpretation of the optimal learning rate; then continue with the next handwritten page before expanding to formal convergence theorems.
