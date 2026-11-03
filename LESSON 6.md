---
title: "Lecture 6 — Affine Classifiers, Margins, and Support Vector Machines"
aliases:
  - "Lecture 6 - 13-10-2026"
date: 2026-10-13
course: "Machine Learning — Sapienza"
academic_year: "2026/2027"
type: lecture-note
status: "Partial bridge from Lecture 5; official Lecture 6 mapped but SVM material not yet verified as studied"
tags: [machine-learning, affine-classifiers, margin, svm, hinge-loss, optimization]
---

# Lecture 6 — Affine Classifiers, Margins, and SVM

> [!IMPORTANT] Honest learning-status boundary
> The available learning log confirms that **affine classifiers and feature augmentation were introduced at the end of Lecture 5**. The official handwritten **Lecture 6** has now been located. However, the learning log available at this update does **not** attest a full guided study session on its margin and SVM derivations. The material below is a **source-grounded reference / orientation**, **not** a claim that we have already completed or mastered the lesson. Start an independent check before marking any new section Operational.

**Sources (2026/27):**
- [Handwritten Lecture 6 — 13 October 2026](https://drive.google.com/file/d/1t0ElkUB94qu45AypUvmvv0gBrHFgpvtH/view): primary sequence — affine classifiers, feature map, VC dimension, separability, geometric distance and margin, hard/soft SVM, hinge loss.
- [Week 2 — Linear Classifiers I, version 2](https://drive.google.com/file/d/1FhP2Vy1u108UjZF8PI0ynV21cHQJrzeZ/view), §1 Remark 1.3 pp. 2–3 and §2 pp. 3–5: formal bridge for affine representation and separability. **The SVM derivations below are taken from the handwritten Lecture 6, not attributed to Week 2.**
- Connected note: [[LESSON 5]].

## A. Bridge already discussed: affine vs homogeneous linear classifiers

For $x,w\in\mathbb R^d$ and scalar bias $b\in\mathbb R$:

$$
h_w(x)=\operatorname{sign}(\langle w,x\rangle),\qquad
h_{w,b}(x)=\operatorname{sign}(\langle w,x\rangle+b).
$$

A homogeneous linear decision boundary goes through the origin. With bias, the boundary $\langle w,x\rangle+b=0$ can be translated.

Define the **feature map** and extended weight:

$$
\phi(x)=(x,1)\in\mathbb R^{d+1},\qquad \widetilde w=(w,b)\in\mathbb R^{d+1}.
$$

Then

$$
\boxed{h_{w,b}(x)=\operatorname{sign}(\langle\widetilde w,\phi(x)\rangle).}
$$

This algebraic identity was introduced in our Lecture 5 study; the handwritten Lecture 6 takes it up again. The VC dimensions are $d$ for homogeneous and $d+1$ for affine classifiers in $\mathbb R^d$. **Independent proof/check for the affine claim remains pending.**

## B. The new geometric question: signed distance

Fix a nonzero normal vector $w$ and a homogeneous separating hyperplane

$$H_w=\{z\in\mathbb R^d:\langle w,z\rangle=0\}.$$

Its **signed perpendicular distance** from $x$ is

$$\boxed{\frac{\langle w,x\rangle}{\|w\|_2}.}$$

Reason: the unit normal is $w/\|w\|_2$, and the scalar projection of $x$ onto that unit normal is the displayed expression. The unsigned distance is its absolute value. If $\|w\|_2=1$, the signed distance is simply $\langle w,x\rangle$.

For labeled data $(x_i,y_i)$, $y_i\in\{-1,+1\}$, the **signed geometric margin** of example $i$ is

$$\frac{y_i\langle w,x_i\rangle}{\|w\|_2}.$$

Multiplying by $y_i$ makes the margin positive when the point lies on the side expected for its label. For an affine boundary, replace the numerator with $y_i(\langle w,x_i\rangle+b)$; the denominator is $\|w\|_2$, not $\|(w,b)\|_2$, when measuring distance in the original feature space.

**Status:** this geometric interpretation is in Lecture 6; mark as **Introduced by source, independent study check pending**.

## C. Margin of a classifier over a sample

For nonzero $w$, define

$$
\gamma(w)=\min_i\frac{y_i\langle w,x_i\rangle}{\|w\|_2}.
$$

It is the **smallest** signed geometric margin among all examples: the most constraining point controls it. The largest achievable margin is

$$\gamma^\star=\max_{w\ne0}\gamma(w)$$

in the separable homogeneous setting. For affine classifiers, optimize over $(w,b)$ with the corresponding affine numerator.

**Connection with Lecture 5:** If a feasible separating $w$ has been rescaled so that $\min_i y_i\langle w,x_i\rangle=1$, then

$$\gamma(w)=\frac{1}{\|w\|_2}.$$

This connects the norm-minimizing separator in the Perceptron analysis to **maximum margin**; it does not make Perceptron a maximum-margin algorithm.

## D. Hard-margin SVM — separable setting

The handwritten Lecture 6 writes a normalized constrained optimization:

$$
\boxed{
\begin{aligned}
\min_{w\in\mathbb R^d}\quad & \frac12\|w\|_2^2\\
\text{subject to}\quad & y_i\langle w,x_i\rangle\ge1,\qquad i=1,\ldots,n.
\end{aligned}}
$$

This is a **homogeneous** version. The affine analogue optimizes $(w,b)$ and uses $y_i(\langle w,x_i\rangle+b)\ge1$, while penalizing $\|w\|_2^2$ only. Under strict separability, the normalization makes this equivalent to maximizing the geometric margin. Squaring the nonnegative norm does not change its minimizer; $1/2$ is a convenient scaling.

**Logic to verify independently:** constraints ensure a separating classifier, normalization fixes the otherwise arbitrary positive scale, and minimizing norm maximizes $1/\|w\|_2$. Examples on the tight constraints are *support vectors* in the usual nondegenerate interpretation.

## E. Agnostic (nonseparable) case and soft-margin SVM

The handwritten notes distinguish the difficulty of empirical 0–1-loss optimization in the agnostic case (NP-hardness is cited) from tractable convex-surrogate optimization. Do not confuse **computing an exact best 0–1 classifier** with **fitting a convex SVM surrogate**.

For slack variables $\xi_i\ge0$ and a regularization/trade-off coefficient $C>0$, the standard soft-margin objective extending the handwritten notes is

$$
\boxed{
\begin{aligned}
\min_{w,\xi}\quad&\frac12\|w\|_2^2+C\sum_{i=1}^n\xi_i\\
\text{subject to}\quad&y_i\langle w,x_i\rangle\ge 1-\xi_i,\\
&\xi_i\ge0.
\end{aligned}}
$$

For fixed $w$, the smallest feasible slack is

$$\boxed{\xi_i=\max\{0,1-y_i\langle w,x_i\rangle\}.}$$

Substituting yields the unconstrained **hinge-loss** expression

$$
\min_w \frac12\|w\|_2^2+C\sum_{i=1}^n
\max\{0,1-y_i\langle w,x_i\rangle\}.
$$

Here $\ell_{\mathrm{hinge}}(y,s)=\max(0,1-ys)$ upper-bounds the 0–1 misclassification indicator under the usual strict-error definition. The hinge objective is convex in $w$, unlike the empirical 0–1 count. The trade-off $C$ balances margin regularization against violations. For affine SVM, use $\langle w,x_i\rangle+b$ in the loss and constraints.

**Attribution:** The handwritten Lecture 6 provides the optimization/slack/hinge narrative. The explicit $C$ parameter and the affine variants above are **clarifying standard notation**, not assertions that the handwritten page uses this exact typography or parameter. The notes appear to use a normalization of the slack sum; check against the actual page before reproducing the professor's exact coefficient.

## F. Boundaries of progress and the next study checkpoint

**Already covered with guidance in the preceding session:** the affine feature map and the shift from $\operatorname{sign}(\langle w,x\rangle)$ to $\operatorname{sign}(\langle w,x\rangle+b)$.

**Source-mapped, but not verified as independently studied:** signed distance, geometric margin, why hard-margin norm minimization solves max-margin, constraints/slack, hinge-loss equivalence, support vectors, and computational comparison with 0–1 ERM. All these are **2026/27 confirmed by handwritten Lecture 6**, but **not marked Understood** solely because they appear here.

**When resuming:** read the handwritten Lecture 6 in order, recover the geometric intuition of perpendicular distance, then derive the hard-margin equivalence on paper; only after that tackle soft-margin slack elimination. Use small arithmetic checkpoints before new exercises. Do not invent a completed exercise record.

### Active recall, not yet attempted

1. Derive the feature-augmentation identity and explain the extra dimension's role.
2. Starting from the definition of scalar projection, justify the distance formula, including the $\|w\|_2$ denominator.
3. Show why positive rescaling changes the functional score but not the geometric margin.
4. Explain why hard-margin normalization makes minimizing $\|w\|_2$ equivalent to maximizing margin.
5. Eliminate each $\xi_i$ from the soft-margin constraints and recover hinge loss.
6. Contrast the objectives of Perceptron, empirical 0–1 ERM and an SVM — they are **not** interchangeable.

**Further current lectures:** [Lecture 7 — 16 October 2026](https://drive.google.com/file/d/1Z8AWAXy2pqgz8oFVPf9a8yLXjehYd2kI/view) exists in the 2026/27 folder but is **not covered by this note**.
