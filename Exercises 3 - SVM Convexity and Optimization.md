---
title: "Exercises 3 — SVM Convexity and Optimization"
date: 2026-11-09
course: "Machine Learning — Sapienza"
academic_year: "2026/2027"
type: exercise-notes
status: "Exercise 1.1(a) guided solution complete; independent recall pending"
tags: [machine-learning, exercises, svm, convexity, strong-convexity, optimization]
---

# Exercises 3 — SVM Convexity and Optimization

> **Source:** Federico Fusco, *Week 3: Linear Classifiers II — Support Vector Machines*, Exercise Sheet (18 October 2026, version 3), Exercise 1.1, pp. 1–2.  
> **Status as of 10 November 2026:** Exercise 1.1(a) has been studied in full: convexity of $C$, $1$-strong convexity of $H$, uniqueness, and existence via continuity + coercivity on closed nonempty $C$. **Independent recall remains pending.** Exercises 1.1(b)–(e) and the remaining sheet are pending.
> **Solutions:** independently worked learning notes, not official instructor solutions.  
> **Related:** [[LESSON 6]] · [[Exercises 0 - Fully Worked Solutions]]

## Exercise 1.1(a) — Hard SVM

The problem is

$$
\min_{w\in C}H(w), \qquad
H(w)=\frac12\|w\|^2, \qquad
C=\{w\in\mathbb R^d:y_i\langle w,x_i\rangle\geq1\ \text{for every }i\}.
$$

We show that $C$ is convex, $H$ is $1$-strongly convex, and that a minimizer **exists and is unique** when $C\neq\varnothing$. All proofs below have been studied with guidance; independent reproduction is still pending.

### 1. Prove that $C$ is convex — COMPLETED

**Definition (course sheet, p. 1).** A set $K$ is convex if

$$
u,v\in K,\quad t\in[0,1]
\quad\Longrightarrow\quad (1-t)u+tv\in K.
$$

This is **convexity of a set**, not convexity of a function. In particular, we do **not** need the separate condition $f((1-t)u+tv)\leq(1-t)f(u)+tf(v)$, which concerns **functions**.

**Proof.** Take any $u,v\in C$ and $t\in[0,1]$. By the definition of $C$, for every index $i$,

$$
y_i\langle u,x_i\rangle\geq1,
\qquad
y_i\langle v,x_i\rangle\geq1.
$$

To prove that $(1-t)u+tv\in C$, we must verify that **this new vector satisfies the same constraint**:

$$
y_i\langle(1-t)u+tv,x_i\rangle\geq1
\quad\text{for every }i.
$$

By linearity of the inner product,

$$
\begin{aligned}
y_i\langle(1-t)u+tv,x_i\rangle
&=(1-t)y_i\langle u,x_i\rangle+t\,y_i\langle v,x_i\rangle\\
&\geq(1-t)\cdot1+t\cdot1\\
&=1.
\end{aligned}
$$

The inequality is justified because $t\in[0,1]$, hence **both weights** $t$ and $1-t$ are nonnegative. Multiplication by a nonnegative weight preserves the direction of an inequality. The two weights **sum to exactly one**, so a weighted average of two numbers each at least $1$ is still at least $1$.

This holds for every $i$ and every $t\in[0,1]$. Thus $(1-t)u+tv\in C$ and **$C$ is convex**. $\square$

**Common pitfall from the first attempt:** checking only $t=0$ and $t=1$ checks the endpoints, not **all** $t\in[0,1]$. Another algebraic caution: the inner-product expansion above is an **equality**; the $\geq$ is introduced only when applying the two constraints.

### 2. Prove that $H$ is $1$-strongly convex — COMPLETED

**Definition (course sheet, p. 1).** A convex function $f:\mathbb R^d\to\mathbb R$ is $\mu$-strongly convex, for $\mu>0$, if

$$
f(u)\geq f(w)+\langle g,u-w\rangle+\frac{\mu}{2}\|u-w\|^2
$$

for every $u,w\in\mathbb R^d$ and every $g\in\partial f(w)$.

For $H(w)=\frac12\|w\|^2$, its gradient is $\nabla H(w)=w$. The squared Euclidean norm is convex, so the subdifferential at $w$ is $\partial H(w)=\{w\}$. For $\mu=1$, we need to verify

$$
\frac12\|u\|^2\geq
\frac12\|w\|^2+\langle w,u-w\rangle+\frac12\|u-w\|^2.
$$

**Proof by direct substitution and algebraic simplification (the method developed during the study session).**

Start from the right-hand side and expand the inner product:

$$
\begin{aligned}
&\frac12\|w\|^2+\langle w,u-w\rangle+\frac12\|u-w\|^2\\
&=\frac12\|w\|^2+\langle w,u\rangle-\langle w,w\rangle+\frac12\|u-w\|^2\\
&=-\frac12\|w\|^2+\langle w,u\rangle+\frac12\|u-w\|^2.
\end{aligned}
$$

Now use the identity

$$
\|u-w\|^2=\|u\|^2+\|w\|^2-2\langle u,w\rangle.
$$

Substituting,

$$
\begin{aligned}
&-\frac12\|w\|^2+\langle w,u\rangle
+\frac12\bigl(\|u\|^2+\|w\|^2-2\langle u,w\rangle\bigr)\\
&=-\frac12\|w\|^2+\langle w,u\rangle
+\frac12\|u\|^2+\frac12\|w\|^2-\langle u,w\rangle\\
&=\frac12\|u\|^2,
\end{aligned}
$$

because the squared-norm terms cancel and $\langle w,u\rangle=\langle u,w\rangle$.

Therefore the required inequality reduces to

$$
\frac12\|u\|^2\geq\frac12\|u\|^2,
$$

which is true **with equality for every $u,w$**. Thus **$H$ is $1$-strongly convex**. $\square$

**Why this proves more than ordinary convexity:** the definition includes an additional nonnegative quadratic term $\frac12\|u-w\|^2$. Here the identity holds exactly with that term.

**Common pitfall from the handwritten calculation:** after expanding $\langle w,u-w\rangle$, do not discard $\langle w,u\rangle$. Also $\frac12\|w\|^2-\|w\|^2=-\frac12\|w\|^2$, **not zero**. Both cancellations become valid only after expanding $\|u-w\|^2$.

**Alternative equivalent proof (not necessary to memorize):** write $u=w+(u-w)$ and expand $\frac12\|w+(u-w)\|^2$, obtaining precisely the right-hand side of the strong-convexity inequality.

### 3. Uniqueness of the minimizer — GUIDED PROOF COMPLETED; INDEPENDENT RECALL PENDING

**Objective.** Assuming a minimizer of $H$ over $C$ exists, show that it is unique. This argument is distinct from the proof of existence.

**Logical steps (first remember the strategy, then reproduce the algebra):**

1. We already know that $C$ is convex and $H$ is $1$-strongly convex. Assume a minimizer exists.
2. **Proof by contradiction:** suppose there are two distinct minimizers $u,v\in C$ with $u\neq v$.
3. Both achieve the same minimum value $m$, so $H(u)=H(v)=m$. By the definition of a minimum, $H(w)\geq m$ for every $w\in C$.
4. Since $C$ is convex, choose the **midpoint** $z=(u+v)/2\in C$. This is **not mandatory**: any interior convex combination would work; the midpoint is chosen to simplify the algebra.
5. Apply strong convexity at reference point $z$, first to $u$ and then to $v$. The resulting inequalities show that $H(z)\leq (H(u)+H(v))/2-\frac18\|u-v\|^2$.
6. As $u\neq v$, the squared norm is positive, hence $H(z)<m$.
7. But $z\in C$ and $m$ is the minimum over $C$, so $H(z)\geq m$. **Contradiction.**
8. Thus two distinct minimizers cannot exist; **the minimizer, if it exists, is unique**.

**Why construct a third point?** Not because a proof by contradiction requires it. We need an admissible point whose value is **strictly below** the assumed minimum. Convexity of $C$ supplies an admissible point between $u$ and $v$, and strong convexity of $H$ supplies the strict improvement. Choosing $t=0$ or $t=1$ would give an endpoint, not this contradiction.

**Detailed derivation from the professor's definition.** Since $\nabla H(z)=z$ and the strong-convexity constant is $1$,

$$
H(u)\geq H(z)+\langle z,u-z\rangle+\frac12\|u-z\|^2
$$

and

$$
H(v)\geq H(z)+\langle z,v-z\rangle+\frac12\|v-z\|^2.
$$

Adding the two inequalities gives

$$
\begin{aligned}
H(u)+H(v)\geq {}&
2H(z)+\langle z,u-z\rangle+\langle z,v-z\rangle\\
&+\frac12\|u-z\|^2+\frac12\|v-z\|^2.
\end{aligned}
$$

From the **definition** $z=(u+v)/2$,

$$
u-z=u-\frac{u+v}{2}=\frac{2u-u-v}{2}=\frac{u-v}{2},
$$

and similarly

$$
v-z=v-\frac{u+v}{2}=\frac{2v-u-v}{2}=\frac{v-u}{2}.
$$

Since $v-u=-(u-v)$, substitution into these **two established identities** yields $v-z=-(u-z)$. Therefore $(u-z)+(v-z)=0$ and

$$
\langle z,u-z\rangle+\langle z,v-z\rangle
=\langle z,(u-z)+(v-z)\rangle=0.
$$

Moreover, by homogeneity of the Euclidean norm,

$$
\frac12\|u-z\|^2
=\frac12\left\|\frac{u-v}{2}\right\|^2
=\frac18\|u-v\|^2,
$$

and the same holds with $v-z$ in place of $u-z$. Consequently,

$$
H(u)+H(v)\geq2H(z)+\frac14\|u-v\|^2.
$$

Subtract the squared-norm term and divide by $2$:

$$
\boxed{H(z)\leq\frac{H(u)+H(v)}{2}-\frac18\|u-v\|^2.}
$$

Because $H(u)=H(v)=m$ and $u\neq v$,

$$
H(z)\leq m-\frac18\|u-v\|^2<m.
$$

But $z\in C$ implies $H(z)\geq m$. Contradiction. **Uniqueness proved.**

> [!IMPORTANT] Study status
> This proof was followed carefully in guided dialogue but **has not yet been independently reproduced**. Reconstruct the logical steps and derive the midpoint inequality without looking. Existence is proved separately below using **continuity and coercivity on a nonempty closed set**; the uniqueness argument alone would not establish existence.

### 4. Existence of the minimizer — COERCIVITY ARGUMENT STUDIED; INDEPENDENT RECALL PENDING

**Goal:** Assuming $C\neq\varnothing$, prove there is some $w^\star\in C$ such that

$$
H(w^\star)\leq H(w)\qquad\text{for every }w\in C.
$$

**Theorem (standard finite-dimensional existence result, used as a known theorem):** A continuous, coercive function on a nonempty closed subset of $\mathbb R^d$ attains its global minimum. Coercivity means that the function tends to $+\infty$ when the norm of its argument tends to infinity.

We check all assumptions rather than appealing to the projection theorem, which has not been established in our course notes.

**1. $C$ is nonempty.** This is the hypothesis $C\neq\varnothing$ in the exercise.

**2. $C$ is closed.** For each $i$, the map

$$
w\longmapsto y_i\langle w,x_i\rangle
$$

is continuous. Therefore the set

$$
C_i=\{w\in\mathbb R^d:y_i\langle w,x_i\rangle\geq1\}
$$

is closed (the inverse image of the closed interval $[1,+\infty)$ under a continuous map). Since the sample is finite,

$$
C=\bigcap_{i=1}^{n}C_i
$$

is closed.

**3. $H$ is continuous.** Since

$$
H(w)=\frac12\sum_{j=1}^{d}w_j^2,
$$

it is a polynomial in the coordinates of $w$ and hence continuous.

**4. $H$ is coercive.** By definition,

$$
\|w\|\longrightarrow+\infty
\quad\Longrightarrow\quad
H(w)=\frac12\|w\|^2\longrightarrow+\infty.
$$

Thus all the theorem's hypotheses hold: $C$ is nonempty and closed, and $H$ is continuous and coercive. Consequently,

$$
\boxed{\exists\,w^\star\in C:
H(w^\star)=\min_{w\in C}H(w).}
$$

**Existence proved.** Combined with the uniqueness proof above, this establishes that the hard-SVM problem has exactly one minimizer when $C\neq\varnothing$.

> [!NOTE] Why the coercivity theorem works
> The function becomes arbitrarily large far from the origin, so a minimizing sequence cannot escape to infinity. Because $C$ is closed, a convergent subsequence of a bounded minimizing sequence has its limit in $C$; continuity gives the minimum value at that limit. This is the underlying finite-dimensional argument (Bolzano–Weierstrass). No extra compact set must be introduced in the exam solution if the coercivity theorem is allowed.
>
> **Course-specific caution:** We recall the instructor discussing coercivity in a previous year, making this an appropriate candidate for the exam proof, but the theorem is not explicitly listed in the first-page conventions of this exercise sheet.

## Exercise 1.1(b)–(e) — NOT YET STUDIED

Do not mark these parts as complete. Continue sequentially after finishing all of 1.1(a).

## Study checkpoint

- [x] Distinguish convexity of a **set** from convexity of a **function**.
- [x] Prove $C$ convex using a convex combination and all margin constraints.
- [x] Understand why $(1-t)A+tB\geq1$ when $A,B\geq1$ and $t\in[0,1]$.
- [x] Verify $1$-strong convexity of $H$ by direct algebraic substitution.
- [ ] Independently reproduce both proofs without looking at the note.
- [ ] Independently reproduce the **uniqueness** proof without notes.
- [x] Study the **existence** argument via continuity, closedness, and coercivity.
- [ ] Independently reproduce the existence argument.
- [ ] Continue with Exercise 1.1(b), then later parts.
