---
title: "Lecture 8 — Gradient Descent, Convexity, Subgradients, SGD, and XOR"
aliases:
  - "Lecture 8"
  - "LESSON 8"
date: 2026-10-20
course: "Machine Learning — Sapienza"
academic_year: "2026/2027"
type: lecture-note
status: "All sections discussed with guidance; independent proofs and timed exercises pending"
tags: [machine-learning, gradient-descent, convexity, lipschitz, jensen, subgradient, hinge-loss, soft-svm, stochastic-gradient-descent, perceptron, xor]
---

# Lecture 8 — Gradient Descent, Convexity, Subgradients, SGD, and XOR

> [!IMPORTANT] Sources, scope, and learning status
> **2026/27 confirmed — handwritten lecture:** [Lezione 8 — 20/10/2026](https://drive.google.com/file/d/1GIn-BhuxXw9gTyNqpBHUnoiHAexOqdV1/view). This note follows the lecturer's progression from gradient descent to Lipschitz functions, convexity, an averaged-iterate guarantee, subgradients, the Soft-SVM objective, stochastic estimation, and the XOR/Perceptron example.
>
> **Secondary source:** [Week 3 — Linear Classifiers II](https://drive.google.com/file/d/1RKYA3bAy2eVzMuIIrhluBwaOMly_53Jh/view), especially §1 for the XOR example and §3 for hinge loss / Soft-SVM. The Week 3 handout is not a substitute for the handwritten Lecture 8 ordering.
>
> **Tutor expansions:** numerical examples, detailed rearrangements, one-dimensional pictures interpreted in words, subtle differentiability exceptions, and the four-point disk VC-dimension exam comparison below. These are not quotations attributed to the lecturer.
>
> **Learning evidence:** the learner followed the first-order convexity equivalence proof and correctly reconstructed the sum of inequalities; independently identified Jensen as the relevant inequality, and later understood the averaging factor $1/T$ after a specific clarification. They worked through hinge-loss kinks, the subgradient interval, the role of the indicator in SGD, and the expectation of a uniformly sampled contribution with guidance. **Not yet demonstrated:** a fully unaided convexity proof, telescoping proof, choice of learning rate, expectation derivation, or XOR analysis under exam conditions. Reading this note does not imply mastery.

**Prerequisites:** [[LESSON 5]] (Perceptron), [[LESSON 6]] (hard/soft SVM), [[LESSON 7]] (hinge loss and subgradient-related material).  
**Related concept:** [[Shattering and VC Dimension]].  
**Notation:** $w\in\mathbb R^d$ is the parameter vector, $u,z$ comparison points, $m$ the training sample size, $T$ the number of iterations, $\eta>0$ a step size, and $\lambda>0$ the Soft-SVM regularization parameter. The lecturer sometimes uses a different symbol for the Lipschitz constant (here $G$) and may index from $0$ instead of $1$; match the intended initialization explicitly.

---

## 1. Gradient descent: update and initialization [2026/27 confirmed — lecture opening]

We seek a minimizer $w^\star\in\arg\min_w F(w)$ of an objective $F:\mathbb R^d\to\mathbb R$. In differentiable optimization, the basic update is

$$
\boxed{w^{(t+1)}=w^{(t)}-\eta\nabla F(w^{(t)})}.
$$

The gradient points in the direction of steepest local increase under the Euclidean metric; the negative gradient is the opposite direction. **Starting at zero is an initialization choice, not a universal requirement.** In particular, for $F(w)=w^2$, initializing at $0$ already gives an optimum. For $F(w)=(w-3)^2$, starting at $0$ with $\eta=1/4$ yields $w^{(1)}=1.5$.

**Distinguish two plots:** $F(w)$ versus the parameter $w$ shows the objective landscape; $F(w^{(t)})$ versus the iteration $t$ shows algorithmic progress. They do **not** have the same horizontal axis.

## 2. Lipschitz continuity and a gradient bound [2026/27 confirmed — following the update]

$F$ is $G$-Lipschitz on a set $S$ if, for all $x_1,x_2\in S$,

$$
\boxed{|F(x_1)-F(x_2)|\le G\|x_1-x_2\|}.
$$

This controls changes in **function value**, not directly changes of the gradient (the latter is a different concept, often called a Lipschitz gradient or smoothness).

If $F$ is differentiable on an open domain and $G$-Lipschitz, then

$$
\boxed{\|\nabla F(w)\|\le G}.
$$

**One-dimensional proof, with the sign detail that caused confusion.** For $h>0$,

$$
\frac{F(w+h)-F(w)}{h}
\le \frac{|F(w+h)-F(w)|}{h}
\le \frac{G|h|}{h}=G.
$$

The first inequality follows from $a\le |a|$. Taking $h\to0^+$ gives $F'(w)\le G$. Likewise, $-a\le|a|$ gives $-F'(w)\le G$, hence $|F'(w)|\le G$. Differentiability ensures the relevant one-sided limits agree. In $\mathbb R^d$, applying the bound to unit directional derivatives yields $\|\nabla F(w)\|\le G$.

**Example and domain dependence.** $F(w)=w^2$ is not globally Lipschitz on $\mathbb R$, but it is $8$-Lipschitz on $[-4,4]$: $|x_1^2-x_2^2|=|x_1-x_2||x_1+x_2|\le8|x_1-x_2|$.

## 3. Convex sets and convex functions [2026/27 confirmed]

A set $S\subseteq\mathbb R^d$ is convex if

$$
\forall x_1,x_2\in S,\ \forall\alpha\in[0,1],\quad
\alpha x_1+(1-\alpha)x_2\in S.
$$

Geometrically, every segment between two points of $S$ stays inside $S. A filled disk is convex; an annulus is not (a segment can cross its central hole).

For a **convex domain** $S$, $F:S\to\mathbb R$ is convex if

$$
\boxed{F(\alpha x_1+(1-\alpha)x_2)\le\alpha F(x_1)+(1-\alpha)F(x_2)}
$$

for all $x_1,x_2\in S$ and $\alpha\in[0,1]$. The graph of a one-dimensional convex function lies below or on every chord joining two of its graph points.

The convexity of the domain does **not** imply convexity of every function on it. For example, $x^2$ is convex on $\mathbb R$, while $-x^2$ is not.

**One-dimensional coordinate interpretation:** $u,w$ are *values along the horizontal input axis*, whereas $(u,F(u))$ and $(w,F(w))$ are points on the graph. The difference $u-w$ is the *signed horizontal displacement*, not an arc length along the curve and not the vertical displacement.

## 4. Equivalent first-order characterization of convexity [2026/27 confirmed — handwritten proof]

For a differentiable function $F:\mathbb R^d\to\mathbb R$,

$$
\boxed{F\text{ convex}\quad\Longleftrightarrow\quad
F(u)\ge F(w)+\langle\nabla F(w),u-w\rangle\quad\forall u,w.}
$$

The term $F(w)+\langle\nabla F(w),u-w\rangle$ is the height at $u$ of the affine tangent approximation built at $w$. In one dimension it is $F(w)+F'(w)(u-w)$: **starting height + tangent slope × horizontal displacement**. For $F(x)=x^2$, $w=1,u=2$, tangent height $=1+2(2-1)=3$, whereas $F(2)=4$. The inequality is not necessarily equality.

### 4.1 Convexity implies the supporting-hyperplane inequality

For any $u,w$ and $\alpha\in(0,1]$, convexity gives

$$
F(\alpha u+(1-\alpha)w)\le\alpha F(u)+(1-\alpha)F(w).
$$

Rewrite the argument as $\alpha u+(1-\alpha)w=w+\alpha(u-w)$, subtract $F(w)$, then divide by $\alpha>0$:

$$
\frac{F(w+\alpha(u-w))-F(w)}{\alpha}\le F(u)-F(w).
$$

Take $\alpha\to0^+$. **Differentiability** means the left side is the directional derivative $\langle\nabla F(w),u-w\rangle$. Thus

$$
F(u)\ge F(w)+\langle\nabla F(w),u-w\rangle.
$$

The rewrite $w+\alpha(u-w)$ is deliberate: it displays a small displacement from $w$ in direction $u-w$, permitting a directional derivative. Division by positive $\alpha$ preserves the inequality.

### 4.2 Supporting-hyperplane inequality implies convexity

Assume the inequality holds for every pair. Choose $x_1,x_2$ and $\alpha\in[0,1]$, and define the *intermediate point*

$$
z=\alpha x_1+(1-\alpha)x_2.
$$

Apply the hypothesis twice, **with the same tangent point $z$**:

$$
F(x_1)\ge F(z)+\langle\nabla F(z),x_1-z\rangle,
$$

$$
F(x_2)\ge F(z)+\langle\nabla F(z),x_2-z\rangle.
$$

Multiply the first by $\alpha\ge0$ and the second by $1-\alpha\ge0$, then add:

$$
\begin{aligned}
\alpha F(x_1)+(1-\alpha)F(x_2)
&\ge F(z)\\
&\quad+\langle\nabla F(z),\,\alpha(x_1-z)+(1-\alpha)(x_2-z)\rangle.
\end{aligned}
$$

The remaining vector equals zero:

$$
\begin{aligned}
\alpha(x_1-z)+(1-\alpha)(x_2-z)
&=\alpha x_1+(1-\alpha)x_2-[\alpha+(1-\alpha)]z\\
&=z-z=0.
\end{aligned}
$$

Therefore the inner product is zero, and

$$
F(\alpha x_1+(1-\alpha)x_2)\le\alpha F(x_1)+(1-\alpha)F(x_2).
$$

This proves the other implication. **Source/proof nuance:** the lecturer's proof uses the same $z$ and cancels the gradient terms; the expanded algebra above is tutor clarification.

## 5. Averaged gradient descent and Jensen [2026/27 confirmed — following page]

Run gradient descent for $T$ steps and form the average

$$
\bar w=\frac1T\sum_{t=1}^T w^{(t)}.
$$

Let $w^\star$ be a global minimizer. The quantity of interest is the **objective gap** $F(\bar w)-F(w^\star)$, not necessarily the Euclidean distance $\|\bar w-w^\star\|$.

By convexity/Jensen,

$$
F(\bar w)\le\frac1T\sum_{t=1}^TF(w^{(t)}).
$$

Subtracting $F(w^\star)$ and writing it as the average of $T$ identical copies gives

$$
F(\bar w)-F(w^\star)
\le\frac1T\sum_{t=1}^T[F(w^{(t)})-F(w^\star)].
$$

**Why not equality?** A convex graph may lie *strictly* below a chord. With $F(x)=x^2$ and iterates $0,2$, the value at their average is $F(1)=1$, but the average of their values is $(0+4)/2=2$. Equality may hold for identical points or a function affine on the relevant segment.

Apply the previously proven first-order condition with $u=w^\star$, $w=w^{(t)}$:

$$
F(w^{(t)})-F(w^\star)
\le\langle\nabla F(w^{(t)}),w^{(t)}-w^\star\rangle.
$$

Hence

$$
\boxed{
F(\bar w)-F(w^\star)
\le\frac1T\sum_{t=1}^T
\langle\nabla F(w^{(t)}),w^{(t)}-w^\star\rangle.
}
$$

**The factor $1/T$ already comes from Jensen/the average of iterates**. It is not newly invented when the next lemma is substituted.

### 5.1 The squared-norm identity and telescoping lemma

Start with the GD step and subtract $w^\star$:

$$
w^{(t+1)}-w^\star=w^{(t)}-w^\star-\eta\nabla F(w^{(t)}).
$$

Expand using $\|a-b\|^2=\|a\|^2-2\langle a,b\rangle+\|b\|^2$:

$$
\begin{aligned}
\|w^{(t+1)}-w^\star\|^2
&=\|w^{(t)}-w^\star\|^2\\
&\quad-2\eta\langle\nabla F(w^{(t)}),w^{(t)}-w^\star\rangle
+\eta^2\|\nabla F(w^{(t)})\|^2.
\end{aligned}
$$

Rearrange to isolate precisely the inner product above:

$$
\begin{aligned}
\langle\nabla F(w^{(t)}),w^{(t)}-w^\star\rangle
&=\frac{\|w^{(t)}-w^\star\|^2-\|w^{(t+1)}-w^\star\|^2}{2\eta}\\
&\quad+\frac{\eta}{2}\|\nabla F(w^{(t)})\|^2.
\end{aligned}
$$

On summing over $t=1,\ldots,T$, the squared-distance terms telescope: every intermediate $+\|w^{(t)}-w^\star\|^2$ cancels the matching negative term. Thus

$$
\begin{aligned}
\sum_{t=1}^T\langle\nabla F(w^{(t)}),w^{(t)}-w^\star\rangle
&=\frac{\|w^{(1)}-w^\star\|^2-\|w^{(T+1)}-w^\star\|^2}{2\eta}\\
&\quad+\frac{\eta}{2}\sum_{t=1}^T\|\nabla F(w^{(t)})\|^2\\
&\le\frac{\|w^{(1)}-w^\star\|^2}{2\eta}
+\frac{\eta}{2}\sum_{t=1}^T\|\nabla F(w^{(t)})\|^2.
\end{aligned}
$$

If the first indexed iterate is initialized as $w^{(1)}=0$, its distance to the optimum satisfies $\|w^{(1)}-w^\star\|^2=\|w^\star\|^2$. Some other presentations call the zero initial vector $w^{(0)}$; shift iteration indices consistently. Combining with Jensen gives

$$
\boxed{
F(\bar w)-F(w^\star)
\le\frac1T\left[
\frac{\|w^\star\|^2}{2\eta}
+\frac{\eta}{2}\sum_{t=1}^T\|\nabla F(w^{(t)})\|^2
\right].
}
$$

**Key student's question:** the telescoping lemma bounds a *sum*, while Jensen bounds the objective gap by *$1/T$ times that sum*. Substitution therefore keeps the previously existing $1/T$.

### 5.2 Lipschitz bound and step-size choice [2026/27 confirmed — theorem/convergence continuation; algebra expanded by tutor]

If $\|\nabla F(w)\|\le G$ and $\|w^\star\|\le B$, then

$$
\sum_{t=1}^T\|\nabla F(w^{(t)})\|^2\le TG^2,
$$

hence

$$
F(\bar w)-F(w^\star)
\le\frac{B^2}{2\eta T}+\frac{\eta G^2}{2}.
$$

Choose $\eta=B/(G\sqrt T)$ (assuming $B,G>0$), by minimizing the upper bound over $\eta>0$. One obtains

$$
\boxed{0\le F(\bar w)-F(w^\star)\le\frac{BG}{\sqrt T}}.
$$

This is an $O(T^{-1/2})$ **objective-value guarantee for the averaged iterate**, not a claim that every step decreases $F$, nor necessarily that the parameter vectors converge to a unique minimizer. To reduce this bound by a factor of ten, increase $T$ by a factor of one hundred.

**Mathematical scope:** the argument assumes the updates and gradient bounds apply on their domain. If using a constrained domain, a projection step and its appropriate lemma must be accounted for; do not silently transfer the unprojected proof.

## 6. Non-differentiable convex functions and subgradients [2026/27 confirmed — lecturer's question]

The lecturer now asks: **What about non-differentiable $F$?**

A vector $g\in\mathbb R^d$ is a **subgradient** at $w$ of convex $F$ if

$$
\boxed{F(u)\ge F(w)+\langle g,u-w\rangle\qquad\forall u\in\mathbb R^d.}
$$

The **subdifferential** is a *set*:

$$
\partial F(w)=\{g:\ F(u)\ge F(w)+\langle g,u-w\rangle\ \forall u\}.
$$

When a convex function is differentiable at $w$, $\partial F(w)=\{\nabla F(w)\}$. At a kink it can contain multiple slopes. Geometrically (1D), every valid subgradient gives a supporting line through $(w,F(w))$ lying below the graph.

### 6.1 Absolute value [2026/27 confirmed — example]

For $F(x)=|x|$ at $w=0$, the condition is $|u|\ge gu$ for all $u$.

- For $u>0$: $u\ge gu$ gives $g\le1$.
- For $u<0$: $-u\ge gu$; dividing by **negative** $u$ reverses the inequality and gives $g\ge-1$.
- For $u=0$: $0\ge0$ gives no restriction.

Thus $\partial F(0)=[-1,1]$, while $\partial F(x)=\{-1\}$ for $x<0$ and $\{1\}$ for $x>0$.

### 6.2 Positive-part / ReLU [2026/27 confirmed — example]

For $F(x)=\max\{x,0\}$, the function is continuous but **not differentiable at $0$**: its left derivative is $0$, right derivative $1$. At the kink, the subgradient condition is $F(u)\ge gu$.

For $u>0$, $u\ge gu$ implies $g\le1$; for $u<0$, $0\ge gu$ implies $g\ge0$; for $u=0$ there is no restriction. Therefore

$$
\boxed{
\partial F(x)=
\begin{cases}
\{0\},&x<0,\\
[0,1],&x=0,\\
\{1\},&x>0.
\end{cases}
}
$$

**Common error corrected in dialogue:** $F(u)=0$ for $u<0$ does *not* mean the right-hand side $gu$ is zero. Because $u<0$ is *nonzero*, the condition $0\ge gu$ constrains $g\ge0$. This differs from the special case $u=0$, where $0\ge0$ is always true.

## 7. Single-example hinge-loss subgradient [2026/27 confirmed]

For $(x_i,y_i)$ with $y_i\in\{-1,+1\}$,

$$
\ell_i(w)=(1-y_i\langle x_i,w\rangle)_+.
$$

It is the maximum of the affine function $1-y_i\langle x_i,w\rangle$ and the constant $0$. The active branch changes at signed score $s_i=y_i\langle x_i,w\rangle=1$.

When $s_i<1$, the gradient is **$-y_ix_i$**, not universally $-1$ (the latter comes from the numerical example $x_i=y_i=1$). When $s_i>1$, the loss is locally constant at zero, hence gradient $0$. If $x_i\ne0$ and $s_i=1$, the function is non-differentiable and the subgradient set is

$$
\boxed{
\partial\ell_i(w)=\{-\alpha y_ix_i:\alpha\in[0,1]\}.
}
$$

**Why the interval?** At the boundary $\ell_i(w)=0$. Substituting $g=-\alpha y_ix_i$ into the supporting inequality yields

$$
\max\{0,1-y_i\langle x_i,u\rangle\}
\ge\alpha(1-y_i\langle x_i,u\rangle).
$$

Writing $a=1-y_i\langle x_i,u\rangle$, this becomes $\max\{0,a\}\ge\alpha a$: positive $a$ requires $\alpha\le1$; negative $a$ requires $\alpha\ge0$. All and only convex combinations of the two active affine gradients are allowed. For completeness, the full characterization (in the realizable kink case $x_i\ne0$) is

$$
\partial\ell_i(w)=
\begin{cases}
\{-y_ix_i\},&s_i<1,\\
\{-\alpha y_ix_i:\alpha\in[0,1]\},&s_i=1,\\
\{0\},&s_i>1.
\end{cases}
$$

**Important distinctions:** $w=0$ generally gives $s_i=0$, not the kink $s_i=1$. Also, *loss value zero* does not automatically imply differentiability at that point; at $s_i=1$ the loss is zero but is kinked.

## 8. Subgradient descent for the Soft-SVM objective [2026/27 confirmed]

The handwritten lecture now moves from a single example to the average of $m$ examples:

$$
\boxed{F(w)=\frac{\lambda}{2}\|w\|^2+\frac1m\sum_{i=1}^m(1-y_i\langle x_i,w\rangle)_+.}
$$

The regularizer has gradient $\nabla(\lambda\|w\|^2/2)=\lambda w$. Choose subgradient $-y_ix_i$ for an example whose signed score is $<1$, and $0$ for a score $\ge1$ (the value $0$ is permitted at equality). By the sum rule, a **selected subgradient** is

$$
\boxed{
g_F(w)=\lambda w-\frac1m
\sum_{i:\,y_i\langle x_i,w\rangle<1}y_ix_i
\in\partial F(w).
}
$$

The notes sometimes use $\partial F(w)$ as shorthand for the selected vector; strictly, $\partial F(w)$ is the full set of allowed subgradients.

The update is

$$
\begin{aligned}
w^{(t+1)}
&=w^{(t)}-\eta g_F(w^{(t)})\\
&=(1-\eta\lambda)w^{(t)}
+\frac{\eta}{m}\sum_{i:\,y_i\langle x_i,w^{(t)}\rangle<1}y_ix_i.
\end{aligned}
$$

Start from the zero vector and, following the lecturer's averaged-iterate presentation, output an average of iterates when specified. The first term is regularization-induced scaling (a contraction when $0<\eta\lambda<1$); the second corrects margin-violating examples. Score $s_i$ can be positive and still be $<1$: those correctly classified but low-margin points also contribute.

### 8.1 Cost per iteration: why randomize?

Dense vector scaling costs $O(d)$. Checking all $m$ signed scores and computing their contributions costs $O(md)$ per full subgradient step. This is the lecturer's computational motivation for the stochastic method; $m$ denotes number of examples, $d$ number of coordinates.

## 9. Stochastic subgradient descent [2026/27 confirmed]

Instead of processing all samples on every update, pick a random index $I_t$ **uniformly** from $\{1,\dots,m\}$, so $P(I_t=i)=1/m$. Define the random vector

$$
\boxed{
V_t=\lambda w^{(t)}
-y_{I_t}x_{I_t}
\mathbf1\{y_{I_t}\langle x_{I_t},w^{(t)}\rangle<1\}.
}
$$

**Why the indicator?** It expresses the single-example piecewise subgradient in one formula. It is $1$ if the selected example has signed score $<1$, otherwise $0$. **It does not choose the sample**—the random index $I_t$ does that.

### 9.1 Why its conditional expectation is the full selected subgradient

Hold the data and current $w^{(t)}$ fixed; only the sampled index is random. By the definition of expectation of a discrete vector variable and linearity,

$$
\begin{aligned}
\mathbb E[V_t\mid w^{(t)}]
&=\frac1m\sum_{i=1}^m\Big[
\lambda w^{(t)}-
y_ix_i\mathbf1\{y_i\langle x_i,w^{(t)}\rangle<1\}\Big]\\
&=\lambda w^{(t)}
-\frac1m\sum_{i=1}^m
y_ix_i\mathbf1\{y_i\langle x_i,w^{(t)}\rangle<1\}\\
&=\lambda w^{(t)}
-\frac1m\sum_{i:\,y_i\langle x_i,w^{(t)}\rangle<1}y_ix_i\\
&=g_F(w^{(t)})\in\partial F(w^{(t)}).
\end{aligned}
$$

The factor $1/m$ is the uniform sampling probability; **do not divide by the number of margin-violating examples**. Examples whose indicator vanishes still belong to the sampling space and contribute zero to this term. The regularizer remains because $(1/m)\sum_i\lambda w^{(t)}=\lambda w^{(t)}$.

The lecturer writes $\mathbb E[V_t]$ informally; conditioning on the current iterate is more precise since $w^{(t)}$ may itself depend on earlier random draws.

A stochastic update has $O(d)$ cost for dense samples, as opposed to $O(md)$ for a full-data step. An unbiased *estimate* does not mean each realized step is the exact full gradient, nor that each step decreases the loss. Additional assumptions are needed for stochastic convergence guarantees.

## 10. XOR and a cycling Perceptron [2026/27 confirmed — closing example; Week 3 §1]

The lecturer closes with XOR in $\mathbb R^2$, assigning $+1$ when coordinates differ and $-1$ when they agree:

| Point $x_i$ | Label $y_i$ |
| --- | --- |
| $(1,1)$ | $-1$ |
| $(1,-1)$ | $+1$ |
| $(-1,1)$ | $+1$ |
| $(-1,-1)$ | $-1$ |

The batch Perceptron starts at $w^{(0)}=(0,0)$ and selects a point with nonpositive signed score $y_i\langle w,x_i\rangle\le0$, updating $w\leftarrow w+y_ix_i$.

With the listed order:

$$
\begin{aligned}
w^{(0)}&=(0,0),\\
w^{(1)}&=(0,0)-(1,1)=(-1,-1),\\
w^{(2)}&=(-1,-1)+(1,-1)=(0,-2),\\
w^{(3)}&=(0,-2)+(-1,1)=(-1,-1),\\
w^{(4)}&=(-1,-1)+(-1)(-1,-1)=(0,0).
\end{aligned}
$$

Thus the selected sequence gives a **cycle** back to initialization. This shows one repeated schedule cycles, not that all schedules must follow exactly the same cycle.

The Perceptron's finite-update theorem requires strict separability; XOR is not separable, **even with an affine bias**. To prove this, write $w=(w_1,w_2)$ and bias $b$. Correct positive classification requires

$$
w_1-w_2+b\ge0,\qquad -w_1+w_2+b\ge0,
$$

which implies $2b\ge0$. Negative classification requires

$$
w_1+w_2+b<0,\qquad -w_1-w_2+b<0,
$$

which implies $2b<0$: impossible. These inequalities use the course's $\operatorname{sign}(0)=+1$ convention.

**Subtlety:** a $+1$ sample with signed score exactly zero may be predicted correctly by this tie convention, yet the Perceptron pseudocode still updates when signed score $\le0$. This is a *zero functional-margin* rule, not merely a test of misclassification.

**Model-class versus optimizer:** choosing Perceptron, hard-SVM, or soft-SVM changes what objective is optimized; it does not automatically allow the underlying linear feature representation to express XOR. Soft-SVM remains feasible but cannot yield a perfect XOR linear separator in the original coordinates.

## 11. Exam transfer — disks versus all convex regions [Exam source, NOT part of handwritten Lecture 8]

A later study discussion revisited [Secondo Appello, question 4](https://drive.google.com/file/d/178jYMtMOvQXA9U1gaKnSRJoVDYwdwI2Z/view), asking the VC dimension of **all disks** $h_{a,b,r}(x)=\mathbf1\{(x_1-a)^2+(x_2-b)^2\le r^2\}$. It is $3$ (not infinite). This is a useful *separate exam revision*, not a claim about Lecture 8 content.

Why the previously studied class of *all convex subsets* of $\mathbb R^2$ can have infinite VC dimension while disks have VC dimension $3$: every disk is convex, but not every convex region is a disk. Changing the center and radius does not allow arbitrary thin convex polygons.

A learner-proposed four-point square labeling $(+1,-1,+1,-1)$ on cyclically ordered vertices is **not realizable by any disk**. If opposite vertices $A,C$ must lie inside and the other two $B,D$ outside, then for an arbitrary disk center $c$, the positive constraints imply $\|A-c\|^2+\|C-c\|^2\le2r^2$ while negative constraints imply $\|B-c\|^2+\|D-c\|^2>2r^2$. On a square the two left-hand sums are equal: contradiction.

**Proof discipline:** showing that *one particular square* is not shattered does **not** prove $\mathrm{VCdim}\le3$. A general upper bound must exclude **every** four-point configuration; remember to distinguish configurations with a point inside the triangle of the other three from four points in convex position. Three suitably placed points can be shattered by disks. Finish the complete upper-bound argument independently before claiming exam mastery.

---

## 12. Active-recall and proof checklist (not yet independently verified)

1. Explain why the first-order characterization is **equivalent** to convexity only under differentiability; prove both directions, especially the cancellation at $z=\alpha x_1+(1-\alpha)x_2$.
2. Explain what $u-w$ means geometrically in one dimension and what $F(w)+F'(w)(u-w)$ measures.
3. Reproduce Jensen for the average iterate; state precisely why the extra factor $1/T$ persists when substituting the telescoping lemma.
4. Derive the squared-norm identity and telescope it without notes. Identify where the zero initialization yields $\|w^\star\|^2$.
5. Derive the $BG/\sqrt T$ guarantee and state the assumptions, the average-iterate scope, and how the bound scales with $T$.
6. Compute $\partial|x|$ and $\partial\max\{0,x\}$ at their kinks; pay attention when dividing an inequality by a negative number.
7. Derive the hinge subgradient at signed score $1$ from the supporting inequality, and distinguish the **loss value**, **gradient**, and **subdifferential set**.
8. Derive the Soft-SVM selected subgradient and explain the $O(md)$ versus $O(d)$ per-step costs.
9. Explain separately what the random index $I_t$ and indicator $\mathbf1\{\cdot\}$ do. Compute $\mathbb E[V_t\mid w^{(t)}]$ and justify keeping the sampling probability $1/m$.
10. Reconstruct the XOR four-step Perceptron cycle and the affine nonseparability proof.
11. Separate the VC-dimension of the *family of disks* from that of the *family of all convex sets*. Formal lower/upper proofs remain separate work.

### Personal pitfalls and corrected misconceptions

- Confused **function value 0** with **derivative 0** at a hinge kink; corrected by checking one-sided slopes.
- Asked why the $u<0$ case of $\max\{0,u\}$ constrains $g$; resolved that $u<0$ is not $u=0$ and division reverses the inequality.
- Initially read $w,u$ as graph points rather than horizontal input values; clarified input values versus pairs $(w,F(w))$.
- Asked why taking an absolute value changes equality to inequality in the Lipschitz proof; the rule is $a\le|a|$, with $h>0$ in the illustrated step.
- Questioned the origin of $1/T$ when passing to the quadratic-norm bound; it was inherited from Jensen.
- Asked whether the left slope of the hinge loss should be $-y_ix_i$; yes, the numerical $-1$ was just one-dimensional $y_i=x_i=1$.
- Asked why SGD needs an indicator; it *switches the sampled example's contribution on or off* and does not perform random selection.
- The exam VC-dimension misconception was about confusing one labeling or one geometric family with a full shattering argument.

**Study status (2026-11-24):** guided coverage complete; independently demonstrated mastery is not yet established. Prioritize one no-notes reconstruction of the convexity equivalence and telescoping proof, then one SGD expectation calculation and one 4-point disk shattering upper-bound exercise.

**Next in course chronology:** [[LESSON 9]] only when that current-year lecture is opened and studied; do not infer its contents from this note.
