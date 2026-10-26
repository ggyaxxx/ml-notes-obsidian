---
title: "Shattering and VC Dimension"
aliases:
  - "VC Dimension"
  - "Shattering"
date: 2026-10-26
course: "Machine Learning — Sapienza"
academic_year: "2026/2027"
type: concept-note
status: "Explained and reasoned through with guidance; independent exam-style reconstruction pending"
tags:
  - machine-learning
  - statistical-learning
  - vc-dimension
  - shattering
  - hypothesis-classes
  - convexity
  - sample-complexity
---

# Shattering and VC Dimension

> [!NOTE]
> **Central question.** A hypothesis family can be infinite, yet generalize well. What matters is not just how many functions it contains, but **which labelings it can produce on fixed finite sets of inputs**. Shattering measures this ability; VC dimension summarizes its size.

**Prerequisites:** [[LESSON 2]] (risk and overfitting), [[LESSON 3]] (ERM, uniform convergence, union bound), [[LESSON 4]] (realizability and thresholds).

**Source and attribution (2026/27 confirmed):**
- [Handwritten Lecture 4, 06 Oct 2026](https://drive.google.com/file/d/1OfAfH81rnBwukGjiNHf_unNg7FQ3o9pc/view): examples of thresholds, intervals, unions of $k$ intervals, convex subsets in the plane; the general VC theorem. The automatic transcription of handwriting is noisy, so formal statements and notation below have been checked against the handout rather than presented as verbatim lecture quotes.
- [Week 1 — Statistical Learning, v2, 06 Oct 2026](https://drive.google.com/file/d/1sw45w97b56JPTQJEdyOz2qjMkfsVWIFS/view), §6: Definitions 6.1–6.2, Theorem 6.3, Examples 6.4–6.7; §7: risk decomposition.
- [Exercise Sheet 1, v1, 06 Oct 2026](https://drive.google.com/file/d/1DpZ45CXHWnegIKxVYsLDldAZ724l6fFV/view), Exercise 1.4(a)–(i): proof practice. In that sheet the binary labels are $\{-1,+1\}$; we use $\{0,1\}$ here for clarity, without changing the combinatorial result.
- **Tutor-added explanations:** block-counting proof steps, explicit quantified statements, geometric examples and the discussion of personal misunderstandings. They should not be attributed as the lecturer's words.

**Progress as of 26 Oct 2026:** The proofs for thresholds, intervals and unions of $k$ intervals were developed via guided conversation. The student recognized the existential quantifier in the definition and resolved an apparent paradox about convex classes. **Not yet checked without prompts**.

---

## 1. Definition: what must a hypothesis family do?

Fix **one set of distinct inputs** $T=\{x_1,\ldots,x_m\}\subseteq\mathcal X$. Do not move its points while testing a labeling. Let $\mathcal H$ be a **family of functions** $h:\mathcal X\to\{0,1\}$.

We say that $\mathcal H$ **shatters** $T$ if every possible binary labeling of those $m$ points can be realized by *at least one* function in the family:

$$ \boxed{\forall(y_1,\ldots,y_m)\in\{0,1\}^m,\ \exists h\in\mathcal H:\ \forall i\in\{1,\ldots,m\},\ h(x_i)=y_i.} $$

The inputs are fixed; **the classifier is allowed to change when the labeling changes**.

- **$\forall$ labelings, $\exists$ a fitting classifier** is the correct quantifier order.
- The same classifier does **not** need to assign both 0 and 1 to the same point.
- The labelings number $2^m$, but one hypothesis may sometimes realize only one labeling *on the fixed set*.
- Shattering concerns the combinatorics of **input locations**, not whether the labels came from the unknown data distribution.

> [!WARNING]
> **Notation trap.** Here $T$ is a fixed set of inputs used to test shattering, whereas in the statistical learning theorems $S=((x_1,y_1),\ldots,(x_n,y_n))$ denotes a random labeled training sample. Both are sometimes called $S$ in textbooks; do not conflate them.

**VC dimension** is the size of the largest shattered finite set:

$$ \boxed{\operatorname{VCdim}(\mathcal H)=\sup\{\,|T|:T\subseteq\mathcal X\text{ finite and shattered by }\mathcal H\,\}.} $$

If shattered sets exist of arbitrarily large finite size, we define $\operatorname{VCdim}(\mathcal H)=\infty$.

### Existence versus universality

If $\operatorname{VCdim}(\mathcal H)=d$, we know **there exists at least one** shattered set of $d$ points; we do **not** know that every set of $d$ points is shattered.

To establish an exact finite value, use two logically different claims:

| Goal | What must be shown? | Proof strategy |
|---|---|---|
| **Lower bound** $\operatorname{VCdim}(\mathcal H)\ge d$ | $\exists T$ with $|T|=d$ that **is** shattered | Construct points and a classifier for **every labeling** |
| **Upper bound** $\operatorname{VCdim}(\mathcal H)\le d$ | **No** set of $d+1$ points is shattered | Take an **arbitrary** $(d+1)$-point set and exhibit an impossible labeling |

Finding **one non-shattered set** does not prove an upper bound.

### Why excluding $d+1$ excludes all larger sizes

**Hereditary property:** if $\mathcal H$ shatters $T$, then it shatters **every subset** $U\subseteq T$.

To prove this, take any desired labeling of $U$, assign arbitrary labels to the other points in $T$, and use the hypothesis guaranteed by shattering of $T$. Its restriction to $U$ is correct.

Consequently, if **no set of size $d+1$** is shattered, **no larger set** can be shattered either. This makes the upper-bound method valid.

**Edge case:** the empty set is shattered by any nonempty family (the sole empty labeling can be realized); it does not interfere with the positive-dimensional examples below.

---

## 2. Thresholds on the real line: VC dimension 1

Let $\mathcal X=\mathbb R$ and

$$ \mathcal H_{\rm thr}=\{h_\theta:\theta\in\mathbb R\},\qquad h_\theta(x)=\mathbf 1\{x\ge\theta\}. $$

**Lower bound ($\ge1$).** Fix any point $x_0$.

- To label it 1, choose $\theta\le x_0$.
- To label it 0, choose $\theta>x_0$.

These are **different** classifiers, both in the family, so $\{x_0\}$ is shattered.

**Upper bound ($\le1$).** Take **any** two distinct points $x_1<x_2$. The labeling $(1,0)$ would require simultaneously

$$ h_\theta(x_1)=1\Rightarrow \theta\le x_1,\qquad h_\theta(x_2)=0\Rightarrow\theta>x_2. $$

Then $\theta\le x_1<x_2<\theta$, a contradiction. Hence no pair is shattered and

$$ \boxed{\operatorname{VCdim}(\mathcal H_{\rm thr})=1.} $$

**Course convention:** In Theorem 5.2, thresholds are restricted to $\theta\in[0,1]$ and $\mathcal X=[0,1]$. For example $\{0.4\}$ is still shattered; the VC dimension remains 1. But the boundary singleton $\{1\}$ cannot receive label 0 in that restricted model, so **not every singleton** is shattered. Exercise 1.4(a) instead takes $\theta\in\mathbb R$ and the entire real line.

**Personal sticking point resolved:** “One point cannot receive both labels from a single classifier” is true but irrelevant: shattering permits **one classifier per labeling**.

---

## 3. One closed interval on the line: VC dimension 2

$$ \mathcal H_{\rm int}=\{h_{a,b}:a\le b,\ a,b\in\mathbb R\},\qquad h_{a,b}(x)=\mathbf 1\{a\le x\le b\}. $$

**Lower bound ($\ge2$).** Fix any distinct $x_1<x_2$. For example $x_1=0.3$, $x_2=0.7$. Every labeling is possible:

| $(h(x_1),h(x_2))$ | Example of $[a,b]$ |
|---|---|
| $(0,0)$ | $[0.4,0.6]$: between the points |
| $(1,0)$ | $[0.2,0.5]$: includes only $x_1$ |
| $(0,1)$ | $[0.5,0.8]$: includes only $x_2$ |
| $(1,1)$ | $[0.2,0.8]$: includes both |

**Upper bound ($\le2$).** Take arbitrary $x_1<x_2<x_3$. The labeling $(1,0,1)$ cannot be realized: an interval containing $x_1$ and $x_3$ must satisfy

$$ a\le x_1<x_2<x_3\le b, $$

which forces $x_2\in[a,b]$ and thus labels $x_2$ as 1.

Hence

$$ \boxed{\operatorname{VCdim}(\mathcal H_{\rm int})=2.} $$

> [!TIP]
> **Endpoint pitfall from our work.** For **closed** intervals, the endpoints are included. $[0.3,0.7]$ labels *both* $0.3$ and $0.7$ as 1, not $(1,0)$. Use non-strict inequalities when writing necessary conditions; strict inequalities are often convenient *sufficient constructions*.

### Geometry versus counting

An interval is **connected**: it cannot label two surrounding points 1 while leaving a point between them labeled 0. This, not a mere failure to find convenient values of $a,b$, gives a universal impossibility proof.

---

## 4. Unions of $k$ intervals: VC dimension $2k$

Let

$$ \mathcal H_k=\left\{h:\ h(x)=\mathbf1\left\{x\in\bigcup_{j=1}^{k}[a_j,b_j]\right\},\quad a_j\le b_j\right\}. $$

A **block (run) of 1s** is a maximal consecutive sequence of positive labels when the points are arranged in increasing order.

Examples:

| Labeling | Positive blocks |
|---|---:|
| $(1,1,1,0)$ | 1 |
| $(1,0,0,1)$ | 2 |
| $(0,1,1,0,1,0)$ | 2 |
| $(1,0,1,0,1)$ | 3 |
| $(0,0,0,0)$ | 0 |

Each positive block can be covered by **one interval** without covering any other sample point labeled 0: if the block is $x_i,\ldots,x_j$, choose $[x_i,x_j]$. When there are fewer than $k$ blocks, unused intervals can be placed beyond all the chosen points. An all-zero labeling also needs no positive block.

### Lower bound: at least $2k$

Fix any $2k$ distinct ordered points

$$ x_1<x_2<\cdots<x_{2k}. $$

Among $2k$ binary labels, there can be **at most $k$ positive blocks**: separating $r$ positive blocks requires at least $r-1$ intervening zero labels, hence at least $r+(r-1)=2r-1$ positions. With only $2k$ positions, $r\le k$.

Therefore **every labeling** has at most $k$ blocks, and our $k$ intervals cover them without including sample points that should be 0. Thus the fixed set is shattered:

$$ \boxed{\operatorname{VCdim}(\mathcal H_k)\ge 2k.} $$

### Upper bound: at most $2k$

Take **any** $2k+1$ distinct ordered points and assign the alternating labels

$$ (1,0,1,0,\ldots,0,1). $$

There are $k+1$ separate positive blocks. No single interval can cover two such blocks without also containing an intervening zero-labeled point. The labeling therefore needs at least $k+1$ intervals, but only $k$ are allowed. Every $(2k+1)$-point set has an impossible labeling:

$$ \boxed{\operatorname{VCdim}(\mathcal H_k)\le 2k.} $$

Combine the bounds:

$$ \boxed{\operatorname{VCdim}(\mathcal H_k)=2k.} $$

**Useful related fact, not the definition of VC dimension:** On $n$ ordered points the maximum possible number of positive blocks is $\lceil n/2\rceil$, so that many intervals suffice for **any labeling of those points**. An individual labeling can need much fewer intervals. The term “maximum number of blocks” must not be confused with the **upper bound on VC dimension**.

> [!WARNING]
> **Scope of the blocks trick.** Block counting works here because 1D intervals are connected. It is **not** a universal technique for all hypothesis families. A one-sided threshold cannot realize $(1,0)$ on two ordered points although it has only one positive block.

---

## 5. Convex subsets of the plane: infinite VC dimension

**First define the *classifier*, not only the geometry.** The input is a point $x=(x_1,x_2)\in\mathbb R^2$. Choose a region $C$ of the plane and classify by **membership**:

$$ h_C(x)=\begin{cases}1,&x\in C,\\0,&x\notin C.\end{cases} $$

We restrict the eligible regions $C$ to **convex** subsets of $\mathbb R^2$:

$$ \boxed{\mathcal H_{\rm conv}=\{h_C:C\subseteq\mathbb R^2\text{ convex}\}.} $$

$\mathcal H_{\rm conv}$ is a **family of classification functions**, not merely a family of geometric sets. Different convex regions $C_1,C_2$ define (generally) different classifiers $h_{C_1},h_{C_2}$ belonging to the **same hypothesis family**.

### What “convex” means; roles of $u,v,t$

A region $C$ is convex if

$$ \boxed{u,v\in C\ \Longrightarrow\ (1-t)u+tv\in C\quad\forall t\in[0,1].} $$

Here $u$ and $v$ are **arbitrary points of the region** (vectors with two coordinates); there is **no left-to-right ordering** between them. The parameter $t$ describes every point on their connecting segment: $t=0$ yields $u$, $t=1$ yields $v$, and $t=1/2$ yields their midpoint. Convexity requires **all** intermediate values of $t$, not merely one.

**A singleton is convex.** If $C=\{x_2\}$, then any $u,v\in C$ must both equal $x_2$, so $(1-t)u+tv=x_2\in C$ for *every* $t$. Thus choosing $C=\{x_2\}$ can produce $(0,1,0)$ on three distinct points. This is **not** the same as “set $t=0$”: $t$ tests membership of intermediate points; $C$ specifies which region we use as classifier.

Other eligible regions include line segments, filled triangles, disks and any convex polygon.

### Why the VC dimension is infinite

For **every positive integer** $m$, fix $m$ distinct points on a circle. For **any labeling** of those fixed points, select the positive-labeled subset $P$ and use its **convex hull** $\operatorname{conv}(P)$ as $C$ (choose $C=\varnothing$, which is convex, when $P=\varnothing$).

This convex region contains all selected positive points **and no other sampled points on the circle**. Geometrically, distinct points on a circle are extreme: a point on the circle cannot be expressed as a convex combination of other distinct points of that same circle.

**Optional precise justification.** For points on a unit circle, let $q$ be a negative-labeled point and $p_i\ne q$ positive-labeled points. Their dot products satisfy $q\cdot p_i<1$. Thus every convex combination $z=\sum_i\lambda_i p_i$ with $\lambda_i\ge0$, $\sum_i\lambda_i=1$ has $q\cdot z<1$, while $q\cdot q=1$. Therefore $q\notin\operatorname{conv}(P)$.

Hence the family shatters a set of $m$ points for every $m$:

$$ \boxed{\forall m\ge1,\ \exists T\subseteq\mathbb R^2:\ |T|=m,\ T\text{ shattered};\qquad\operatorname{VCdim}(\mathcal H_{\rm conv})=\infty.} $$

### An impossible labeling is not a contradiction

Let three sample points be the vertices of a triangle and a fourth lie strictly inside. The labeling that gives the vertices 1 and the interior point 0 is **impossible**: convexity forces any region containing the three vertices to contain their entire triangle.

So that **particular four-point set is not shattered**. This does **not** contradict infinite VC dimension. The definition requires the **existence** of *some* shattered set of each size (for example, all points on a circle), **not** that all point configurations be shattered.

> [!IMPORTANT]
> **Personal conceptual breakthrough:** “We found one configuration that fails — why isn't VC dimension finite?” Because the failed configuration disproves that **specific set** is shattered, not that **every set of the same cardinality** fails. To prove an upper bound one must rule out **all arrangements** of the next cardinality.

---

## 6. Finite families, uniform convergence and ERM

### Finite family consistency: Example 6.7

If $\mathcal H$ shatters $d$ points, it must realize the $2^d$ different labelings on those points. A single hypothesis has only one labeling on the fixed set, so the family needs at least $2^d$ distinct hypotheses:

$$ 2^d\le|\mathcal H|\quad\Longrightarrow\quad\boxed{\operatorname{VCdim}(\mathcal H)\le\log_2|\mathcal H|}. $$

This is an **upper bound**, not an equality for every finite class.

### Theorem 6.3 (Week 1 §6) — what it guarantees

For a **fixed** hypothesis family $\mathcal H$ with finite VC dimension $d$, and a sufficiently large i.i.d. sample $S\sim\mathcal D^n$, the theorem guarantees, **with probability at least $1-\delta$ over the sampling of $S$**, that:

$$ \boxed{\forall h\in\mathcal H,\quad |R_{\mathcal D}(h)-\widehat R_S(h)|\le\varepsilon.} $$

This is **uniform convergence**, including for infinite hypothesis families of finite VC dimension. A sufficient sample bound has a dependence of order $1/\varepsilon^2$, linear/logarithmic terms in $d$, and $\log(1/\delta)$; refer to the **exact printed Theorem 6.3 in Week 1 §6** for its chosen universal-constant formulation. Its full proof is **omitted by the instructor**.

Let $\hat h^\star$ minimize empirical risk and $h^\star$ minimize true risk **within $\mathcal H$** (assuming a minimizer exists). On the uniform-convergence event,

$$ \boxed{R_{\mathcal D}(\hat h^\star)\le\widehat R_S(\hat h^\star)+\varepsilon\le\widehat R_S(h^\star)+\varepsilon\le R_{\mathcal D}(h^\star)+2\varepsilon.} $$

- **First and third comparisons:** uniform convergence, once for each of the two hypotheses.
- **Middle comparison:** definition of ERM, no extra $\varepsilon$.
- **Not a realizability theorem:** no need for a zero-risk hypothesis or a uniform distribution of $X$.

**Contrast to Theorem 5.2:** its $1/\varepsilon$ sufficient rate was for a *special realizable threshold model* under uniform $X$ on $[0,1]$. The general VC theorem makes broader assumptions and gives a different type of guarantee.

**Connection:** [[LESSON 3]] contains the finite-class uniform-convergence proof and the $2\varepsilon$ chain. [[LESSON 4]] contains the guided proofs of Theorems 5.1 and 5.2.

### Risk decomposition (Week 1 §7)

With Bayes risk $R_{\mathcal D}^\star$, best-in-class $h^\star$, and the actual selected $\hat h^\star$,

$$ R_{\mathcal D}(\hat h^\star)=\underbrace{R_{\mathcal D}^\star}_{\text{Bayes error}}+\underbrace{[R_{\mathcal D}(h^\star)-R_{\mathcal D}^\star]}_{\text{inductive bias}}+\underbrace{[R_{\mathcal D}(\hat h^\star)-R_{\mathcal D}(h^\star)]}_{\text{error within }\mathcal H}. $$

This is an **exact algebraic decomposition**. Uniform convergence can separately give a *probabilistic bound* on the final term; it does not alter the Bayes risk or automatically eliminate approximation bias. For the detailed explanation, see [[LESSON 3]] §7; this equation is also in **Week 1 §7**. Do not assert that the exact §7 wording is necessarily in the handwritten Lecture 4.

---

## 7. Mistakes and repairs from our actual study

| What originally caused difficulty | What resolves it |
|---|---|
| Assuming a *single* $h$ must simultaneously output 0 and 1 on one fixed point | For **each labeling**, a potentially **different** $h$ may be chosen |
| Treating an example with two points as proof for every pair | For an **upper bound**, start with **arbitrary** $x_1<x_2$ and prove contradiction symbolically |
| Assuming “VCdim $=d$” means **every** $d$-point set is shattered | It means **there exists at least one** shattered $d$-point set |
| Confusing “at most 2 blocks” with “VC upper bound 2” | On four points, at most two positive blocks imply two intervals suffice for **all labelings**, giving **VC lower bound 4** |
| Thinking “one 1 per interval” | A block of **several consecutive 1s** can be covered by **one** interval |
| Thinking $(1,1,1,0)$ impossible with one interval | Its three 1s form **one block**, so it needs one interval |
| Forgetting the interval includes its boundary | Use $a\le x_i\le b$ for membership; $[0.3,0.7]$ labels both endpoints 1 |
| Extending “count positive blocks” to every hypothesis class | Only use it after proving an appropriate **geometric property** of that family |
| Reading $\mathcal H_{\rm conv}$ as merely the set of geometric regions | The elements of $\mathcal H_{\rm conv}$ are functions **$h_C$**; $C$ determines each function's positive region |
| Reading $u,v$ as ordered scalar positions, or treating $t=0$ as definition of a singleton classifier | $u,v$ are arbitrary **vectors** in $C$; all $t\in[0,1]$ test convexity; choose $C=\{x_2\}$ to make a singleton region |
| Thinking an unshatterable triangle-with-interior set disproves infinite VC dimension | An upper-bound impossibility must hold for **every arrangement** of the chosen size; circle construction gives a shattered set of every size |

---

## 8. Exam-oriented active recall (without looking at the proofs)

**Understanding:**
- [ ] Explain *why* different hypotheses are allowed for different labelings of a fixed set.
- [ ] Distinguish the quantifiers in “at least one shattered set” and “no set of the next size is shattered”.
- [ ] Explain the hereditary property of shattering in your own words.
- [ ] Explain why $|\mathcal H|=\infty$ does **not** imply infinite VC dimension.
- [ ] Define $C$, $h_C$, $\mathcal H_{\rm conv}$, $u$, $v$ and $t$ without confusing their roles.
- [ ] Give one **shattered** and one **not shattered** 4-point arrangement for convex-set classifiers; explain why this is consistent with VCdim $=\infty$.

**Reconstruct on paper (no prompts):**
- [ ] Exercise Sheet 1, **1.4(a)**: VC dimension of thresholds over $\mathbb R$ — both bounds.
- [ ] Exercise Sheet 1, **1.4(b)**: VC dimension of intervals — all four 2-point labelings, and the generic 3-point obstruction.
- [ ] Exercise Sheet 1, **1.4(c)**: VC dimension of unions of $k$ intervals — prove the $2k$ lower bound and the $2k+1$ upper bound for general $k$ (not merely by extrapolating a few examples).
- [ ] Exercise Sheet 1, **1.4(f)**: infinite VC dimension of all convex subsets of $\mathbb R^2$; use points on a circle.
- [ ] Exercise Sheet 1, **1.4(h)**: prove $\operatorname{VCdim}(\mathcal H)\le\log_2|\mathcal H|$ for finite $\mathcal H$.
- [ ] Reproduce the three comparisons behind the ERM $2\varepsilon$ guarantee and identify what justifies each one.

**Extra examples confirmed in Week 1 §6, not yet fully proved here:** axis-aligned rectangles in $\mathbb R^2$ have VC dimension 4 (Exercise 1.4(d) extends to $2d$ in $\mathbb R^d$); triangles in $\mathbb R^2$ have VC dimension 7. Do not claim an independent proof of these examples from merely reading their stated results.

**Next study stage:** Continue current Lecture 5 after its handwritten notes are available; return to the checklist through selective retrieval and cumulative exercises, not by endlessly restarting Lecture 4.
