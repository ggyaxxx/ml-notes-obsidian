---
title: "Lecture 3 — Statistical Learning: Concentration, Uniform Convergence, ERM, and Risk Decomposition"
aliases:
  - "Lecture 3 - 02-10-2026"
  - "Statistical Learning - Generalization Bounds"
date: 2026-10-02
course: "Machine Learning — Sapienza"
academic_year: "2026/2027"
type: lecture-note
status: "Theory covered with guidance; independent exercises and delayed recall pending"
tags:
  - machine-learning
  - statistical-learning
  - concentration-inequalities
  - uniform-convergence
  - empirical-risk-minimization
  - generalization
  - risk-decomposition
---

# Lecture 3 — Statistical Learning

> [!abstract] Central question
> We observe a **finite random dataset** $S$, but ultimately care about a classifier's **true risk** on the unknown distribution $\mathcal D$. How can we trust empirical performance when selecting a classifier, and what prevents our selected classifier from being optimal?

**Conceptual path in the lecturer's handwritten Lecture 3:** true risk versus empirical risk → 0–1 loss and concentration → Hoeffding and sample complexity → union bound → finite hypothesis families and simultaneous guarantees → empirical risk minimization (ERM) → risk decomposition.

**Study status:** These ideas have been discussed and several derivations completed with assistance. *This does not yet count as independent, exam-ready mastery.* See [Active recall and pending exercises](#active-recall-and-pending-exercises).

## Sources and scope

| Source | Status | Used for |
|---|---|---|
| [**Lezione 3 — 02/10/2026**, handwritten notes](https://drive.google.com/file/d/1qhydXH9Npe42MbyerGesmRh4uUlpcTdx/view) | **2026/27 confirmed — lecture** | Authoritative lecture order, notation and final message on risk decomposition |
| [**Week 1: Statistical Learning**, version 2, 06/10/2026](https://drive.google.com/file/d/1sw45w97b56JPTQJEdyOz2qjMkfsVWIFS/view) | **2026/27 confirmed — handout** | §§2–4 (pp. 1–5): motivation, Hoeffding, union bound, finite families and ERM; §7 (pp. 7–8): risk decomposition |
| [**Exercise Sheet 1: Statistical Learning Theory**, 06/10/2026](https://drive.google.com/file/d/1DpZ45CXHWnegIKxVYsLDldAZ724l6fFV/view) | **2026/27 confirmed — exercises** | Exercise 1.1 (p. 1) is the main next practice exercise for model selection and a reused validation set |

**Source boundary:** The explanations about *set theory, De Morgan, the auxiliary symbol $q$, numerical toy examples, common notation errors, and the step-by-step inequality manipulation* are **tutor-added prerequisites/clarifications**, not quotations or separate requirements attributed to the lecturer. The full proof of the general Chernoff–Hoeffding inequality is **not developed** in the handwritten lecture or Week 1 §3; the handout states it and derives its applications. The later **realizable setting, threshold examples, shattering and VC dimension** appear in Week 1 §§5–6 and the subsequent lecture; they are not included here as if they had already been covered in Lecture 3.

---

## 1. What are we estimating?

Consider binary supervised classification with 0–1 loss. We assume:

- $\mathcal X$: input/feature space; $\mathcal Y$: label space (usually $\{0,1\}$ or $\{-1,+1\}$).
- $\mathcal D$: **unknown joint probability distribution** on $\mathcal X\times\mathcal Y$; it is not a dataset.
- $(X,Y)\sim\mathcal D$: a random labeled example; lower-case $(x_i,y_i)$ denotes an observed realization.
- $S=((x_1,y_1),\ldots,(x_n,y_n))$: a dataset consisting of $n$ independently and identically distributed (**i.i.d.**) observations from $\mathcal D$.
- $h:\mathcal X\to\mathcal Y$: a classifier, predicting $h(x)$ from an input $x$.
- $\ell(h(x),y)=\mathbf1\{h(x)\ne y\}$: the **0–1 loss**; it equals $1$ for a wrong prediction and $0$ for a correct one.

**True/expected risk:**

$$
R_{\mathcal D}(h)
=\mathbb E_{(X,Y)\sim\mathcal D}[\ell(h(X),Y)]
=\Pr_{(X,Y)\sim\mathcal D}(h(X)\ne Y).
$$

This is the probability that $h$ misclassifies a **new random example**. It depends on the unknown distribution and is generally inaccessible directly.

**Empirical risk:**

$$
\widehat R_S(h)
=\frac1n\sum_{i=1}^n\ell(h(x_i),y_i)
=\frac1n\sum_{i=1}^n\mathbf1\{h(x_i)\ne y_i\}.
$$

This is the **fraction of observed examples** misclassified by $h$.

> [!note] Notation used while studying
> The official 2026/27 Week 1 handout writes $\widehat R_S(h)$ for empirical risk. During tutoring we often wrote the shorter $R_S(h)$. These denote the same quantity here; **prefer $\widehat R_S$ when matching the official handout**. The hat on the risk estimate is unrelated to the hat marking an ERM-selected classifier.

**Three levels that must never be conflated:**

1. **One example** $(x_i,y_i)$: its loss for $h$ is a number, $0$ or $1$.
2. **One realized dataset** $S$ containing $n$ examples: $\widehat R_S(h)$ is their **average loss**, not a separate risk “per example.” Two classifiers can have different empirical risks on *exactly the same* dataset.
3. **Possible random datasets** $S\sim\mathcal D^n$: before observing $S$, the empirical risk is a random quantity, and we may ask how likely it is to approximate true risk.

The fact that all classifiers are tested on one common $S$ in each run does **not** mean their loss vectors, empirical risks, or failure events are identical.

> [!example] One shared dataset, different estimation outcomes (tutor-added)
> Suppose $\mathcal D$ is uniform over **four possible labeled examples** $a,b,c,d$, and each training dataset contains exactly **one** example ($n=1$). Three fixed classifiers $h_1,h_2,h_3$ each misclassify exactly one different example: respectively $a$, $b$, and $c$. All three have true risk $0.25$. Let $\varepsilon=0.50$.
>
> | One common realized dataset $S$ | $h_1$ estimate | $h_2$ estimate | $h_3$ estimate |
> |---|---|---|---|
> | $\{a\}$ | **Fails** | Succeeds | Succeeds |
> | $\{b\}$ | Succeeds | **Fails** | Succeeds |
> | $\{c\}$ | Succeeds | Succeeds | **Fails** |
> | $\{d\}$ | Succeeds | Succeeds | Succeeds |
>
> A misclassified singleton sample produces empirical risk $1$ and deviation $|1-0.25|=0.75>0.50$; a correctly classified singleton produces empirical risk $0$ and deviation $|0-0.25|=0.25\le0.50$. **In each row every classifier sees the same dataset.** The rows represent *alternative possible draws*, not separate datasets given to individual classifiers. Here individual estimation success has probability $0.75$, while simultaneous success has probability only $0.25$ (the draw $S=\{d\}$). This is a concrete counterexample to confusing individual and joint success probabilities.

### Why estimation is possible for a *fixed* classifier

Fix a classifier $h$ **before looking at the sample** and define random error indicators

$$
Z_i=\ell(h(X_i),Y_i)=\mathbf1\{h(X_i)\ne Y_i\},\qquad i=1,\ldots,n.
$$

Because the training examples are i.i.d. and the same fixed $h$ is applied to each one, the $Z_i$ are i.i.d. Bernoulli variables. In particular:

$$
Z_i\in\{0,1\},\qquad
\mathbb E[Z_i]=R_{\mathcal D}(h),\qquad
\frac1n\sum_{i=1}^n Z_i=\widehat R_S(h).
$$

Thus **estimating a classifier's true risk is estimating the mean of bounded i.i.d. random variables**. This is precisely where concentration inequalities apply.

### Motivation: why empirical perfection can be misleading (Week 1 §2)

The current-year handout gives an **overfitting example** on the continuous square $[-1,1]^2$: the true label is positive inside the smaller central square $Q_\Delta=[-\Delta,\Delta]^2$ and negative outside it. Consider a predictor that **memorizes the positive training points** and returns the negative label for any other input. Its empirical error can be zero, yet a fresh draw from a continuous distribution almost surely does not exactly reproduce a memorized point. The predictor consequently misclassifies nearly every new positive point. The positive region has probability

$$
\Pr(X\in Q_\Delta)=\frac{(2\Delta)^2}{(2)^2}=\Delta^2,
$$

so its true risk is $\Delta^2$ even though its training error is $0$. This is the **2026/27 Week 1 §2 handout's illustrative construction**, included here as supplementary motivation; it should not be mistaken for an additional handwritten Lecture 3 derivation.

> [!important] A fixed classifier is not the same as a data-selected classifier
> A hypothesis chosen *after inspecting $S$* may depend on the same observations used to evaluate it. We cannot simply take the fixed-$h$ Hoeffding guarantee and substitute this data-dependent choice. Later, a simultaneous guarantee over a **preselected family $\mathcal H$** solves this problem.

## 2. Precision, failure probability, and Hoeffding

For a **fixed** $h$, the lecturer asks: given a precision $\varepsilon\in(0,1)$ and an acceptable failure probability $\delta\in(0,1)$, how many i.i.d. samples suffice to guarantee

$$
\Pr_S\!\left(
\left|\widehat R_S(h)-R_{\mathcal D}(h)\right|\le\varepsilon
\right)\ge 1-\delta\ ?
$$

Read this carefully:

- $\varepsilon$ (*epsilon*) is a **risk-deviation tolerance**: how far the empirical estimate may be from the true risk.
- $\delta$ (*delta*) is a **desired upper bound on the probability that the tolerance is violated**. It is not necessarily the exact failure probability.
- $\Pr_S$ means the probability is taken over **random draws of the complete dataset** $S$, not over the predictions made by $h$ on an individual test point.
- $1-\delta$ is a **lower bound on the probability of a sufficiently accurate risk estimate**, not the prediction accuracy of the classifier.

For example, $\varepsilon=0.05$, $\delta=0.01$ means: with probability **at least 99% over the random dataset**, the empirical risk lies within $0.05$ of the true risk. It emphatically **does not** mean the classifier predicts 99% of future labels correctly.

### Chernoff–Hoeffding inequality

For i.i.d. random variables $Z_1,\ldots,Z_n\in[0,1]$ with common expectation $\mu$, the handout states the one-sided bounds

$$
\Pr\!\left(\frac1n\sum_i Z_i>\mu+\varepsilon\right)
\le e^{-2n\varepsilon^2},
$$

$$
\Pr\!\left(\frac1n\sum_i Z_i<\mu-\varepsilon\right)
\le e^{-2n\varepsilon^2}.
$$

A deviation larger than $\varepsilon$ in absolute value means **either** an excessive overestimate **or** an excessive underestimate. The two events form a union; applying the union bound gives the two-sided result (Week 1, Corollary 3.4):

$$
\Pr\!\left(\left|\frac1n\sum_i Z_i-\mu\right|>\varepsilon\right)
\le 2e^{-2n\varepsilon^2}.
$$

For our fixed classifier, substitute $\mu=R_{\mathcal D}(h)$ and $(1/n)\sum_i Z_i=\widehat R_S(h)$:

$$
\boxed{
\Pr_S\!\left(
\left|\widehat R_S(h)-R_{\mathcal D}(h)\right|>\varepsilon
\right)\le 2e^{-2n\varepsilon^2}
}
$$

**Interpretation:** the chance of an estimation deviation exceeding $\varepsilon$ is **at most** the expression on the right. It is generally **not equal** to that expression. As $n$ increases, the bound decreases exponentially for fixed $\varepsilon$.

### Sample complexity for one fixed classifier

To guarantee an acceptable failure probability $\delta$, it is **sufficient** to impose

$$
2e^{-2n\varepsilon^2}\le\delta.
$$

Here are **all** the algebraic steps:

$$
\begin{aligned}
2e^{-2n\varepsilon^2}&\le\delta\\
e^{-2n\varepsilon^2}&\le\frac\delta2
&&\text{(divide by 2)}\\
-2n\varepsilon^2&\le\ln\!\left(\frac\delta2\right)
&&\text{(natural log is increasing)}\\
2n\varepsilon^2&\ge-\ln\!\left(\frac\delta2\right)
&&\text{(multiply by -1: reverse inequality)}\\
2n\varepsilon^2&\ge\ln\!\left(\frac2\delta\right)
&&\text{(logarithm identity)}\\
n&\ge\frac{\ln(2/\delta)}{2\varepsilon^2}
&&\text{(divide by }2\varepsilon^2>0\text{).}
\end{aligned}
$$

Since $n$ is an integer, round the sufficient threshold **up**.

Consequences: halving $\varepsilon$ requires **four times** as many samples under this sufficient bound, with $\delta$ unchanged; demanding higher confidence (decreasing $\delta$) also increases the required sample count, only logarithmically in $1/\delta$.

> [!warning] Logical and algebraic checks
> - A sufficient bound on $n$ is **not** a claim that fewer samples can never work.
> - $a\le b$ is different from $a=b$; a probabilistic bound rarely gives an exact probability.
> - Multiplication or division by a **negative** number reverses an inequality.
> - The tolerance condition for **success** is $|\widehat R_S-R_{\mathcal D}|\le\varepsilon$; **failure** is the strict inequality $>\varepsilon$.

## 3. Set operations and De Morgan: the prerequisite for union bound

This section is a **tutor-added mathematics review** prompted by questions during study.

Given events $A$ and $B$:

| Expression | Read as | True when |
|---|---|---|
| $A\cup B$ | union: $A$ **OR** $B$ | At least one occurs, possibly both |
| $A\cap B$ | intersection: $A$ **AND** $B$ | Both occur |
| $A^c$ | complement: **NOT** $A$ | $A$ does not occur |
| $(A\cup B)^c$ | NOT ($A$ OR $B$) | Neither occurs |
| $(A\cap B)^c$ | NOT ($A$ AND $B$) | At least one does not occur |

The **laws of De Morgan** are

$$
\boxed{(A\cup B)^c=A^c\cap B^c},
\qquad
\boxed{(A\cap B)^c=A^c\cup B^c}.
$$

This is a logical identity, not a trick for interchanging $\cup$ and $\cap$ arbitrarily: **taking the complement** is what changes the operation.

Suppose $E_i$ is the event “classifier $h_i$ has a risk-estimation deviation larger than $\varepsilon$” and $G_i=E_i^c$ is the corresponding success event. Then

$$
\underbrace{G_1\cap G_2\cap G_3}_{\text{ALL estimates succeed}}
=
\underbrace{(E_1\cup E_2\cup E_3)^c}_{\text{NOT even one estimate fails}}.
$$

Taking probabilities and using $\Pr(A^c)=1-\Pr(A)$,

$$
\boxed{
\Pr_S(\text{all succeed})
=1-\Pr_S(\text{at least one fails}).
}
$$

**Why do we study failure events?** Because the union bound directly limits the probability that **at least one** unwanted event occurs. We never *want* a failure: we bound its probability in order to infer that all classifiers succeed simultaneously.

## 4. Union bound and uniform convergence for a finite family

The **union bound** (Week 1, Proposition 3.2) says that for any events $A_1,\ldots,A_k$,

$$
\boxed{\Pr\!\left(\bigcup_{i=1}^k A_i\right)
\le\sum_{i=1}^k\Pr(A_i).}
$$

For two events, this follows from

$$
\Pr(A\cup B)=\Pr(A)+\Pr(B)-\Pr(A\cap B)
\le\Pr(A)+\Pr(B).
$$

**No independence assumption is needed**; the events may overlap and be dependent. The sum is an **upper bound**, sometimes conservative. In particular, it is not generally the exact probability of the union.

### Why a family changes the question

ERM will eventually choose one classifier based on the observed $S$. To avoid unjustified use of fixed-classifier Hoeffding after this selection, we **commit in advance** to a finite class

$$
\mathcal H=\{h_1,\ldots,h_M\},\qquad M=|\mathcal H|.
$$

The relevant good event is

$$
\mathcal G
=\left\{S:\ \forall h\in\mathcal H,\
\left|\widehat R_S(h)-R_{\mathcal D}(h)\right|\le\varepsilon\right\}.
$$

This is a condition on a **single common realized dataset $S$**: every $h\in\mathcal H$ must give a sufficiently accurate estimate on that same $S$.

For each *fixed* $h$, define the bad event

$$
E_h=\{S:|\widehat R_S(h)-R_{\mathcal D}(h)|>\varepsilon\}.
$$

Then, by De Morgan, $\mathcal G^c=\bigcup_{h\in\mathcal H}E_h$. Apply the union bound, then the individual Hoeffding bound:

$$
\begin{aligned}
\Pr_S(\mathcal G^c)
&=\Pr_S\left(\bigcup_{h\in\mathcal H}E_h\right)\\
&\le\sum_{h\in\mathcal H}\Pr_S(E_h)\\
&\le\sum_{h\in\mathcal H}2e^{-2n\varepsilon^2}\\
&=2|\mathcal H|e^{-2n\varepsilon^2}.
\end{aligned}
$$

Therefore the **simultaneous success probability** obeys

$$
\boxed{
\Pr_S\left(\forall h\in\mathcal H:\
|\widehat R_S(h)-R_{\mathcal D}(h)|\le\varepsilon\right)
\ge 1-2|\mathcal H|e^{-2n\varepsilon^2}.
}
$$

To achieve a desired probability of simultaneous failure at most $\delta$, require

$$
2|\mathcal H|e^{-2n\varepsilon^2}\le\delta,
$$

which yields the **sufficient** sample size (Week 1, Theorem 4.1):

$$
\boxed{
 n\ge\frac{\ln(2|\mathcal H|/\delta)}{2\varepsilon^2}.
}
$$

If this condition holds, then $\Pr_S(\mathcal G)\ge1-\delta$.

### Three different symbols, three different jobs

An **auxiliary study notation**, not introduced as the lecturer's separate parameter, is

$$
q=2e^{-2n\varepsilon^2}.
$$

| Symbol | What it controls | What it **does not** mean |
|---|---|---|
| $\varepsilon$ | Magnitude of permitted risk-estimation deviation | Not itself a probability of estimation failure |
| $q$ | **Upper bound** on failure probability for **one fixed** $h$ | Not an exact probability; not a tolerance in risk units |
| $\delta$ | **Desired upper bound** on probability that **at least one** $h\in\mathcal H$ exceeds $\varepsilon$ | Not the true risk of a classifier; not necessarily exact |

Under the common Hoeffding bound, $\Pr_S(\bigcup_h E_h)\le|\mathcal H|q$. One sufficient design condition is $|\mathcal H|q\le\delta$.

**Worked illustration:** with $10$ hypotheses and a desired simultaneous failure bound $\delta=0.05$, the union-bound method is sufficient if

$$
10q\le0.05\quad\Longrightarrow\quad q\le0.005.
$$

This means each fixed-hypothesis failure probability must have a bound of **at most 0.5%** *under this sufficient method*, to guarantee that the chance of **any** failure is at most 5%. The actual failure probabilities and their union can be smaller. It does **not** mean that the risk-deviation tolerance $\varepsilon$ equals $0.05$.

> [!warning] Individual versus simultaneous quantifiers
> $\forall h\in\mathcal H,\ \Pr_S(G_h)\ge0.95$ does **not** imply $\Pr_S(\forall h\in\mathcal H:\ G_h)\ge0.95$.
>
> The first statement controls each classifier **separately**. The second controls **all classifiers on the same sample**. Their bad events can occur on different possible draws of the *same random dataset* $S$; there is never a need to assign a different dataset to each classifier within one run.

## 5. Empirical risk minimization (ERM)

The basic training rule is to choose the classifier with the lowest empirical risk:

$$
\boxed{
\hat h\in\arg\min_{h\in\mathcal H}\widehat R_S(h).
}
$$

The official Week 1 handout denotes this ERM output by **$\hat h^\star$**; during our walkthrough we used the shorter **$\hat h$**. Both names refer to the *data-selected empirical minimizer* in this note.

The **best classifier in the family under the true distribution**, instead, is

$$
\boxed{
 h^\star\in\arg\min_{h\in\mathcal H}R_{\mathcal D}(h).
}
$$

Do not confuse $h^\star$ (best *within* $\mathcal H$) with the unrestricted **Bayes predictor**, denoted $f^\star$ or identified by its Bayes risk $R_{\mathcal D}^\star$.

**`min` versus `argmin`:**

- $\min_{h\in\mathcal H}\widehat R_S(h)$ returns the **numerical minimum** empirical risk.
- $\arg\min_{h\in\mathcal H}\widehat R_S(h)$ returns the **set of classifiers attaining** it. If there is a tie, the ERM may choose any minimizer.

### The deterministic inequality supplied by ERM

Because $\hat h$ minimizes empirical risk, for **every** $h\in\mathcal H$ we have $\widehat R_S(\hat h)\le\widehat R_S(h)$. In particular, since $h^\star\in\mathcal H$,

$$
\boxed{\widehat R_S(\hat h)\le\widehat R_S(h^\star).}
$$

This is **deterministic** for any sample where ERM exists; it does **not** use Hoeffding. Likewise, from the definition of $h^\star$,

$$
R_{\mathcal D}(h^\star)\le R_{\mathcal D}(\hat h).
$$

These compare *different pairs of quantities*. In general, $\hat h\ne h^\star$ because the training sample can be unusually favorable to some classifiers.

### Why simultaneous guarantees solve the selection issue

Suppose a randomly drawn dataset happens to belong to the good set $\mathcal G$. By definition, **every** $h\in\mathcal H$ has risk deviation at most $\varepsilon$. This includes the value $\hat h$ selected *after inspecting that dataset*, simply because $\hat h\in\mathcal H$.

- **Conditional on $S\in\mathcal G$:** the deviation inequality for $\hat h$ holds with certainty.
- **Before seeing $S$:** $\Pr_S(S\in\mathcal G)\ge1-\delta$ under the sample-size condition.

The guarantee is therefore **high-probability over datasets**, not a deterministic statement about every possible sample.

## 6. Deriving the ERM $2\varepsilon$ guarantee

**Goal:** bound the true risk of the data-selected $\hat h$ using the true risk of the best-in-class $h^\star$.

**Available ingredients:** (i) simultaneous uniform convergence, and (ii) $\hat h$ minimizes empirical risk.

On a good dataset $S\in\mathcal G$, we have for all $h\in\mathcal H$:

$$
\left|\widehat R_S(h)-R_{\mathcal D}(h)\right|\le\varepsilon.
$$

**Step 1 — True risk of ERM to empirical risk of ERM.** The absolute-value inequality implies

$$
-\varepsilon\le\widehat R_S(\hat h)-R_{\mathcal D}(\hat h)\le\varepsilon.
$$

Using its **left-hand** part, $-\varepsilon\le\widehat R_S(\hat h)-R_{\mathcal D}(\hat h)$, we obtain

$$
R_{\mathcal D}(\hat h)\le\widehat R_S(\hat h)+\varepsilon.
$$

**Step 2 — Empirical risk of ERM to empirical risk of $h^\star$.** By the **definition of ERM**,

$$
\widehat R_S(\hat h)\le\widehat R_S(h^\star).
$$

Adding $\varepsilon$ on both sides preserves the inequality:

$$
\widehat R_S(\hat h)+\varepsilon
\le\widehat R_S(h^\star)+\varepsilon.
$$

**Step 3 — Empirical risk of $h^\star$ to its true risk.** Uniform convergence also applies to $h^\star$, hence

$$
\widehat R_S(h^\star)\le R_{\mathcal D}(h^\star)+\varepsilon.
$$

But the previous line contains **$\widehat R_S(h^\star)+\varepsilon$**, so we **must add $\varepsilon$ to both sides**, not replace it without carrying the extra term:

$$
\widehat R_S(h^\star)+\varepsilon
\le R_{\mathcal D}(h^\star)+\varepsilon+\varepsilon
=R_{\mathcal D}(h^\star)+2\varepsilon.
$$

**Step 4 — Combine by transitivity.** We now have exactly matching intermediate terms:

$$
\boxed{
\begin{aligned}
R_{\mathcal D}(\hat h)
&\le\widehat R_S(\hat h)+\varepsilon
&&\text{(uniform convergence)}\\
&\le\widehat R_S(h^\star)+\varepsilon
&&\text{(ERM definition)}\\
&\le R_{\mathcal D}(h^\star)+2\varepsilon
&&\text{(uniform convergence again).}
\end{aligned}}
$$

Therefore

$$
\boxed{
0\le R_{\mathcal D}(\hat h)-R_{\mathcal D}(h^\star)
\le2\varepsilon.
}
$$

The left inequality follows because $h^\star$ is the best classifier in $\mathcal H$; the right inequality follows from the three steps above. By the union-bound estimate, the right inequality holds with probability **at least $1-\delta$** if $n\ge\ln(2|\mathcal H|/\delta)/(2\varepsilon^2)$.

**Why exactly $2\varepsilon$?** There are **two** switches between empirical and true risk—one for $\hat h$, one for $h^\star$. Each contributes up to $\varepsilon$. The middle comparison is deterministic and introduces **no new** $\varepsilon$.

> [!example] The algebra error to watch for
> From $A\le B+c$ and $B\le C+d$, we may infer $A\le C+c+d$, **not** $A\le C+d$. The first $+c$ cannot disappear.
>
> Concrete example: $A\le B+4$ and $B\le C+7$ imply $A\le C+11$.

**Alternative form in Week 1 §4:** The lecturer writes the *difference* $R_{\mathcal D}(\hat h)-R_{\mathcal D}(h^\star)$ as three added differences, inserting and subtracting empirical risks. This is algebraically equivalent to the inequality chain above. The chain is used here because it made each mathematical justification easier to track during study.

## 7. Risk decomposition: why can an ERM still be imperfect?

Even if ERM is near-optimal **within $\mathcal H$**, the family itself may be incapable of representing a near-Bayes-optimal classifier. That is the new question addressed at the end of Lecture 3 and Week 1 §7.

Let $R_{\mathcal D}^\star$ denote the **Bayes risk**: the minimum achievable expected risk among all allowable predictors under the stated input space, loss, and distribution. Keep $h^\star$ for the best predictor **restricted to $\mathcal H$**.

Insert and subtract two expressions that cancel:

$$
\begin{aligned}
R_{\mathcal D}(\hat h)
&=R_{\mathcal D}(\hat h)
 +R_{\mathcal D}(h^\star)-R_{\mathcal D}(h^\star)
 +R_{\mathcal D}^\star-R_{\mathcal D}^\star\\[3pt]
&=\underbrace{R_{\mathcal D}^\star}_{\text{Bayes error}}
 +\underbrace{\left(R_{\mathcal D}(h^\star)-R_{\mathcal D}^\star\right)}_{\text{Inductive bias}}
 +\underbrace{\left(R_{\mathcal D}(\hat h)-R_{\mathcal D}(h^\star)\right)}_{\text{Error within }\mathcal H}.
\end{aligned}
$$

**This equation is an exact algebraic identity.** No concentration theorem, probability estimate, or random-sample assumption is required simply to add and subtract equal terms.

### Meaning of the three terms

| Term | Exact quantity | Main reason it exists | Effect of increasing $n$ with fixed $\mathcal H$ |
|---|---|---|---|
| **Bayes error** | $R_{\mathcal D}^\star$ | Intrinsic uncertainty/overlap under the given features, loss, and distribution | Not changed by collecting more examples |
| **Inductive bias** | $R_{\mathcal D}(h^\star)-R_{\mathcal D}^\star$ | Restricting ourselves to a possibly insufficiently expressive $\mathcal H$ | Not changed merely by collecting more examples |
| **Error within $\mathcal H$** | $R_{\mathcal D}(\hat h)-R_{\mathcal D}(h^\star)$ | ERM selects from finite data, not direct knowledge of true risk | Its **high-probability upper bound** can be improved with more examples |

All three terms are nonnegative under these definitions. The handwritten lecture also uses the phrase **“variance error”** for the last term; the Week 1 handout says **“error within $\mathcal H$.”** Here it is a difference between two true risks; **do not automatically interpret it as the variance of a random variable** or as the full classical bias–variance decomposition in regression.

**Numerical illustration** (tutor-added):

$$
R_{\mathcal D}^\star=0.05,\quad
R_{\mathcal D}(h^\star)=0.20,\quad
R_{\mathcal D}(\hat h)=0.24.
$$

Therefore

$$
\underbrace{0.24}_{\text{ERM true risk}}
=\underbrace{0.05}_{\text{Bayes}}
+\underbrace{(0.20-0.05)}_{\text{bias }0.15}
+\underbrace{(0.24-0.20)}_{\text{within-class }0.04}.
$$

**Do not call $0.05$ the “bias risk”: it is the *Bayes risk*.** The inductive bias is $0.15$.

### Decomposition versus generalization guarantee

The identity above always holds. Under uniform convergence we may **additionally** upper-bound the last term, with high probability:

$$
R_{\mathcal D}(\hat h)-R_{\mathcal D}(h^\star)\le2\varepsilon.
$$

Thus, with probability at least $1-\delta$ under the finite-family sample-size assumption,

$$
\boxed{
R_{\mathcal D}(\hat h)
\le R_{\mathcal D}^\star
+\big(R_{\mathcal D}(h^\star)-R_{\mathcal D}^\star\big)
+2\varepsilon.
}
$$

- **Exact decomposition:** explains *where* the risk comes from.
- **High-probability generalization bound:** controls the **third** component, not automatically Bayes error or inductive bias.

### The choice of hypothesis family matters

A more expressive family may reduce the **inductive bias** because it contains better candidates. More formally, if $\mathcal H_1\subseteq\mathcal H_2$, then the minimum true risk over $\mathcal H_2$ cannot exceed that over $\mathcal H_1$. This does **not** promise a strict improvement in every problem.

At the same time, our **finite-family sufficient bound** contains $\ln|\mathcal H|$: for a fixed $n$ and $\delta$, a larger family leads to a weaker worst-case uniform-convergence bound (or requires more examples to certify the same $\varepsilon$). It does **not** prove that a bigger family necessarily has worse realized test performance.

A curved decision boundary, for example, might be poorly approximated by linear classifiers. But curvature **alone** does not rigorously prove a strictly positive bias; we must also consider where $\mathcal D$ places probability mass.

> [!important] The take-home message
> **More samples can control the cost of selecting the classifier from data. Choosing an appropriate family controls the cost of restricting which classifiers are even available.** These are different problems.

---

## 8. Proof map (for fast reconstruction without notes)

1. **One fixed $h$:** define bounded i.i.d. loss variables $Z_i$ with mean $R_{\mathcal D}(h)$.
2. **Hoeffding:** $\Pr_S(|\widehat R_S(h)-R_{\mathcal D}(h)|>\varepsilon)\le2e^{-2n\varepsilon^2}$.
3. **Finite family:** define one failure event $E_h$ per $h\in\mathcal H$; use $\Pr_S(\bigcup_hE_h)\le\sum_h\Pr_S(E_h)$.
4. **De Morgan/complement:** $\Pr_S(\text{all succeed})\ge1-2|\mathcal H|e^{-2n\varepsilon^2}$.
5. **Sample complexity:** force $2|\mathcal H|e^{-2n\varepsilon^2}\le\delta$ to get $n\ge\ln(2|\mathcal H|/\delta)/(2\varepsilon^2)$.
6. **ERM:** $\widehat R_S(\hat h)\le\widehat R_S(h^\star)$ by definition.
7. **Two uses of uniform convergence:** $R_{\mathcal D}(\hat h)\le\widehat R_S(\hat h)+\varepsilon\le\widehat R_S(h^\star)+\varepsilon\le R_{\mathcal D}(h^\star)+2\varepsilon$.
8. **Risk decomposition:** add and subtract $R_{\mathcal D}^\star$ and $R_{\mathcal D}(h^\star)$ to separate Bayes error, inductive bias, and within-family selection error.

## 9. Active recall and pending exercises

> [!warning] Not yet marked consolidated
> Reading this file or following a guided derivation is **not** evidence of autonomous exam performance. Attempt these without consulting the preceding sections; use paper for the proofs.

### Short conceptual checks

- [ ] Explain what probability is measured by $\Pr_S$ and why it is not prediction accuracy.
- [ ] Distinguish a realized example, a realized dataset, and alternative random draws of a dataset.
- [ ] Define $\varepsilon$, $q$, and $\delta$ without confusing tolerance with failure probability.
- [ ] Explain why the **union of failures** is the complement of the **intersection of successes**.
- [ ] Explain why individual high-probability estimates do not automatically imply an equally high simultaneous guarantee.
- [ ] Explain why Hoeffding for fixed $h$ cannot simply be applied to $\hat h$ selected on the same $S$.
- [ ] Explain why $h^\star$ can have lower true risk than $\hat h$ but higher empirical risk.
- [ ] Explain why the middle ERM step adds **no** $\varepsilon$ in the $2\varepsilon$ proof.
- [ ] Explain why a larger $n$ need not monotonically improve the actual ERM selected on every sample.
- [ ] Distinguish the **exact** risk-decomposition identity from the **probabilistic** generalization inequality.

### Paper exercises — *attempt before viewing a solution*

1. **Reconstruct the finite-family bound from scratch.** Starting with the two-sided Hoeffding bound for a fixed $h$, derive the uniform-convergence condition for a finite $\mathcal H$ and solve the required-sample inequality for $n$. Explicitly justify each inequality and each quantifier.
2. **Prove the $2\varepsilon$ theorem twice.** First use the chain of inequalities; then attempt the three-difference decomposition used in **Week 1 §4**. Explain where each $\varepsilon$ arises.
3. **Study-specific algebra check.** From $A\le B+4$ and $B\le C+7$, give the strongest bound on $A$ obtainable by transitivity alone. Explain why simply discarding $+4$ is invalid.
4. **Source exercise: [Exercise Sheet 1, Exercise 1.1, p. 1](https://drive.google.com/file/d/1DpZ45CXHWnegIKxVYsLDldAZ724l6fFV/view).** Work on selecting one of $K$ models by their error on a **fresh validation sample**. Derive the stated high-probability bound and then do the numerical evaluation. Note why the candidate models must be fixed independently of that validation sample for this standard argument.
5. **Conceptual trade-off, pending from the lesson.** If $\mathcal H_1$ has 5 classifiers and a larger, more expressive $\mathcal H_2$ has 10,000, what may happen to inductive bias and to the finite-family generalization guarantee at fixed $n$? Clearly distinguish a worst-case upper bound from actual test risk.

**Next topic, not part of the mastered Lecture 3 content:** Lecture 4's realizable setting, thresholds, shattering, and VC dimension (Week 1 §§5–6; Exercise Sheet 1, Exercises 1.2–1.4). Do not mark these as completed just because they appear in the same Week 1 handout.

## 10. Personal study review — what caused mistakes

This list is **derived from our tutoring sessions**, not an attribution to the lecturer.

| Difficulty encountered | Precision check for future work |
|---|---|
| Mixing up $\varepsilon$, $q$, and $\delta$ | Before using a symbol, say whether it represents a **risk-distance threshold**, an **individual failure-probability bound**, or a **simultaneous failure-probability target**. |
| Treating a single example as interchangeable with a dataset | Write $S=((x_1,y_1),\ldots,(x_n,y_n))$ and distinguish $n=1$ as a special case. |
| Confusing prediction errors with risk-estimation failure | A classifier misclassifies an example when its **loss** is 1; an **estimation failure** occurs when $|\widehat R_S(h)-R_{\mathcal D}(h)|>\varepsilon$. |
| Assuming the same dataset forces the same outcome for different hypotheses | Same **inputs and labels**; potentially different predictions and loss vectors. |
| Union/intersection reversal | **Union of bad events = at least one fails. Complement = none fail = intersection of good events.** |
| Assuming a bound equals the true probability | Say **at most** $\delta$, **at least** $1-\delta$. Do not assert equality. |
| Losing the direction of a comparison | Ask: *Which risk is minimized, and over which set?* ERM minimizes empirical risk; $h^\star$ minimizes true risk **within $\mathcal H$**. |
| Sign errors in inequalities | If multiplying/dividing by negative numbers, **reverse** $\le/\ge$. |
| Dropping a term during transitivity | If $A\le B+\varepsilon$ and $B\le C+\varepsilon$, add the second $\varepsilon$: $A\le C+2\varepsilon$. |
| Using an imprecise name for a theorem | $\Pr(A^c)=1-\Pr(A)$ is the **complement rule**, not the law of total probability. |
| Mixing Bayes risk with inductive bias | Bayes risk is $R_{\mathcal D}^\star$; inductive bias is **best-in-class risk minus Bayes risk**. |

> [!tip] Method for every new proof
> Before manipulating equations, write **(a) what is given, (b) the target inequality, and (c) the single definition or theorem authorizing the next step**. After each step, check that no additive term or inequality sign has silently changed.
