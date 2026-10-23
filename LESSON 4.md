---
title: "Lecture 4 — Realizability and ERM: Finite-Class Generalization (Theorem 5.1)"
aliases:
  - "Lecture 4 - Realizable ERM"
  - "Theorem 5.1 - Finite Hypothesis Class"
date: 2026-10-23
course: "Machine Learning — Sapienza"
academic_year: "2026/2027"
type: lecture-note
status: "In progress — Theorem 5.1 studied; rest of lecture pending"
tags:
  - machine-learning
  - statistical-learning
  - realizability
  - empirical-risk-minimization
  - generalization
  - union-bound
  - sample-complexity
---

# Lecture 4 — Realizability and ERM

> [!abstract] Central question
> **If the best classifier in a finite hypothesis family has zero true risk, why can ERM still select an imperfect classifier — and how many training examples ensure that ERM's true risk is at most $\varepsilon$ with probability at least $1-\delta$?**

**Scope / study status:** This note records **Theorem 5.1 and its complete proof**, including the conceptual obstacles encountered while studying it. It is **not** a claim that the rest of Lecture 4 (e.g. shattering or VC dimension) has already been studied. Continue this file as the lecture progresses.

**Connections:** [[LESSON 2]] (true/empirical risk and overfitting) · [[LESSON 3]] (concentration, union bound, ERM, finite families).

**Source boundary:** The theorem statement below is the one transcribed during our study discussion. The analogies, numerical examples, common mistakes, and step-by-step explanations are **study clarifications**, not additional claims about the lecturer's notes.

---

## 1. Theorem 5.1 — Statement

Let $\mathcal H$ be a **fixed, finite** family of classifiers, and assume **realizability**. For precision $\varepsilon\in(0,1)$, failure probability $\delta\in(0,1)$ and an i.i.d. training sample

$$
S=((x_1,y_1),\ldots,(x_n,y_n))\sim\mathcal D^n,
$$

if

$$
\boxed{n\geq\frac{1}{\varepsilon}\log\left(\frac{|\mathcal H|}{\delta}\right),}
$$

then an empirical risk minimizer $\hat h_S$ satisfies

$$
\boxed{\Pr_{S\sim\mathcal D^n}\left(R_{\mathcal D}(\hat h_S)\leq\varepsilon\right)\geq1-\delta.}
$$

Here $\log$ is the natural logarithm. The probability is **over the random draw of the entire training set**. The chosen $\hat h_S$ can change as $S$ changes.

**In words:** With sufficiently many i.i.d. observations, the classifier selected by ERM has **true risk at most $\varepsilon$**, except with probability at most $\delta$.

- $\varepsilon$: tolerated **true error rate** (precision).
- $\delta$: tolerated **probability that the guarantee fails** (failure probability).
- $1-\delta$: confidence in the guarantee.
- $|\mathcal H|$: number of candidate classifiers; $\mathcal H$ must be finite for this bound.
- $n$: number of independent training observations.

> [!important] What the theorem does NOT say
> It does not claim that the ERM classifier is identical to the best-in-class classifier; it does not claim $R_{\mathcal D}(\hat h_S)=0$; it does not claim $R_S(h)=R_{\mathcal D}(h)$ for a particular sample.

---

## 2. Setup: distribution, sample, loss and risks

We consider classification with **0–1 loss**:

$$
\ell(h(x),y)=\mathbf 1\{h(x)\neq y\}.
$$

For a **fixed** classifier $h$:

**True risk** (error probability for one new draw from the underlying distribution):

$$
R_{\mathcal D}(h)
=\mathbb E_{(X,Y)\sim\mathcal D}[\ell(h(X),Y)]
=\Pr_{(X,Y)\sim\mathcal D}(h(X)\neq Y).
$$

**Empirical risk** (observed fraction of errors in a particular sample):

$$
\widehat R_S(h)
=\frac1n\sum_{i=1}^n\mathbf 1\{h(x_i)\neq y_i\}.
$$

We sometimes write $R_S(h)$ instead of $\widehat R_S(h)$; they mean the **same empirical quantity** in our discussions.

**Precision check:** $\mathcal D$ is a **probability distribution**, not a finite collection of examples. Thus “$S$ is a subset of $\mathcal D$” is an intuitive shorthand but not literally the mathematical relationship: **$S$ is sampled from $\mathcal D$**.

### Best in class versus ERM

$$
h^\star\in\arg\min_{h\in\mathcal H}R_{\mathcal D}(h),
\qquad
\hat h_S\in\arg\min_{h\in\mathcal H}\widehat R_S(h).
$$

They minimize **different functions**, so they may select **different classifiers**.

Because $h^\star$ minimizes true risk **within $\mathcal H$**, we always have

$$
R_{\mathcal D}(\hat h_S)\geq R_{\mathcal D}(h^\star).
$$

This fact alone does **not** give an upper bound on ERM's true risk: it is a **lower** bound.

---

## 3. The crucial assumption: realizability

> [!definition] Realizability
> We assume, **before sampling**, that there exists **at least one** classifier $h^\star\in\mathcal H$ satisfying
>
> $$
> \boxed{R_{\mathcal D}(h^\star)=0.}
> $$
>
> The assumption guarantees its **existence**, **not our knowledge of its identity**.

The student or algorithm generally does **not** know $\mathcal D$, and ERM is **not handed $h^\star$**. The theorem's proof is allowed to assume $h^\star$ exists because *realizability is one of the theorem's hypotheses*. It may be uncheckable in practice.

A zero true risk means that $h^\star$ is correct for a random $(X,Y)\sim\mathcal D$ **with probability 1** (almost surely). It does not necessarily assert correctness on points with zero probability under $\mathcal D$.

Since the training observations are drawn from that same $\mathcal D$, with probability 1,

$$
\widehat R_S(h^\star)=0.
$$

Now use **only the definition of ERM**:

$$
\widehat R_S(\hat h_S)
=\min_{h\in\mathcal H}\widehat R_S(h)
\leq\widehat R_S(h^\star)=0.
$$

Empirical risk is nonnegative, hence

$$
\boxed{\widehat R_S(\hat h_S)=0\quad\text{almost surely}.}
$$

> [!important] Explicit conclusion — why ERM has zero empirical risk
> **Under the realizability assumption, we can say with probability 1 (not merely with confidence $1-\delta$) that the empirical risk of an exact ERM classifier is zero.** We know *a priori* that some $h^\star\in\mathcal H$ has zero true risk, so, for an i.i.d. sample $S$ drawn from $\mathcal D$, this same classifier has zero empirical risk almost surely. Because ERM minimizes empirical risk over $\mathcal H$, it must attain that available minimum of zero, even if it selects a different classifier. **This conclusion does not require the sample-size bound of Theorem 5.1**; that bound is needed to control ERM's *true* risk.

**This does not mean that ERM chooses $h^\star$.** It means that a zero empirical-risk candidate is available, so an **exact** empirical-risk minimizer cannot choose a candidate with positive empirical risk.

> [!tip] The key insight that unlocked the proof
> **We know a perfect classifier exists, but not which one it is.** Its existence guarantees that the smallest training error available to ERM is zero. What remains unknown is the **true risk** of the zero-training-error classifier that ERM actually picks.

### Concrete example: two distinct perfect-on-sample classifiers

| Classifier | True risk $R_{\mathcal D}$ | Empirical risk $\widehat R_S$ |
|---|---:|---:|
| $h^\star$ (best in class) | $0$ | $0$ |
| $h_1$ | $0.20$ | $0$ |
| $h_2$ | $0.10$ | $0.05$ |
| $h_3$ | $0.30$ | $0.15$ |

For this **possible** sample, ERM can choose $h^\star$ **or** $h_1$ (a tie). It cannot choose $h_2$ or $h_3$ as an **exact ERM**, because their empirical risk is higher than the minimum zero.

$$
\boxed{\widehat R_S(\hat h_S)=0\ \nRightarrow\ R_{\mathcal D}(\hat h_S)=0.}
$$

Under realizability, the difficulty is **not** whether ERM achieves zero *training* error; the difficulty is whether **a bad classifier can also achieve zero training error and be selected**.

---

## 4. Why true risk is used to compute a probability about the sample

This was one of the main conceptual sticking points.

Suppose a **fixed** classifier $h$ has true risk

$$
R_{\mathcal D}(h)=0.2.
$$

By definition, for **one newly sampled example**,

$$
\Pr(h(X)\neq Y)=0.2,
\qquad
\Pr(h(X)=Y)=0.8.
$$

For $n=20$ independently sampled examples, the probability that **all twenty** are classified correctly is

$$
\Pr_{S\sim\mathcal D^{20}}\left(\widehat R_S(h)=0\right)
=0.8^{20}
\approx 0.01153
=1.15\%.
$$

**Why is using $0.2$ legitimate?** Because every observation in $S$ is drawn from **the same distribution** $\mathcal D$ used to define true risk. The factor $0.8$ is the chance of a **correct prediction on one random draw**, not a claim that $80\%$ of **this particular** sample must be correct.

A sample of 20 might contain 0, 2, 4, or 8 mistakes. Its empirical risk could be $0$, $0.1$, $0.2$, or $0.4$. The true risk remains $0.2$ for that fixed classifier.

In particular,

$$
R_{\mathcal D}(h)=0.2
\quad\not\implies\quad
\widehat R_S(h)=0.2.
$$

The correct statement for a **fixed** $h$ is

$$
\boxed{\mathbb E_{S\sim\mathcal D^n}[\widehat R_S(h)]
=R_{\mathcal D}(h).}
$$

This is an **equality of an expected sample average and the true risk**, **not** an equality for each realized training set. Moreover, one cannot blindly apply the fixed-$h$ argument to the sample-dependent $\hat h_S$.

**Analogy:** A coin lands heads with probability $0.8$. Twenty independent tosses need not contain exactly sixteen heads, but the probability of twenty heads is $0.8^{20}$.

> [!warning] Boundary of this reasoning
> If training examples were drawn from some different distribution $\mathcal Q$, the single-draw error probability on $S$ would be $R_{\mathcal Q}(h)$, **not necessarily** $R_{\mathcal D}(h)$. Independence is also essential for multiplying the probabilities across the $n$ draws.

---

## 5. The bad classifiers and the events $E_h$

Fix the accuracy tolerance $\varepsilon$. Define the set of **bad hypotheses**:

$$
\boxed{\mathcal H_B
=\{h\in\mathcal H:R_{\mathcal D}(h)>\varepsilon\}.}
$$

These are hypotheses that would **violate the theorem's desired conclusion** if ERM selected them.

For each **fixed** $h\in\mathcal H_B$, define an event in the space of possible training sets:

$$
\boxed{E_h
=\{S\sim\mathcal D^n:\widehat R_S(h)=0\}.}
$$

Read this as:

> “The **bad** classifier $h$ happens to make **zero errors on the sampled training set**.”

The event is superficially “positive” (perfect training accuracy), but **dangerous** because $h$ actually has large true risk.

Since $h\in\mathcal H_B$,

$$
R_{\mathcal D}(h)>\varepsilon
\quad\Longrightarrow\quad
1-R_{\mathcal D}(h)<1-\varepsilon.
$$

**$1-\varepsilon$ is not the empirical risk.** It is an **upper bound on the probability that one observation is classified correctly** by this bad hypothesis.

Using i.i.d. sampling,

$$
\begin{aligned}
\Pr(E_h)
&=\Pr(\text{$h$ is correct on all $n$ draws})\\
&=(1-R_{\mathcal D}(h))^n\\
&\leq(1-\varepsilon)^n\\
&\leq e^{-\varepsilon n}.
\end{aligned}
$$

The last step uses $1-t\leq e^{-t}$ for $t\geq0$.

**Crucial quantifier:** This inequality holds **separately for every fixed bad classifier**. Raising a probability to $n$ means “correct on all **$n$ independent samples**”; it does **not** mean “correct for all the classifiers in $\mathcal H_B$”.

---

## 6. Why we take a union, not an intersection

The **failure** we care about is that **at least one bad classifier** looks perfect on the training set, giving ERM an opportunity to select it.

Thus define

$$
E=\bigcup_{h\in\mathcal H_B}E_h.
$$

**Union:** $E_{h_1}\cup E_{h_2}\cup\cdots$ means **at least one** bad classifier gets zero training error.

**Intersection:** $E_{h_1}\cap E_{h_2}\cap\cdots$ would mean **all** bad classifiers simultaneously get zero training error — far stronger than needed, and not the dangerous condition we need to exclude.

**Complement:** $E^c$ means **none** of the bad classifiers has zero training error:

$$
E^c
=\bigcap_{h\in\mathcal H_B} E_h^c.
$$

On this complement event, because realizability guarantees ERM has zero training error, **ERM cannot select any classifier in $\mathcal H_B$**.

Apply the **union bound** (no independence assumption among the $E_h$ is required):

$$
\begin{aligned}
\Pr(E)
&=\Pr\left(\bigcup_{h\in\mathcal H_B}E_h\right)\\
&\leq\sum_{h\in\mathcal H_B}\Pr(E_h)\\
&\leq|\mathcal H_B|e^{-\varepsilon n}\\
&\leq|\mathcal H|e^{-\varepsilon n}.
\end{aligned}
$$

**Why replace $|\mathcal H_B|$ by $|\mathcal H|$?** Simply because $\mathcal H_B\subseteq\mathcal H$, so $|\mathcal H_B|\leq|\mathcal H|$. We may not know how many hypotheses are actually bad, but we know they cannot outnumber the entire finite class.

The events $E_h$ share **the same sampled dataset** and can be dependent. The union bound still holds.

---

## 7. Full proof map: from realizability to the bound

**Given:** fixed finite $\mathcal H$; realizability; i.i.d. $S\sim\mathcal D^n$; $0<\varepsilon,\delta<1$.

**Goal:**

$$
\Pr\big(R_{\mathcal D}(\hat h_S)>\varepsilon\big)\leq\delta.
$$

**Step A — realizability gives a perfect candidate.** There exists $h^\star\in\mathcal H$ with $R_{\mathcal D}(h^\star)=0$, hence $\widehat R_S(h^\star)=0$ almost surely.

**Step B — exact ERM also has zero sample risk.**

$$
\widehat R_S(\hat h_S)=0\qquad\text{almost surely}.
$$

**Step C — a bad ERM selection would imply a bad but sample-perfect classifier.**

$$
\left\{S:R_{\mathcal D}(\hat h_S)>\varepsilon\right\}
\subseteq
\bigcup_{h\in\mathcal H_B}\{S:\widehat R_S(h)=0\},
$$

apart from a probability-zero exception under realizability.

This is the **logical bridge** from “the ERM selected a bad hypothesis” to the union of fixed-hypothesis events. We **do not** pretend that $\hat h_S$ was fixed before observing $S$.

**Step D — bound the dangerous union.**

$$
\Pr\big(R_{\mathcal D}(\hat h_S)>\varepsilon\big)
\leq|\mathcal H|e^{-\varepsilon n}.
$$

**Step E — choose $n$ so that the failure probability is at most $\delta$.**

$$
\begin{aligned}
|\mathcal H|e^{-\varepsilon n}&\leq\delta\\
e^{-\varepsilon n}&\leq\frac{\delta}{|\mathcal H|}\\
-\varepsilon n&\leq\log\delta-\log|\mathcal H|\\
\varepsilon n&\geq\log|\mathcal H|-\log\delta\\
\boxed{n&\geq\frac1\varepsilon\log\left(\frac{|\mathcal H|}{\delta}\right).}
\end{aligned}
$$

The sign reverses when multiplying by $-1$.

Consequently,

$$
\Pr\big(R_{\mathcal D}(\hat h_S)>\varepsilon\big)\leq\delta
\quad\Longleftrightarrow\quad
\boxed{\Pr\big(R_{\mathcal D}(\hat h_S)\leq\varepsilon\big)\geq1-\delta.}
$$

The final equivalence uses the **complement rule**.

> [!success] One-sentence proof
> Under realizability, ERM always fits the training set perfectly; for it to have true risk above $\varepsilon$, **some fixed bad hypothesis must also fit that random training set perfectly**; the union bound shows that this becomes improbable for sufficiently large $n$.

---

## 8. What specifically caused confusion in our discussion

| Initial thought / question | Resolution |
|---|---|
| “If the best-in-class risk is zero, shouldn't the true risk of ERM also be zero?” | No. They optimize different risks and may select different hypotheses. Only $R_{\mathcal D}(h^\star)=0$ is assumed. |
| “Maybe ERM chooses a classifier with small but nonzero training risk.” | **Not under realizability and exact ERM.** A hypothesis with zero true risk gives zero training risk almost surely, so the minimum empirical risk is already zero. |
| “Are we choosing $h^\star$ because we already know which classifier is perfect?” | No. Realizability tells the **theorist** that such a hypothesis **exists**, not the learner which one it is. |
| “Are we equating $R_{\mathcal D}(h)$ with $\widehat R_S(h)$?” | No. $R_{\mathcal D}(h)$ gives the **single-draw error probability**; $\widehat R_S(h)$ is a random sample average. |
| “Why use $0.8^{20}$ if true risk $=0.2$ is defined on $D$?” | Each sample observation is itself independently drawn from $D$, hence each is correct with probability $0.8$ for that fixed $h$. |
| “Does $(1-\varepsilon)^n$ already cover all bad classifiers?” | No. The exponent $n$ refers to **$n$ observations for one fixed bad classifier**. Multiple bad classifiers enter through the union bound. |
| “Why union rather than complement/intersection?” | We first bound the **dangerous union**: at least one bad classifier looks perfect. Its complement is the **desired** event: none of the bad ones looks perfect. |
| “Why replace $|\mathcal H_B|$ with $|\mathcal H|$?” | Because $\mathcal H_B\subseteq\mathcal H$ and therefore $|\mathcal H_B|\leq|\mathcal H|$. |
| “Why is the last term $\leq\delta$?” | The assumed lower bound on $n$ is **chosen precisely to make** $|\mathcal H|e^{-\varepsilon n}\leq\delta$. |
| “Does probability $1-\delta$ describe the accuracy on each example?” | No. It describes **how likely a random training sample is to produce an ERM whose true error rate is at most $\varepsilon$**. |

### A useful distinction: guarantee versus observed behavior

Even with large $n$, a **particular** training sample might allow a bad classifier to get zero empirical error. The theorem does not ban that possibility. It puts a bound on its **probability over draws of $S$**. Nor does the worst-case bound assert the ERM's true risk decreases monotonically for every individual realized sample as $n$ grows.

### What changes without realizability?

If $\min_{h\in\mathcal H}R_{\mathcal D}(h)>0$, the inference “there is a zero-risk hypothesis, therefore ERM has zero empirical risk” is unavailable. ERM may have positive empirical risk. This is the **agnostic** setting; it requires a different style of argument (see [[LESSON 3]] for finite-class uniform convergence).

---

## 9. Active recall — close the notes before answering

- [ ] State Theorem 5.1, identifying what each of $\varepsilon$, $\delta$, $n$, and $|\mathcal H|$ means.
- [ ] Define realizability. Is it **existence**, **identification**, or **construction** of $h^\star$?
- [ ] Why does $R_{\mathcal D}(h^\star)=0$ imply $\widehat R_S(h^\star)=0$ almost surely?
- [ ] Why must an **exact** ERM have empirical risk zero, even without knowing $h^\star$?
- [ ] Give an example where $\hat h_S\neq h^\star$ yet both have empirical risk zero.
- [ ] Explain how $R_{\mathcal D}(h)=0.2$ and $\widehat R_S(h)=0$ can coexist.
- [ ] Why is $P(\widehat R_S(h)=0)=(1-R_{\mathcal D}(h))^n$ valid for **fixed** $h$?
- [ ] What breaks if the sample is not i.i.d. from $\mathcal D$?
- [ ] Define $\mathcal H_B$ and $E_h$ precisely. What is random in $E_h$?
- [ ] What would the intersection of all $E_h$ mean, and why do we need their union instead?
- [ ] Reproduce the union-bound chain, including the transition $|\mathcal H_B|\leq|\mathcal H|$.
- [ ] Explain why we cannot simply apply a fixed-$h$ probability bound directly to the sample-dependent ERM classifier.
- [ ] Solve $|\mathcal H|e^{-\varepsilon n}\leq\delta$ for $n$ without losing the inequality direction.
- [ ] Explain the final conclusion in plain language, distinguishing **true error tolerance** from **confidence over training samples**.

## 10. Small practice checks

**Check 1.** For a fixed $h$, let $R_{\mathcal D}(h)=0.2$ and $n=20$. Compute $\Pr(\widehat R_S(h)=0)$. Explain why observing zero training mistakes does **not** revise $R_{\mathcal D}(h)$ within this mathematical setup.

**Check 2.** Assume realizability. You find two classifiers with empirical errors $0$ and $1/n$. Can an exact ERM choose the second? Why?

**Check 3.** Write the *complement* of $\bigcup_{h\in\mathcal H_B}E_h$ using De Morgan. Interpret it in words.

**Check 4.** Let $|\mathcal H|=100$, $\varepsilon=0.2$, and $\delta=0.05$. Compute the minimum integer $n$ guaranteed by the theorem. Interpret *both* $0.2$ and $0.05$ without confusing their meanings.

**Check 5.** Explain why proving that “at least one bad classifier fits $S$” is **not** the same as proving that ERM definitely chooses one: the event bounded in the proof is a potentially larger **dangerous event**, which is enough for an upper bound.

---

## Where we stopped

**Covered now:** realizability; distinction between true risk, empirical risk, and expected empirical risk; best in class versus exact ERM; fixed-hypothesis probability of zero training errors; bad hypotheses and events $E_h$; union versus intersection; union bound; exponential upper bound; sample-size algebra; final $1-\delta$ conclusion.

**Next time:** Return to the lecture materials and continue from here. In particular, **do not mark later Lecture 4 topics as mastered until they have been studied and practiced independently.**
