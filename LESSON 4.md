---
title: "Lecture 4 — Realizability and ERM: Theorems 5.1 and 5.2"
aliases:
  - "Lecture 4 - Realizable ERM"
  - "Theorem 5.1 - Finite Hypothesis Class"
  - "Theorem 5.2 - Threshold Classifiers"
date: 2026-10-23
course: "Machine Learning — Sapienza"
academic_year: "2026/2027"
type: lecture-note
status: "In progress — Theorems 5.1 and 5.2 studied with guidance; VC dimension pending"
tags:
  - machine-learning
  - statistical-learning
  - realizability
  - empirical-risk-minimization
  - generalization
  - union-bound
  - sample-complexity
  - threshold-classifiers
---

# Lecture 4 — Realizability and ERM

> [!NOTE]
> **Central question**
> **Under realizability, how can ERM generalize despite zero training error — first for a finite family (Theorem 5.1), then for the infinite family of threshold classifiers (Theorem 5.2)?**

**Scope / study status:** This note records **Theorems 5.1 and 5.2 with their complete guided proofs**, including the conceptual obstacles encountered while studying it. It is **not** a claim that the rest of Lecture 4 (e.g. shattering or VC dimension) has already been studied. Continue this file as the lecture progresses.

**Connections:** [Lesson 2](LESSON%202.md) (true/empirical risk and overfitting) · [Lesson 3](LESSON%203.md) (concentration, union bound, ERM, finite families).

**Source boundary:** The theorem statement below is the one transcribed during our study discussion. The analogies, numerical examples, common mistakes, and step-by-step explanations are **study clarifications**, not additional claims about the lecturer's notes.

---

## 1. Theorem 5.1 — Statement

Let $\mathcal H$ be a **fixed, finite** family of classifiers, and assume **realizability**. For precision $\varepsilon\in(0,1)$, failure probability $\delta\in(0,1)$ and an i.i.d. training sample

$$ S=((x_1,y_1),\ldots,(x_n,y_n))\sim\mathcal D^n, $$

if

$$ \boxed{n\geq\frac{1}{\varepsilon}\log\left(\frac{\lvert\mathcal H\rvert}{\delta}\right),} $$

then an empirical risk minimizer $\hat h_S$ satisfies

$$ \boxed{\Pr_{S\sim\mathcal D^n}\left(R_{\mathcal D}(\hat h_S)\leq\varepsilon\right)\geq1-\delta.} $$

Here $\log$ is the natural logarithm. The probability is **over the random draw of the entire training set**. The chosen $\hat h_S$ can change as $S$ changes.

**In words:** With sufficiently many i.i.d. observations, the classifier selected by ERM has **true risk at most $\varepsilon$**, except with probability at most $\delta$.

- $\varepsilon$: tolerated **true error rate** (precision).
- $\delta$: tolerated **probability that the guarantee fails** (failure probability).
- $1-\delta$: confidence in the guarantee.
- $\lvert\mathcal H\rvert$: number of candidate classifiers; $\mathcal H$ must be finite for this bound.
- $n$: number of independent training observations.

> [!IMPORTANT]
> **What the theorem does NOT say**
> It does not claim that the ERM classifier is identical to the best-in-class classifier; it does not claim $R_{\mathcal D}(\hat h_S)=0$; it does not claim $R_S(h)=R_{\mathcal D}(h)$ for a particular sample.

---

## 2. Setup: distribution, sample, loss and risks

We consider classification with **0–1 loss**:

$$ \ell(h(x),y)=\mathbf 1\{h(x)\neq y\}. $$

For a **fixed** classifier $h$:

**True risk** (error probability for one new draw from the underlying distribution):

$$ R_{\mathcal D}(h) =\mathbb E_{(X,Y)\sim\mathcal D}[\ell(h(X),Y)] =\Pr_{(X,Y)\sim\mathcal D}(h(X)\neq Y). $$

**Empirical risk** (observed fraction of errors in a particular sample):

$$ \widehat R_S(h) =\frac1n\sum_{i=1}^n\mathbf 1\{h(x_i)\neq y_i\}. $$

We sometimes write $R_S(h)$ instead of $\widehat R_S(h)$; they mean the **same empirical quantity** in our discussions.

**Precision check:** $\mathcal D$ is a **probability distribution**, not a finite collection of examples. Thus “$S$ is a subset of $\mathcal D$” is an intuitive shorthand but not literally the mathematical relationship: **$S$ is sampled from $\mathcal D$**.

### Best in class versus ERM

$$ h^\star\in\arg\min_{h\in\mathcal H}R_{\mathcal D}(h), \qquad \hat h_S\in\arg\min_{h\in\mathcal H}\widehat R_S(h). $$

They minimize **different functions**, so they may select **different classifiers**.

Because $h^\star$ minimizes true risk **within $\mathcal H$**, we always have

$$ R_{\mathcal D}(\hat h_S)\geq R_{\mathcal D}(h^\star). $$

This fact alone does **not** give an upper bound on ERM's true risk: it is a **lower** bound.

---

## 3. The crucial assumption: realizability

> [!NOTE]
> **Realizability**
> We assume, **before sampling**, that there exists **at least one** classifier $h^\star\in\mathcal H$ satisfying
>
> $R_{\mathcal D}(h^\star)=0$.
>
> The assumption guarantees its **existence**, **not our knowledge of its identity**.

The student or algorithm generally does **not** know $\mathcal D$, and ERM is **not handed $h^\star$**. The theorem's proof is allowed to assume $h^\star$ exists because *realizability is one of the theorem's hypotheses*. It may be uncheckable in practice.

A zero true risk means that $h^\star$ is correct for a random $(X,Y)\sim\mathcal D$ **with probability 1** (almost surely). It does not necessarily assert correctness on points with zero probability under $\mathcal D$.

Since the training observations are drawn from that same $\mathcal D$, with probability 1,

$$ \widehat R_S(h^\star)=0. $$

Now use **only the definition of ERM**:

$$ \widehat R_S(\hat h_S) =\min_{h\in\mathcal H}\widehat R_S(h) \leq\widehat R_S(h^\star)=0. $$

Empirical risk is nonnegative, hence

$$ \boxed{\widehat R_S(\hat h_S)=0\quad\text{almost surely}.} $$

> [!IMPORTANT]
> **Explicit conclusion — why ERM has zero empirical risk**
> **Under the realizability assumption, we can say with probability 1 (not merely with confidence $1-\delta$) that the empirical risk of an exact ERM classifier is zero.** We know *a priori* that some $h^\star\in\mathcal H$ has zero true risk, so, for an i.i.d. sample $S$ drawn from $\mathcal D$, this same classifier has zero empirical risk almost surely. Because ERM minimizes empirical risk over $\mathcal H$, it must attain that available minimum of zero, even if it selects a different classifier. **This conclusion does not require the sample-size bound of Theorem 5.1**; that bound is needed to control ERM's *true* risk.

**This does not mean that ERM chooses $h^\star$.** It means that a zero empirical-risk candidate is available, so an **exact** empirical-risk minimizer cannot choose a candidate with positive empirical risk.

> [!TIP]
> **The key insight that unlocked the proof**
> **We know a perfect classifier exists, but not which one it is.** Its existence guarantees that the smallest training error available to ERM is zero. What remains unknown is the **true risk** of the zero-training-error classifier that ERM actually picks.

### Concrete example: two distinct perfect-on-sample classifiers

| Classifier | True risk $R_{\mathcal D}$ | Empirical risk $\widehat R_S$ |
|---|---:|---:|
| $h^\star$ (best in class) | $0$ | $0$ |
| $h_1$ | $0.20$ | $0$ |
| $h_2$ | $0.10$ | $0.05$ |
| $h_3$ | $0.30$ | $0.15$ |

For this **possible** sample, ERM can choose $h^\star$ **or** $h_1$ (a tie). It cannot choose $h_2$ or $h_3$ as an **exact ERM**, because their empirical risk is higher than the minimum zero.

$$ \boxed{\widehat R_S(\hat h_S)=0\ \nRightarrow\ R_{\mathcal D}(\hat h_S)=0.} $$

Under realizability, the difficulty is **not** whether ERM achieves zero *training* error; the difficulty is whether **a bad classifier can also achieve zero training error and be selected**.

---

## 4. Why true risk is used to compute a probability about the sample

This was one of the main conceptual sticking points.

Suppose a **fixed** classifier $h$ has true risk

$$ R_{\mathcal D}(h)=0.2. $$

By definition, for **one newly sampled example**,

$$ \Pr(h(X)\neq Y)=0.2, \qquad \Pr(h(X)=Y)=0.8. $$

For $n=20$ independently sampled examples, the probability that **all twenty** are classified correctly is

$$ \Pr_{S\sim\mathcal D^{20}}\left(\widehat R_S(h)=0\right) =0.8^{20} \approx 0.01153 =1.15\%. $$

**Connection to previous lessons:** [Lesson 2](LESSON%202.md) distinguishes **single-draw true risk** from the **observed error fraction in a sample**; [Lesson 3](LESSON%203.md) uses their relationship to establish the Hoeffding–Chernoff concentration bound. Theorem 5.1 uses the same single-draw risk in a different calculation: the probability that **all $n$ i.i.d. observations** are classified correctly.

**Why is using $0.2$ legitimate?** Because every observation in $S$ is drawn from **the same distribution** $\mathcal D$ used to define true risk. The factor $0.8$ is the chance of a **correct prediction on one random draw**, not a claim that $80\%$ of **this particular** sample must be correct.

A sample of 20 might contain 0, 2, 4, or 8 mistakes. Its empirical risk could be $0$, $0.1$, $0.2$, or $0.4$. The true risk remains $0.2$ for that fixed classifier.

In particular,

$$ R_{\mathcal D}(h)=0.2 \quad\nRightarrow\quad \widehat R_S(h)=0.2. $$

The correct statement for a **fixed** $h$ is

$$ \boxed{\mathbb E_{S\sim\mathcal D^n}[\widehat R_S(h)] =R_{\mathcal D}(h).} $$

This is an **equality of an expected sample average and the true risk**, **not** an equality for each realized training set. Moreover, one cannot blindly apply the fixed-$h$ argument to the sample-dependent $\hat h_S$.

**Analogy:** A coin lands heads with probability $0.8$. Twenty independent tosses need not contain exactly sixteen heads, but the probability of twenty heads is $0.8^{20}$.

> [!WARNING]
> **Boundary of this reasoning**
> If training examples were drawn from some different distribution $\mathcal Q$, the single-draw error probability on $S$ would be $R_{\mathcal Q}(h)$, **not necessarily** $R_{\mathcal D}(h)$. Independence is also essential for multiplying the probabilities across the $n$ draws.

---

## 5. The bad classifiers and the events $E_h$

Fix the accuracy tolerance $\varepsilon$. Define the set of **bad hypotheses**:

$$ \boxed{\mathcal H_B =\{h\in\mathcal H:R_{\mathcal D}(h)>\varepsilon\}.} $$

These are hypotheses that would **violate the theorem's desired conclusion** if ERM selected them.

For each **fixed** $h\in\mathcal H_B$, define an event in the space of possible training sets:

$$ \boxed{E_h =\{S\sim\mathcal D^n:\widehat R_S(h)=0\}.} $$

Read this as:

> “The **bad** classifier $h$ happens to make **zero errors on the sampled training set**.”

The event is superficially “positive” (perfect training accuracy), but **dangerous** because $h$ actually has large true risk.

Since $h\in\mathcal H_B$,

$$ R_{\mathcal D}(h)>\varepsilon \quad\Longrightarrow\quad 1-R_{\mathcal D}(h)<1-\varepsilon. $$

**$1-\varepsilon$ is not the empirical risk.** It is an **upper bound on the probability that one observation is classified correctly** by this bad hypothesis.

Using i.i.d. sampling,

$$ \begin{aligned} \Pr(E_h) &=\Pr(\text{all }n\text{ predictions of }h\text{ are correct})\\ &=(1-R_{\mathcal D}(h))^n\\ &\leq(1-\varepsilon)^n\\ &\leq e^{-\varepsilon n}. \end{aligned} $$

The last step uses $1-t\leq e^{-t}$ for $t\geq0$.

**Crucial quantifier:** This inequality holds **separately for every fixed bad classifier**. Raising a probability to $n$ means “correct on all **$n$ independent samples**”; it does **not** mean “correct for all the classifiers in $\mathcal H_B$”.

---

## 6. Why we take a union, not an intersection

The **failure** we care about is that **at least one bad classifier** looks perfect on the training set, giving ERM an opportunity to select it.

Thus define

$$ E=\bigcup_{h\in\mathcal H_B}E_h. $$

**Union:** $E_{h_1}\cup E_{h_2}\cup\cdots$ means **at least one** bad classifier gets zero training error.

**Intersection:** $E_{h_1}\cap E_{h_2}\cap\cdots$ would mean **all** bad classifiers simultaneously get zero training error — far stronger than needed, and not the dangerous condition we need to exclude.

**Complement:** $E^c$ means **none** of the bad classifiers has zero training error:

$$ E^c =\bigcap_{h\in\mathcal H_B} E_h^c. $$

On this complement event, because realizability guarantees ERM has zero training error, **ERM cannot select any classifier in $\mathcal H_B$**.

Apply the **union bound** (no independence assumption among the $E_h$ is required):

$$ \begin{aligned} \Pr(E) &=\Pr\left(\bigcup_{h\in\mathcal H_B}E_h\right)\\ &\leq\sum_{h\in\mathcal H_B}\Pr(E_h)\\ &\leq\lvert\mathcal H_B\rvert e^{-\varepsilon n}\\ &\leq\lvert\mathcal H\rvert e^{-\varepsilon n}. \end{aligned} $$

**Why replace $\lvert\mathcal H_B\rvert$ by $\lvert\mathcal H\rvert$?** Simply because $\mathcal H_B\subseteq\mathcal H$, so $\lvert\mathcal H_B\rvert\leq\lvert\mathcal H\rvert$. We may not know how many hypotheses are actually bad, but we know they cannot outnumber the entire finite class.

The events $E_h$ share **the same sampled dataset** and can be dependent. The union bound still holds.

---

## 7. Full proof map: from realizability to the bound

**Given:** fixed finite $\mathcal H$; realizability; i.i.d. $S\sim\mathcal D^n$; $0<\varepsilon,\delta<1$.

**Goal:**

$$ \Pr\big(R_{\mathcal D}(\hat h_S)>\varepsilon\big)\leq\delta. $$

**Step A — realizability gives a perfect candidate.** There exists $h^\star\in\mathcal H$ with $R_{\mathcal D}(h^\star)=0$, hence $\widehat R_S(h^\star)=0$ almost surely.

**Step B — exact ERM also has zero sample risk.**

$$ \widehat R_S(\hat h_S)=0\qquad\text{almost surely}. $$

**Step C — a bad ERM selection would imply a bad but sample-perfect classifier.**

$$ \left\{S:R_{\mathcal D}(\hat h_S)>\varepsilon\right\} \subseteq \bigcup_{h\in\mathcal H_B}\{S:\widehat R_S(h)=0\}, $$

apart from a probability-zero exception under realizability.

This is the **logical bridge** from “the ERM selected a bad hypothesis” to the union of fixed-hypothesis events. We **do not** pretend that $\hat h_S$ was fixed before observing $S$.

**Step D — bound the dangerous union.**

$$ \Pr\big(R_{\mathcal D}(\hat h_S)>\varepsilon\big) \leq\lvert\mathcal H\rvert e^{-\varepsilon n}. $$

**Step E — choose $n$ so that the failure probability is at most $\delta$.**

$$ \begin{aligned} \lvert\mathcal H\rvert e^{-\varepsilon n}&\leq\delta\\ e^{-\varepsilon n}&\leq\frac{\delta}{\lvert\mathcal H\rvert}\\ -\varepsilon n&\leq\log\delta-\log\lvert\mathcal H\rvert\\ \varepsilon n&\geq\log\lvert\mathcal H\rvert-\log\delta\\ n&\geq\frac1\varepsilon\log\left(\frac{\lvert\mathcal H\rvert}{\delta}\right). \end{aligned} $$

The sign reverses when multiplying by $-1$.

Consequently,

$$ \Pr\big(R_{\mathcal D}(\hat h_S)>\varepsilon\big)\leq\delta \quad\Longleftrightarrow\quad \boxed{\Pr\big(R_{\mathcal D}(\hat h_S)\leq\varepsilon\big)\geq1-\delta.} $$

The final equivalence uses the **complement rule**.

> [!TIP]
> **One-sentence proof**
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
| “Why replace $\lvert\mathcal H_B\rvert$ with $\lvert\mathcal H\rvert$?” | Because $\mathcal H_B\subseteq\mathcal H$ and therefore $\lvert\mathcal H_B\rvert\leq\lvert\mathcal H\rvert$. |
| “Why is the last term $\leq\delta$?” | The assumed lower bound on $n$ is **chosen precisely to make** $\lvert\mathcal H\rvert e^{-\varepsilon n}\leq\delta$. |
| “Does probability $1-\delta$ describe the accuracy on each example?” | No. It describes **how likely a random training sample is to produce an ERM whose true error rate is at most $\varepsilon$**. |

### A useful distinction: guarantee versus observed behavior

Even with large $n$, a **particular** training sample might allow a bad classifier to get zero empirical error. The theorem does not ban that possibility. It puts a bound on its **probability over draws of $S$**. Nor does the worst-case bound assert the ERM's true risk decreases monotonically for every individual realized sample as $n$ grows.

### What changes without realizability?

If $\min_{h\in\mathcal H}R_{\mathcal D}(h)>0$, the inference “there is a zero-risk hypothesis, therefore ERM has zero empirical risk” is unavailable. ERM may have positive empirical risk. This is the **agnostic** setting; it requires a different style of argument (see [Lesson 3](LESSON%203.md) for finite-class uniform convergence).

---

## 9. Active recall — close the notes before answering

- [ ] State Theorem 5.1, identifying what each of $\varepsilon$, $\delta$, $n$, and $\lvert\mathcal H\rvert$ means.
- [ ] Define realizability. Is it **existence**, **identification**, or **construction** of $h^\star$?
- [ ] Why does $R_{\mathcal D}(h^\star)=0$ imply $\widehat R_S(h^\star)=0$ almost surely?
- [ ] Why must an **exact** ERM have empirical risk zero, even without knowing $h^\star$?
- [ ] Give an example where $\hat h_S\neq h^\star$ yet both have empirical risk zero.
- [ ] Explain how $R_{\mathcal D}(h)=0.2$ and $\widehat R_S(h)=0$ can coexist.
- [ ] Why is $P(\widehat R_S(h)=0)=(1-R_{\mathcal D}(h))^n$ valid for **fixed** $h$?
- [ ] What breaks if the sample is not i.i.d. from $\mathcal D$?
- [ ] Define $\mathcal H_B$ and $E_h$ precisely. What is random in $E_h$?
- [ ] What would the intersection of all $E_h$ mean, and why do we need their union instead?
- [ ] Reproduce the union-bound chain, including the transition $\lvert\mathcal H_B\rvert\leq\lvert\mathcal H\rvert$.
- [ ] Explain why we cannot simply apply a fixed-$h$ probability bound directly to the sample-dependent ERM classifier.
- [ ] Solve $\lvert\mathcal H\rvert e^{-\varepsilon n}\leq\delta$ for $n$ without losing the inequality direction.
- [ ] Explain the final conclusion in plain language, distinguishing **true error tolerance** from **confidence over training samples**.

## 10. Small practice checks

**Check 1.** For a fixed $h$, let $R_{\mathcal D}(h)=0.2$ and $n=20$. Compute $\Pr(\widehat R_S(h)=0)$. Explain why observing zero training mistakes does **not** revise $R_{\mathcal D}(h)$ within this mathematical setup.

**Check 2.** Assume realizability. You find two classifiers with empirical errors $0$ and $1/n$. Can an exact ERM choose the second? Why?

**Check 3.** Write the *complement* of $\bigcup_{h\in\mathcal H_B}E_h$ using De Morgan. Interpret it in words.

**Check 4.** Let $\lvert\mathcal H\rvert=100$, $\varepsilon=0.2$, and $\delta=0.05$. Compute the minimum integer $n$ guaranteed by the theorem. Interpret *both* $0.2$ and $0.05$ without confusing their meanings.

**Check 5.** Explain why proving that “at least one bad classifier fits $S$” is **not** the same as proving that ERM definitely chooses one: the event bounded in the proof is a potentially larger **dangerous event**, which is enough for an upper bound.

---

## 11. Theorem 5.2 — Threshold classifiers (2026/27 confirmed)

**Sources and status:** Current-year *Week 1 — Statistical Learning*, §5, Theorem 5.2; professor's handwritten *Lecture 4 (6 October 2026)* treats threshold classifiers. The examples, cautions, and explanations here reconstruct our guided study, **not quotations from the professor**. Theorem 5.2 has been explained and solved step-by-step; **full independent proof reconstruction remains to be verified**. See the current-year *Exercise 1*, exercise 1.3 for practice.

### 11.1 Why the finite-family proof does not directly apply

Theorem 5.1 uses $|\mathcal H|e^{-\varepsilon n}$. The family of all real-valued thresholds is **infinite**, so replacing $|\mathcal H|$ by an infinite cardinality gives no useful finite bound. We need a different argument: replace infinitely many possible *bad classifiers* with **two fixed intervals** that any badly chosen ERM threshold would force the sample to miss.

We use binary classification with the threshold rule

$ h_\theta(x)=\mathbf1\{x\geq\theta\},\qquad\theta\in[0,1]. $

The assumptions are:

- $X\sim\mathrm{Unif}([0,1])$;
- **realizability:** for some unknown $\theta^\star\in[0,1]$, $Y=h_{\theta^\star}(X)$ almost surely;
- $S=((X_1,Y_1),\ldots,(X_n,Y_n))$ consists of $n$ i.i.d. draws;
- $\hat\theta=\hat\theta(S)$ denotes a threshold chosen by an **exact ERM** from this family;
- $0<\varepsilon,\delta<1$.

Since $h_{\theta^\star}$ is available within the class and perfectly labels sampled examples, realizability and ERM give

$ \boxed{\widehat R_S(h_{\hat\theta})=0\quad\text{almost surely}.} $

This prerequisite remains in force throughout the argument. **Without it, we cannot infer that the ERM threshold's disagreement region contains no training points.**

### 11.2 What is the true risk of a threshold?

For any fixed candidate threshold $\theta$, realizability and 0–1 loss give

$ R_{\mathcal D}(h_\theta)=\Pr(h_\theta(X)\neq h_{\theta^\star}(X)). $

The two predictions differ **exactly between the two thresholds**, meaning on the interval from $\min(\theta,\theta^\star)$ to $\max(\theta,\theta^\star)$ (with an endpoint convention that does not affect probabilities under the continuous uniform distribution).

Uniformity now lets us identify **probability mass with interval length**:

$ \boxed{R_{\mathcal D}(h_\theta)=\max(\theta,\theta^\star)-\min(\theta,\theta^\star)=|\theta-\theta^\star|.} $

The absolute value appears because it measures the **length of the disagreement interval**, not merely because risk must be nonnegative. If $X$ were not uniform, true risk would instead be the *probability mass* of the interval, generally not its length.

**Meaning of $\varepsilon$:** It is an **allowed true-risk tolerance**. In this particular uniform-threshold model, the risk requirement $R_{\mathcal D}(h_{\hat\theta})\leq\varepsilon$ is equivalent to the geometrical condition

$ \theta^\star-\varepsilon\leq\hat\theta\leq\theta^\star+\varepsilon. $

**Do not replace $>$ by $=$.** The bad case $\hat\theta>\theta^\star+\varepsilon$ ranges over *all* thresholds more than $\varepsilon$ too high; it does **not** assume $\hat\theta=\theta^\star+\varepsilon$.

### 11.3 Too-high threshold: a necessary empty interval

Let $\theta^\star=0.4$, $\varepsilon=0.1$, and suppose $\hat\theta=0.7$. The ERM classifier disagrees with the perfect one throughout **$[0.4,0.7)$**, so its true risk is **$0.3$**, not $0.1$.

For instance, $x_i=0.55$ lies *outside* $[0.4,0.5)$ but still yields $h_{0.4}(0.55)=1$ and $h_{0.7}(0.55)=0$. If such a point appeared in the **training sample**, empirical risk would be at least $1/n$, contradicting zero ERM empirical risk.

To study **every possible** threshold above $0.5$, instead of a different error interval for each threshold, choose the **fixed** subinterval $I_+=[0.4,0.5)$ of length $\varepsilon$. It is contained in $[0.4,\hat\theta)$ whenever $\hat\theta>0.5$. Therefore,

$ \boxed{\hat\theta>\theta^\star+\varepsilon\ \Longrightarrow\ S\cap[\theta^\star,\theta^\star+\varepsilon)=\varnothing.} $

Here $S\cap I=\varnothing$ is shorthand for **no feature value $X_i$ in $S$ belongs to $I$**; formally $\forall i,\ X_i\notin I$.

This is an implication, **not** an equivalence. The training sample might avoid the fixed small interval and still lead ERM to choose a perfectly acceptable threshold. Example: for $\theta^\star=0.4$, the sample $(0.2,0),(0.8,1)$ avoids $[0.4,0.5)$, but $\hat\theta=0.45$ fits the sample perfectly and is **not** too high.

Define $A_+=\{\hat\theta>\theta^\star+\varepsilon\}$ and $B_+=\{\forall i,\ X_i\notin I_+\}$. From $A_+\subseteq B_+$, **monotonicity of probability** gives

$ \Pr_S(A_+)\leq\Pr_S(B_+). $

For one draw, $\Pr(X_i\notin I_+)=1-\varepsilon$. **For all $n$ independent draws**, the avoidance probability is $(1-\varepsilon)^n$, rather than $1-\varepsilon$:

$ \boxed{\Pr_S(\hat\theta>\theta^\star+\varepsilon)\leq(1-\varepsilon)^n\leq e^{-\varepsilon n}.} $

The second inequality uses $1-t\leq e^{-t}$ for $t\geq0$. We have an **upper bound**, not necessarily the exact probability that ERM chooses a high threshold: avoiding $I_+$ is *necessary but not sufficient* for that outcome.

### 11.4 Too-low threshold: the symmetric fixed interval

If $\hat\theta<\theta^\star-\varepsilon$, then the complete disagreement interval is $[\hat\theta,\theta^\star)$, but every such disagreement interval contains the **fixed** interval $I_-=[\theta^\star-\varepsilon,\theta^\star)$.

For $\theta^\star=0.4$ and $\varepsilon=0.1$, the key interval is $[0.3,0.4)$, regardless of whether the chosen threshold equals $0.29$, $0.2$, or $0.15$.

As before, empirical risk zero implies no sample observation can fall there. Set $A_-=\{\hat\theta<\theta^\star-\varepsilon\}$ and $B_-=\{\forall i,\ X_i\notin I_-\}$:

$ A_-\subseteq B_-\quad\Longrightarrow\quad\Pr_S(A_-)\leq\Pr_S(B_-)=(1-\varepsilon)^n\leq e^{-\varepsilon n}. $

**Boundary caveat:** If the interval $I_+$ or $I_-$ would extend outside $[0,1]$, the corresponding bad-threshold event is impossible, so its probability is zero. In the nontrivial cases the fixed interval lies inside $[0,1]$ and has length $\varepsilon$. This avoids silently treating an out-of-range interval as having probability $\varepsilon$.

### 11.5 Why the bad event is a UNION, not an intersection

The goal is to bound the failure event:

$ F=\{R_{\mathcal D}(h_{\hat\theta})>\varepsilon\}=\{|\hat\theta-\theta^\star|>\varepsilon\}. $

For a *strict* absolute-value inequality,

$ |a|>\varepsilon\quad\Longleftrightarrow\quad(a<-\varepsilon)\ \textbf{OR}\ (a>\varepsilon). $

Therefore the failure event is **too low OR too high**:

$ \boxed{F=A_-\cup A_+.} $

A threshold cannot be simultaneously too high and too low. The two events are disjoint, but using the **union bound** remains valid:

$ \begin{aligned}\Pr_S(F)&=\Pr_S(A_-\cup A_+)\\&\leq\Pr_S(A_-)+\Pr_S(A_+)\\&\leq2e^{-\varepsilon n}.\end{aligned} $

**Contrast:** $|a|\leq\varepsilon$ means $a\geq-\varepsilon$ **AND** $a\leq\varepsilon$; remaining **inside** an interval requires both conditions, whereas being **outside** requires only one.

Crucially, the coefficient $2$ counts the **two types of failure events**, *not* the number of hypotheses (which is infinite).

### 11.6 Sample complexity and the final PAC guarantee

To make $\Pr_S(F)\leq\delta$, it is **sufficient** to require

$ 2e^{-\varepsilon n}\leq\delta. $

Taking natural logarithms preserves the inequality because $\ln$ is increasing. Then dividing by the negative value $-\varepsilon$ **reverses its direction**:

$ \begin{aligned}e^{-\varepsilon n}&\leq\delta/2\\-\varepsilon n&\leq\ln(\delta/2)\\n&\geq-\frac1\varepsilon\ln(\delta/2)=\frac1\varepsilon\ln(2/\delta).\end{aligned} $

Thus Theorem 5.2 guarantees:

$ \boxed{n\geq\frac{1}{\varepsilon}\ln\left(\frac2\delta\right)\quad\Longrightarrow\quad \Pr_{S\sim\mathcal D^n}\big(R_{\mathcal D}(h_{\hat\theta})\leq\varepsilon\big)\geq1-\delta.} $

For an actual dataset, round the required $n$ **up to the nearest integer**. The guarantee is a probability **over random training samples**: it does not say every particular sample leads to risk at most $\varepsilon$.

### 11.7 Critical distinctions discovered during study

| Tempting but incorrect thought | Correct interpretation |
|---|---|
| “$\hat\theta$ is *equal* to $\theta^\star+\varepsilon$.” | The proof studies **all** $\hat\theta>\theta^\star+\varepsilon$. |
| “Errors happen only in the interval of length $\varepsilon$.” | Errors happen on the **entire** interval between $\hat\theta$ and $\theta^\star$. The shorter $\varepsilon$ interval is a *fixed subset* useful for bounding probabilities. |
| “Why must the sample avoid that interval?” | **Realizability + exact ERM** force empirical risk zero; even one observed point in the disagreement region would contradict this. |
| “If no point falls in the fixed interval, ERM must choose a bad threshold.” | No: the avoidance event is only a **necessary**, not a sufficient, condition. $A\subseteq B$ yields $\Pr(A)\leq\Pr(B)$ by monotonicity. |
| “The probability all $n$ samples avoid it is $1-\varepsilon$.” | That is the **one-draw** probability. For $n$ i.i.d. draws it is $(1-\varepsilon)^n$. |
| “Because the risk involves a modulus, the bad event uses AND.” | $|a|>\varepsilon$ uses **OR**; the **union** covers too-high and too-low thresholds. |
| “The bound is the actual probability that ERM fails.” | It is a **worst-case upper bound**, potentially loose. |

### 11.8 Active recall and exercises (do not read the proof while attempting)

- [ ] Explain why realizability makes the empirical risk of an exact ERM equal to zero.
- [ ] Starting from $h_\theta(x)=\mathbf1\{x\geq\theta\}$, derive $R_{\mathcal D}(h_\theta)=|\theta-\theta^\star|$, naming the uniformity assumption.
- [ ] For $\theta^\star=0.4$, $\varepsilon=0.1$, $\hat\theta=0.7$, specify the **whole** error interval, the **fixed subinterval** used in the proof, and the actual true risk.
- [ ] Reconstruct the implication $A_+\subseteq B_+$ and explain why its converse can fail.
- [ ] Explain why $P(B_+)=(1-\varepsilon)^n$ and **which** independence is used.
- [ ] Reconstruct the symmetric bound for too-low thresholds without consulting the section above.
- [ ] Translate $|\hat\theta-\theta^\star|>\varepsilon$ into two inequalities; identify the union.
- [ ] Reproduce the union-bound inequality and derive the sample-size requirement from $2e^{-\varepsilon n}\leq\delta$.
- [ ] Explain why the finite-$|\mathcal H|$ theorem cannot simply be used with this infinite threshold family.
- [ ] **Autonomous proof on paper:** hypotheses → zero empirical risk → fixed intervals → event inclusions → $(1-\varepsilon)^n$ → union bound → PAC conclusion.
- [ ] **Current course exercise:** Attempt *Exercise 1*, problem **1.3** (Theorem 5.2) independently; record where assistance was needed.

---

## Where we stopped

**Covered with guidance:** Theorem 5.1 (finite hypothesis families) and Theorem 5.2 (threshold classifiers): realizability, zero empirical risk under exact ERM, the error-region geometry, fixed $\varepsilon$ intervals, necessary versus sufficient event conditions, monotonicity of probability, i.i.d. avoidance probabilities, union bound, and sample-size algebra.

**Current status:** Both theorems are **understood via guided reasoning**, not yet independently reconstructed in full from a blank page. In the Theorem 5.2 dialogue the student independently recovered the two one-sided bounds, union-bound factor $2$, and final $n\geq\varepsilon^{-1}\ln(2/\delta)$ once earlier confusions were resolved.

**Next step:** Write the Theorem 5.2 proof from memory, then solve the current-year exercise 1.3. Re-test Theorem 5.1 and proceed to shattering / VC dimension from the current Lecture 4 material. Do **not** mark future topics as mastered before independent testing.

