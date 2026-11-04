# Exercises 0 — Optimal Predictors: Fully Worked Solutions

> **Course:** Machine Learning, Sapienza, 2026/27 — Week 0, *Decision Theory*, exercise sheet by Federico Fusco (28 September 2026, v1).  
> **Source:** [Exercises 0.pdf](https://drive.google.com/file/d/1_Hof095ba6LRs4LqzgetMNyAV2tqh1Tu/view) (pp. 1–3).  
> **Status:** Solutions and teaching commentary independently derived, **not official instructor solutions**.  
> **Conventions:** The sheet uses $\ell(a,b)$ with **true label first** ($a$) and **prediction second** ($b$). Thus $R(f)=\mathbb{E}[\ell(Y,f(X))]$. This differs from some notes that write prediction first; for asymmetric losses the difference matters. All optimizations below are conditional on a fixed $X=x$ unless otherwise stated.

## Read this first: notation and methodology

- $X$ is a random input, $x$ is a realized input; $Y$ is a random outcome, $y$ a realized label; $z$ is a proposed prediction. $\mathcal D$ is a distribution, **not a dataset**.
- $\eta(x)=P(Y=1\mid X=x)$; $p_+(x)$ and $p_-(x)$ are *densities of X conditional on the class*, **not posterior probabilities**.
- Define conditional risk $r_x(z)=\mathbb{E}[\ell(Y,z)\mid X=x]$. A Bayes predictor chooses $f^*(x)$ minimizing $r_x(z)$. Total risk is $R(f)=\mathbb{E}_X[r_X(f(X))]$ by the law of total expectation.
- $\mathbb{E}[Y\mid X=x]$ is a number depending on $x$, while $\mathbb{E}[Y\mid X]$ is a random variable/function of $X$.
- $t_+=\max(t,0)$; $\Phi$ and $\phi$ denote the standard normal CDF and PDF. Assume required moments exist wherever expectations are taken. Ties may be broken arbitrarily, without affecting the risk.

---

## Exercise 1.1 — Mean, median, mode (source p. 1)

### (a) Squared loss: why the mean?

For $\ell(y,z)=(y-z)^2$, expand around $\mu_x=\mathbb{E}[Y\mid X=x]$:

$$
\begin{aligned}
r_x(z)&=\mathbb{E}[(Y-z)^2\mid x]\\
&=\mathbb{E}[((Y-\mu_x)+(\mu_x-z))^2\mid x]\\
&=\mathbb{E}[(Y-\mu_x)^2\mid x]+2(\mu_x-z)\mathbb{E}[Y-\mu_x\mid x]+(\mu_x-z)^2\\
&=\mathrm{Var}(Y\mid X=x)+(z-\mu_x)^2.
\end{aligned}
$$

The middle term is zero by definition of $\mu_x$. The variance is independent of $z$, whereas the last term is nonnegative and vanishes precisely when $z=\mu_x$. Therefore

$$f^*(x)=\mathbb{E}[Y\mid X=x],\qquad r_x(f^*)=\mathrm{Var}(Y\mid X=x).$$

**Likely stumbling point:** minimizing the expected loss does **not** require knowing a realized $Y$: we use its conditional distribution. Do not replace this expectation with an empirical training-set average.

### (b) Absolute loss: why a median?

Let $F_x(z)=P(Y\le z\mid X=x)$, with conditional density $q_x$. Then

$$r_x(z)=\int_{-\infty}^z(z-y)q_x(y)\,dy+\int_z^{\infty}(y-z)q_x(y)\,dy.$$

Differentiate using Leibniz's rule. The boundary terms at $y=z$ vanish because the integrands vanish there:

$$r_x'(z)=\int_{-\infty}^z q_x(y)\,dy-\int_z^\infty q_x(y)\,dy=2F_x(z)-1.$$

Thus a minimizer satisfies $F_x(z)=1/2$ when such a point exists. More generally, the minimizers are all conditional medians: $F_x(z)\ge1/2$ and $P(Y\ge z\mid x)\ge1/2$. The derivative changes from negative to positive, so the point is a minimum. Unlike squared loss, **absolute loss targets a median, not necessarily a mean**.

### (c) Power loss as $p\to\infty$: midrange

**Important technical qualification (external mathematical clarification):** the stated endpoints $m,M$ must be the *essential* minimum and maximum of the conditional distribution, not merely arbitrary enclosing bounds. If, for instance, $Y\equiv0$ but someone supplies the enclosing interval $[-2,3]$, the answer is $0$, not $1/2$.

Set $J_p(z)=\mathbb{E}[|Y-z|^p\mid x]$ and $A_p(z)=J_p(z)^{1/p}$. Since taking a positive $p$-th root is strictly increasing, $J_p$ and $A_p$ have the same minimizers. For a bounded random variable, the $L^p$ norm tends to its essential supremum:

$$A_p(z)\longrightarrow A_\infty(z)=\max(|z-m|,|z-M|).$$

To minimize the maximum endpoint distance, balance the two distances:

$$z-m=M-z\quad\Longrightarrow\quad z_\infty=\frac{m+M}{2}.$$

Why does convergence of functions imply convergence of minimizers here? Every minimizer can be taken inside $[m,M]$ (moving an outside $z$ toward the interval decreases every distance). On this compact interval, the $L^p$ norms increase to the continuous limit $A_\infty$; this convergence is uniform (or use a compact-subsequence argument and inequalities comparing $L^p$ norms). Every subsequential limit of minimizers therefore minimizes $A_\infty$. Its minimizer is unique, hence all minimizers converge to $(m+M)/2$.

**Intuition:** large $p$ makes the largest errors dominate. The midrange is the point minimizing the worst possible distance; the midpoint is *not* generally the median or mean.

### (d) Tolerance loss: a local mode (challenging)

$$\ell(y,z)=\mathbf{1}\{|y-z|>\varepsilon\}.$$

Because an indicator has expected value equal to the probability of its event,

$$\begin{aligned}
r_x(z)&=P(|Y-z|>\varepsilon\mid x)\\
&=1-P(|Y-z|\le\varepsilon\mid x).
\end{aligned}$$

Thus minimizing risk is equivalent to maximizing the probability mass in the window $[z-\varepsilon,z+\varepsilon]$:

$$f^*(x)\in\arg\max_z\int_{z-\varepsilon}^{z+\varepsilon}q_x(y)\,dy.$$

For a continuous density $q_x$, the mean value of the density in this interval tends to the density at its center:

$$\frac{1}{2\varepsilon}\int_{z-\varepsilon}^{z+\varepsilon}q_x(y)\,dy\longrightarrow q_x(z).$$

Therefore the objective favors **modes** of $q_x$ as $\varepsilon\to0$. To assert convergence of *maximizers*, additional regularity/uniqueness and tightness conditions are needed; pointwise approximation alone is not a general convergence theorem. With multiple modes, different optimal centers may be possible.

### (e) Equal Gaussian mixture: compare what each loss means

Let $Y\mid x\sim\tfrac12N(-3,1)+\tfrac12N(3,1)$. By symmetry $\mathbb{E}[Y\mid x]=0$ and $0$ is a median. The variance is

$$\begin{aligned}\mathrm{Var}(Y\mid x)&=\mathbb{E}[\mathrm{Var}(Y\mid\text{component},x)]\\
&\quad+\mathrm{Var}(\mathbb{E}[Y\mid\text{component},x])\\
&=1+\tfrac12(-3)^2+\tfrac12(3)^2=10.
\end{aligned}$$

**Squared loss:** optimal prediction $0$, conditional risk $10$. Predicting $3$ instead gives $\mathbb{E}[(Y-3)^2]=\mathrm{Var}(Y)+(\mathbb{E}[Y]-3)^2=10+9=19$.

**Absolute loss:** optimal prediction $0$. For $Z\sim N(\mu,1)$ with $\mu\ge0$,

$$\mathbb{E}|Z|=2\phi(\mu)+\mu(2\Phi(\mu)-1).$$

By symmetry, the mixture absolute risk at $0$ is

$$r_x(0)=2\phi(3)+3(2\Phi(3)-1)\approx3.00076.$$

This formula follows by splitting $\int |z|\phi(z-\mu)\,dz$ at $0$, replacing $t=z-\mu$, and using $\int t\phi(t)\,dt=-\phi(t)$.

**Subtlety:** the actual mixture modes are *near*, but not exactly at, $\pm3$ because the other Gaussian contributes a small slope. The exercise calls $a=3$ a mode informally; our comparison still evaluates the requested prediction $3$ exactly.

**Conclusion:** squared loss rewards the mean, absolute loss rewards medians, and small-window tolerance loss rewards peaks of density. The "best prediction" has no meaning until a **loss** has been specified.

---

## Exercise 1.2 — Quantiles and newsvendor (source pp. 1–2)

### Crucial source inconsistency: arguments of the pinball loss

The sheet defines $\ell(a,b)=\tau(b-a)_++(1-\tau)(a-b)_+$, with **true value $a$ and prediction $b$**. This literal formula targets the $(1-\tau)$-quantile, whereas part (a) requests the $\tau$-quantile. We derive both, rather than silently swapping arguments.

### (a) Literal formula vs requested quantile

With $y$ true and $z$ predicted, the **printed** loss is

$$\ell(y,z)=\tau(z-y)_++(1-\tau)(y-z)_+.$$

So

$$r_x(z)=\tau\int_{-\infty}^z(z-y)q_x(y)\,dy+(1-\tau)\int_z^\infty(y-z)q_x(y)\,dy.$$

Differentiate, as in 1.1(b):

$$r_x'(z)=\tau F_x(z)-(1-\tau)(1-F_x(z))=F_x(z)-(1-\tau).$$

Consequently the **literal sheet formula** yields $F_x(z)=1-\tau$ (or the corresponding quantile condition if the CDF is flat). To yield the *stated requested* $\tau$-quantile, the loss should instead be

$$\ell_{\tau}(y,z)=\tau(y-z)_++(1-\tau)(z-y)_+,$$

whose derivative is $F_x(z)-\tau$. With either convention $\tau=1/2$ yields the median of 1.1(b).

**Exam warning:** state the argument order and the derivative; this is safer than quoting "pinball predicts the $\tau$-quantile" without checking signs.

### (b) Stock optimization

Stock $z$; random demand $Y$; shortage cost $c_u>0$ per unit and overstock cost $c_o>0$ per unit:

$$C(Y,z)=c_u(Y-z)_++c_o(z-Y)_+.$$

Differentiate expected cost:

$$\frac{d}{dz}\mathbb{E}[C(Y,z)]=(c_o+c_u)F_Y(z)-c_u.$$

At an interior optimum,

$$\boxed{F_Y(z^*)=\frac{c_u}{c_u+c_o}.}$$

If demand is discrete, choose a quantile with $F_Y(z^*-)\le c_u/(c_u+c_o)\le F_Y(z^*)$. For nonnegative/integer stock, enforce those feasibility restrictions. High shortage cost moves the optimal stock *up*; high overstock cost moves it *down*. This is the quantile in the **corrected orientation** of part (a), not the literal printed $\tau$ convention.

---

## Exercise 1.3 — Randomization never helps (source p. 2)

### (a) Uniform random $k$-class prediction

Let $Z$ be uniformly distributed over $k$ labels, independent of $(X,Y)$. Conditional on any true label $Y=y$, $P(Z=y\mid Y=y)=1/k$. Hence

$$R(Z)=P(Z\ne Y)=1-\frac1k.$$

A deterministic Bayes rule is no worse than any randomized rule (proved next), so $R^*\le 1-1/k$. More concretely, at each $x$ let $p_j(x)=P(Y=j\mid X=x)$. The Bayes error is $1-\max_j p_j(x)$, and $\max_jp_j(x)\ge\tfrac1k$. Equality of the total risk holds **if and only if** $p_j(X)=1/k$ for all classes $j$ almost surely: the posterior labels are uniform for almost every input.

### (b) Any loss: average cannot beat its minimum (challenging)

Let $Z\sim Q(\cdot\mid X=x)$ be conditionally independent of $Y$ given $X$. Define $r_x(z)=\mathbb{E}[\ell(Y,z)\mid X=x]$. By conditional independence and the tower property,

$$\mathbb{E}[\ell(Y,Z)\mid X=x]=\mathbb{E}_{Z\sim Q(\cdot\mid x)}[r_x(Z)].$$

For every possible $z$, $r_x(z)\ge\inf_u r_x(u)$. Taking an average of these inequalities preserves the direction:

$$\mathbb{E}_Z[r_x(Z)]\ge\inf_u r_x(u)=r_x(f^*(x)).$$

Average over $X$: $R(Q)\ge R(f^*)$. Randomization cannot improve Bayes risk. It can tie if it only randomizes among equally optimal decisions.

### (c) Squared loss: variance penalty (challenging)

Let $m(x)=\mathbb{E}[Z\mid X=x]$ and suppose conditional second moments are finite. Expand $Y-Z=(Y-m)+(m-Z)$:

$$\begin{aligned}\mathbb{E}[(Y-Z)^2\mid x]&=\mathbb{E}[(Y-m)^2\mid x]+\mathbb{E}[(Z-m)^2\mid x]\\
&\quad-2\mathbb{E}[(Y-m)(Z-m)\mid x]\\
&=\mathbb{E}[(Y-m)^2\mid x]+\mathrm{Var}(Z\mid x).
\end{aligned}$$

The cross term vanishes by conditional independence and $\mathbb{E}[Z-m\mid x]=0$. Thus randomizing is strictly worse than predicting its mean whenever $\mathrm{Var}(Z\mid X)>0$ on a set with positive probability (not merely at a single zero-probability $x$). The excess risk is exactly $\mathbb{E}[\mathrm{Var}(Z\mid X)]$.

---

## Exercise 1.4 — Information never hurts (source p. 2)

### (a) Proof by inclusion of decision rules

A predictor based only on $g(X)$ has the form $h(g(X))$. A predictor based on full $X$ can always **imitate** it by taking $f(x)=h(g(x))$. Therefore the set of rules based on $g(X)$ is contained in the set of rules based on $X$:

$$\{h\circ g\}\subseteq\{f:\mathcal X\to\mathcal Y\}.$$

Minimizing the same risk over a larger set cannot increase the minimum:

$$\boxed{R^*(X)\le R^*(g(X)).}$$

This does not say every predictor using more information is good; it says **the best achievable** predictor cannot become worse.

### (b) Three explicit examples

**Strict inequality:** $X\in\{-1,1\}$ uniform, $Y=X$, 0-1 loss, $g(X)=0$ constant. Using $X$, predict $Y=X$ for risk $0$; using $g(X)$ alone, labels are equiprobable and optimal risk is $1/2$.

**Equality, discarded irrelevant information:** $X=(U,V)$ where $U,V$ are independent fair bits and $Y=U$; set $g(U,V)=U$. It discards $V$, but both optimal risks are $0$.

**Equality, noninjective $g$ without loss:** $X\in\{-2,-1,1,2\}$ uniform, $Y=\mathbf{1}\{X>0\}$, $g(X)=\mathbf{1}\{X>0\}$. Different input values map to the same feature, yet it perfectly determines $Y$; both risks are $0$. More generally, equality holds when $g(X)$ retains sufficient information for an optimal decision, not necessarily all information about $X$.

---

## Exercise 1.5 — Binary classification and excess risk (source p. 2)

### (a) Opposite classifiers

For $f(X)\in\{-1,1\}$, exactly one of $f(X)$ and $-f(X)$ agrees with $Y$ on each outcome:

$$\mathbf{1}\{f(X)\ne Y\}+\mathbf{1}\{-f(X)\ne Y\}=1.$$

Take expectations: $\boxed{R(f)+R(-f)=1}$. The statement requires labels and predictions in $\{-1,1\}$ with 0-1 loss.

### (b) Why Bayes risk cannot exceed one half

The smaller of $R(f)$ and $R(-f)$ is at most their average $1/2$. Since the Bayes rule is no worse than either, $\boxed{R^*\le1/2}$.

### (c) Bayes decision and exact excess-risk identity

At $x$, predicting $+1$ makes an error if $Y=-1$, so its conditional risk is $1-\eta(x)$. Predicting $-1$ gives error $\eta(x)$. Pick $+1$ when $1-\eta(x)<\eta(x)$, i.e. $2\eta(x)-1>0$:

$$f^*(x)=\mathrm{sign}(2\eta(x)-1),$$

with arbitrary tie-breaking at $\eta=1/2$ (the two labels have equal conditional risk). If $f(x)=f^*(x)$, conditional excess risk is zero. Otherwise we choose the worse of the two risks, whose difference is

$$|\eta(x)-(1-\eta(x))|=|2\eta(x)-1|.$$

Combining the cases and averaging over $X$ gives

$$\boxed{R(f)-R(f^*)=\mathbb{E}[|2\eta(X)-1|\,\mathbf{1}\{f(X)\ne f^*(X)\}].}$$

**Important:** the indicator marks *disagreement with Bayes*, not necessarily disagreement with the realized label $Y$. The difference is crucial.

### (d) Margin/noise lower bound (challenging)

Given $|2\eta(X)-1|\ge h>0$ almost surely, multiply by the nonnegative disagreement indicator, obtaining a pointwise inequality. Expectations preserve it:

$$R(f)-R(f^*)\ge h\mathbb{E}[\mathbf{1}\{f(X)\ne f^*(X)\}]=hP(f(X)\ne f^*(X)).$$

In particular, $P(f(X)\ne f^*(X))\le[R(f)-R(f^*)]/h$. Since $h>0$, there are no posterior ties almost surely.

### (e) Regression analogue under squared loss

Let $m(X)=f^*(X)=\mathbb{E}[Y\mid X]$. Expand

$$\begin{aligned}(Y-f(X))^2&=(Y-m(X)+m(X)-f(X))^2\\
&=(Y-m(X))^2+(m(X)-f(X))^2\\
&\quad+2(Y-m(X))(m(X)-f(X)).\end{aligned}$$

The cross term has expectation zero: conditional on $X$, the second factor is fixed and $\mathbb{E}[Y-m(X)\mid X]=0$. Subtract the Bayes risk:

$$\boxed{R(f)-R(f^*)=\mathbb{E}[(f(X)-f^*(X))^2].}$$

Interpretation of $|2\eta-1|$: disagreement hurts most where the posterior is confident (near $0$ or $1$), and hurts little near a $50$–$50$ tie. In regression, squared distance to the conditional mean quantifies excess risk.

---

## Exercise 1.6 — Likelihood ratios, priors, imbalanced data (source p. 3)

### (a) From Bayes' theorem to the likelihood-ratio test

Bayes chooses $+1$ if $P(Y=1\mid X=x)>P(Y=-1\mid X=x)$. By Bayes' theorem, with marginal density $p_X(x)$,

$$P(Y=1\mid x)=\frac{\pi p_+(x)}{p_X(x)},\qquad P(Y=-1\mid x)=\frac{(1-\pi)p_-(x)}{p_X(x)}.$$

The common positive denominator cancels. Thus choose $+1$ if

$$\pi p_+(x)>(1-\pi)p_-(x),$$

or, where $p_-(x)>0$,

$$\boxed{\frac{p_+(x)}{p_-(x)}>\frac{1-\pi}{\pi}.}$$

If $\pi=1/2$, the threshold is $1$ and the larger class-conditional density wins. A **density value is not a probability**: the comparison works because Bayes' theorem converts densities and priors into posterior probabilities. Treat zero-density cases via the undivided inequality.

### (b) Two equal-variance Gaussians

For $p_-=N(0,1)$ and $p_+=N(s,1)$ with $s>0$:

$$\begin{aligned}\log\frac{p_+(x)}{p_-(x)}&=-\frac{(x-s)^2}{2}+\frac{x^2}{2}\\&=sx-\frac{s^2}{2}.\end{aligned}$$

Compare with $\log((1-\pi)/\pi)$:

$$sx-\frac{s^2}{2}>\log\frac{1-\pi}{\pi}\quad\Longleftrightarrow\quad\boxed{x>t_\pi=\frac s2+\frac1s\log\frac{1-\pi}{\pi}.}$$

If the prior changes to $\pi'$, the shift is

$$t_{\pi'}-t_\pi=\frac1s\log\frac{(1-\pi')\pi}{\pi'(1-\pi)}.$$

Increasing $\pi$ lowers the threshold: a more common positive class takes less evidence to predict.

### (c) Severe prior shift: is the old $s/2$ threshold ever useful?

Training with $\pi=1/2$ gives $t=s/2$. At deployment, let $q=\pi'=10^{-6}$. False positives come from $X\sim N(0,1)$ exceeding $s/2$; false negatives from $X\sim N(s,1)$ not exceeding it:

$$\begin{aligned}P(\text{FP}\mid Y=-1)&=1-\Phi(s/2)=\Phi(-s/2),\\P(\text{FN}\mid Y=1)&=\Phi(-s/2).\end{aligned}$$

So deployment risk of the *old* classifier is

$$\boxed{R_{\rm old}=\Phi(-s/2),}$$

independent of $q$ because the two conditional error rates happen to coincide. The trivial constant-negative classifier has $R_{\rm triv}=q=10^{-6}$. The old classifier is better precisely if

$$\Phi(-s/2)<10^{-6}\quad\Longleftrightarrow\quad s>-2\Phi^{-1}(10^{-6})\approx9.50685.$$

At equality they tie. This is a **very large separation**: for a moderately separated signal, severe imbalance can make an apparently reasonable training threshold worse than always predicting the majority class. The Bayes classifier for deployment has threshold $s/2+(1/s)\log((1-q)/q)$, not $s/2$.

### (d) Why undersampling shifts decisions

Undersampling negatives changes the **sample class prior** (say from $\pi$ to $\tilde\pi$) without necessarily changing $p_+,p_-$, provided sampling within each class is random. Training the 0-1 Bayes rule with the artificial prior uses ratio threshold $(1-\tilde\pi)/\tilde\pi$, rather than the true deployment threshold $(1-\pi)/\pi$.

Correct by thresholding the learned likelihood ratio with the *deployment prior*, or by calibrating posterior odds:

$$\frac{P_{\rm deploy}(Y=1\mid x)}{P_{\rm deploy}(Y=-1\mid x)}=\frac{P_{\rm train}(Y=1\mid x)}{P_{\rm train}(Y=-1\mid x)}\cdot\frac{\pi/(1-\pi)}{\tilde\pi/(1-\tilde\pi)}.$$

This correction assumes the class-conditional distributions remain unchanged (prior shift); it is not automatically valid under covariate shift or biased within-class selection.

---

## Exercise 1.7 — Needle in a haystack (source p. 3)

### (a) Compute the cost-sensitive Bayes rule

The sheet gives **negative reward** $\ell(1,1)=-10^6$, not a typo to silently remove; negative loss is permitted mathematically. Let $\eta(x)=P(Y=1\mid X=x)$ and $q=10^{-6}$.

Predict $-1$: $\ell(-1,-1)=0$, $\ell(1,-1)=100$, hence

$$r_x(-1)=100\eta(x).$$

Predict $+1$: $\ell(-1,1)=0$, $\ell(1,1)=-10^6$, hence

$$r_x(+1)=-10^6\eta(x).$$

For every $x$ with $\eta(x)>0$, $r_x(+1)<r_x(-1)$: **always predict $+1$**. At $\eta(x)=0$ either action ties. For the specified Gaussian models and nonzero prior, $\eta(x)>0$ for every real $x$, even when $s=0$, so the global Bayes classifier is identically $+1$.

**Crucial interpretation:** this extreme conclusion follows from the *literal cost table*: there is **no cost whatsoever for a false positive** ($\ell(-1,1)=0$) and a huge reward for a true positive. The number $100$ penalizes a **false negative** ($\ell(1,-1)$), not a false positive. Do not "repair" the table without saying so.

### (b) Compare to ordinary 0–1 loss

For $s>0$, ordinary 0-1 loss predicts $+1$ if

$$x>\frac s2+\frac1s\log\frac{1-q}{q}.$$

For $s<0$ the inequality reverses (divide by a negative $s$); for $s=0$, $X$ carries no class information and ordinary 0-1 Bayes predicts $-1$ because $q<1/2$. Under the **given asymmetric loss**, predict $+1$ everywhere regardless of $s$.

This is not a contradiction: **losses encode what is being optimized**, and a reward can completely overwhelm a rare-class prior. If the intended exercise was to yield a nontrivial cost-dependent threshold, its supplied loss table would need a positive false-positive cost; that is a *hypothetical revision*, not the printed exercise.

---

## Exercise 1.8 — Linear vs quadratic boundaries (source p. 3)

Equal priors and 0-1 loss mean choose $+1$ exactly where $p_+(x)>p_-(x)$. Write $x=(u,v)$, $r^2=u^2+v^2$, and $d^2=(u-1)^2+(v-1)^2$.

### (a) Different covariance matrices: circular boundary

Negative class: $N((0,0),I)$ with density

$$p_-(u,v)=\frac1{2\pi}\exp(-\frac{u^2+v^2}{2}).$$

Positive class: $N((1,1),\tfrac12I)$; in dimension $2$, determinant of covariance is $(1/2)^2=1/4$, so the density prefactor is $1/\pi$:

$$p_+(u,v)=\frac1\pi\exp\bigl(-[(u-1)^2+(v-1)^2]\bigr).$$

Take natural logs (strictly increasing, so the comparison direction is unchanged):

$$\begin{aligned}\log\frac{p_+}{p_-}&=\log2-d^2+\frac{r^2}{2}\\
&=\log2-\frac{r^2}{2}+2u+2v-2.
\end{aligned}$$

Choose $+1$ when this is positive. Multiply by $-2$ and complete the square:

$$\begin{aligned}r^2-4u-4v+4-2\log2&<0\\(u-2)^2+(v-2)^2&<4+2\log2.\end{aligned}$$

Therefore the **positive region is the interior of a circle** centered at $(2,2)$ with radius $\sqrt{4+2\log2}$; its boundary is quadratic. Far away, the negative class wins because its covariance is larger and its tails are heavier. Sketch: circle centered at $(2,2)$, shade its interior $+1$.

### (b) Equal covariance matrices: linear boundary

Now $p_+=N((1,1),I)$ and $p_-=N((0,0),I)$. The normalizing constants cancel:

$$\begin{aligned}\log\frac{p_+}{p_-}&=-\frac{(u-1)^2+(v-1)^2}{2}+\frac{u^2+v^2}{2}\\&=u+v-1.
\end{aligned}$$

Thus $\boxed{f^*(u,v)=+1\ \text{iff}\ u+v>1}$, with the line $u+v=1$ as boundary. In general Gaussian discriminant analysis, **common covariance makes quadratic terms cancel**; unequal covariances generally yield quadratic boundaries.

---

## Exercise 1.9 — Abstention: Chow's rule (source p. 3)

### (a) Why reject only near uncertainty?

At input $x$, the three actions have conditional costs

$$r_x(+1)=1-\eta(x),\qquad r_x(-1)=\eta(x),\qquad r_x(\text{reject})=c.$$

Choose the minimum of these three. The best classification cost without rejecting is $\min(\eta(x),1-\eta(x))=1-\max(\eta(x),1-\eta(x))$. Reject when

$$c<1-\max(\eta,1-\eta)\quad\Longleftrightarrow\quad\boxed{\max(\eta,1-\eta)<1-c.}$$

Equivalently, $c<\eta(x)<1-c$. On the boundaries either rejection or classification is optimal. The smaller of $\eta$ and $1-\eta$ is always at most $1/2$, so if $c\ge1/2$ rejecting never strictly improves the cost. Hence restricting to $0<c<1/2$ captures the interesting regime.

### (b) Uniform X and eta(x)=x: full integrations

Given $X\sim\mathrm{Uniform}[0,1]$ and $\eta(x)=x$, the optimal action is $-1$ for $0\le x\le c$, reject for $c<x<1-c$, and $+1$ for $1-c\le x\le1$.

Reject probability = length of interval under the uniform distribution:

$$\boxed{P(\text{reject})=(1-c)-c=1-2c.}$$

Total optimal risk is the integral of minimum conditional costs:

$$\begin{aligned}R^*(c)&=\int_0^c x\,dx+\int_c^{1-c}c\,dx+\int_{1-c}^1(1-x)\,dx\\
&=\frac{c^2}{2}+c(1-2c)+\frac{c^2}{2}\\
&=\boxed{c-c^2}.\end{aligned}$$

**Error rate among non-rejected examples** is conditional probability of error given *acceptance*, not the overall risk. The total probability mass of accepted examples is $2c$. The joint probability of error and acceptance is

$$\int_0^c x\,dx+\int_{1-c}^1(1-x)\,dx=c^2.$$

Thus

$$\boxed{P(\text{error}\mid\text{accepted})=\frac{c^2}{2c}=\frac c2.}$$

Sanity checks: as $c\to0$, almost everything is rejected and $R^*(c)\to0$; as $c\to1/2$, rejection probability tends to $0$, total risk tends to $1/4$ and conditional error tends to $1/4$ (the ordinary Bayes risk when $\eta(X)=X$ uniformly distributed).

---

## Cross-exercise connections and self-check

The same mechanism underlies every exercise: **first compute conditional risks of the available actions, then minimize them pointwise, and finally average over X if total risk is requested**. The loss determines the decision: squared loss $\to$ conditional mean; absolute loss $\to$ conditional median; correct-or-not $\to$ posterior mode; asymmetric linear costs $\to$ quantile; reject option $\to$ a third action that wins near posterior uncertainty.

Before looking at a solved exercise again, try to reconstruct on paper (i) why the conditional expectation's cross term vanishes in squared-loss algebra, (ii) why a derivative of an expected absolute loss produces a CDF, (iii) how Bayes' theorem changes prior-odds into a likelihood-ratio threshold, and (iv) why an error rate *among accepted points* divides by acceptance probability.

### Three precision traps worth revisiting

1. **Risks vs samples:** $R(f)$ averages over the underlying distribution, not over a fixed dataset; a finite-sample average is empirical risk.
2. **Unclear argument order:** always label which of $\ell(a,b)$ is the actual $Y$ and which is the prediction $z$—especially for pinball, false positives, and false negatives.
3. **Optimization of what?** The *argmin* is a prediction or set of predictions, whereas the *min* is the numerical optimal risk. Do not use them interchangeably.

### Source corrections / qualified statements (not claimed as instructor corrections)

- **1.1(c):** "$Y$ is supported on $[m,M]$" should identify $m,M$ as essential endpoints for the midrange conclusion.
- **1.1(d):** convergence to a specific mode needs regularity/uniqueness assumptions beyond pointwise density approximation.
- **1.1(e):** $3$ is a convenient component mean, but not literally the exact mode of the equal Gaussian mixture.
- **1.2(a):** the printed pinball-loss argument order yields the $(1-\tau)$-quantile, conflicting with the requested $\tau$-quantile. See both derivations above.
- **1.7:** the printed loss table makes predicting $+1$ everywhere optimal; do not invent a nontrivial threshold or silently transpose the costs.

*Derived solutions based on the 2026/27 exercise sheet; explanations and mathematical qualifications are independent tutoring additions.*
