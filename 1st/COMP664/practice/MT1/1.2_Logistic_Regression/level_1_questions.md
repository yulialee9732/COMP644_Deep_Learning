# 1.2 Logistic Regression — Level 1

Sources: [section summary](../../../MT1/1.2%20Logistic%20Reg/1.2_Logistic_Regression.md) and [study notes](../../../MT1/1.2%20Logistic%20Reg/note.md).

Guided questions with formulas and step-by-step worked solutions.

Show your reasoning and state units, dimensions, and assumptions where relevant.

### 1.2-L1-Q01 — Binary probabilities and training

For x=2, y=1, w=b=0, compute probability, cross-entropy, both gradients, and an update with eta=0.1.

**Formula support:** $p=\sigma(w^\top x+b)$, $\sigma(z)=1/(1+e^{-z})$; $\ell=-y\ln p-(1-y)\ln(1-p)$; $g_w=(p-y)x$, $g_b=p-y$.

**Step-by-step solution**

1. $z=0$, so $p=0.5$.
2. $\ell=-\ln0.5\approx0.6931$.
3. $g_w=(0.5-1)2=-1$ and $g_b=-0.5$.
4. Updated parameters are $w=0.1$, $b=0.05$.

### 1.2-L1-Q02 — Multiclass softmax

Logits are (ln 2,0,0) and the true class is 1, using class indices 1–3. Calculate probabilities, loss, and logit gradients.

**Formula support:** $p_c=e^{s_c-m}/\sum_ke^{s_k-m}$, $m=\max s$; $\ell=-\ln p_y$; $\partial\ell/\partial s_c=p_c-\mathbf1_{c=y}$.

**Step-by-step solution**

1. Exponentials are $(2,1,1)$, summing to 4.
2. $p=(0.5,0.25,0.25)$.
3. $\ell=-\ln0.5=\ln2\approx0.6931$.
4. Subtract target $(1,0,0)$: gradient $(-0.5,0.25,0.25)$.

### 1.2-L1-Q03 — Decision boundaries and averaging

For p=sigmoid(2x-4), find the class-1 region at threshold 0.5 and at threshold sigmoid(2).

**Formula support:** $p\ge\tau\iff w^\top x+b\ge\ln[\tau/(1-\tau)]$; mean batch gradients divide summed gradients by $N$.

**Step-by-step solution**

1. At threshold 0.5 the logit cutoff is 0: $2x-4\ge0$, so $x\ge2$.
2. At threshold $\sigma(2)$, the cutoff is 2: $2x-4\ge2$, so $x\ge3$.
