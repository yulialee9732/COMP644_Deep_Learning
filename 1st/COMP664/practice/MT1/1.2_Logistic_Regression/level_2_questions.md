# 1.2 Logistic Regression — Level 2

Sources: [section summary](../../../MT1/1.2%20Logistic%20Reg/1.2_Logistic_Regression.md) and [study notes](../../../MT1/1.2%20Logistic%20Reg/note.md).

Questions with formula support. Answers are in the separate [Level 2 answer sheet](level_2_answers.md).

Show your reasoning and state units, dimensions, and assumptions where relevant.

### 1.2-L2-Q01 — Binary probabilities and training

For x=-2, y=0, w=b=0 and eta=0.2, compute probability, loss, gradients, and updated parameters. Explain the direction of the logit change.

**Formula support:** $p=\sigma(w^\top x+b)$, $\sigma(z)=1/(1+e^{-z})$; $\ell=-y\ln p-(1-y)\ln(1-p)$; $g_w=(p-y)x$, $g_b=p-y$.

### 1.2-L2-Q02 — Multiclass softmax

Logits are (0,ln 3,ln 2); the true class is 3. Compute probabilities, loss, and logit gradient. Explain numerical stabilization.

**Formula support:** $p_c=e^{s_c-m}/\sum_ke^{s_k-m}$, $m=\max s$; $\ell=-\ln p_y$; $\partial\ell/\partial s_c=p_c-\mathbf1_{c=y}$.

### 1.2-L2-Q03 — Decision boundaries and averaging

Two examples have (p,y,x)=(0.8,1,2) and (0.4,0,-1). Compute mean weight and bias gradients and explain their signs.

**Formula support:** $p\ge\tau\iff w^\top x+b\ge\ln[\tau/(1-\tau)]$; mean batch gradients divide summed gradients by $N$.

Solutions are in the separate [Level 2 answer sheet](level_2_answers.md).
