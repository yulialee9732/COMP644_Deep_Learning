# 1.7 BatchNorm, LayerNorm, and Residual Connections — Level 1

Sources: [section summary](../../../MT1/1.7%20BatchNorm%2C%20LayerNorm%2C%20Residual%20Connections/1.7_Normalization_and_Residual_Connections.md) and [study notes](../../../MT1/1.7%20BatchNorm%2C%20LayerNorm%2C%20Residual%20Connections/note.md).

Guided questions with formulas and step-by-step worked solutions.

Show your reasoning and state units, dimensions, and assumptions where relevant.

### 1.7-L1-Q01 — Normalization arithmetic and axes

Normalize values (1,3) together using population variance, gamma=2, beta=1, epsilon=0. Identify how grouping differs for dense BatchNorm and LayerNorm.

**Formula support:** $\mu=n^{-1}\sum_i x_i$, $\sigma^2=n^{-1}\sum_i(x_i-\mu)^2$; $y=\gamma(x-\mu)/\sqrt{\sigma^2+\epsilon}+\beta$.

**Step-by-step solution**

1. Mean is 2; variance $[(1-2)^2+(3-2)^2]/2=1$.
2. Standardized values are (-1,1); transformed values are (-1,3).
3. For rows as examples, BatchNorm groups a feature across examples; LayerNorm groups features within one row.

### 1.7-L1-Q02 — Residual forward and backward paths

Let F(x)=0.2x and x=5. Compute y and the gradient with respect to x when the upstream gradient is 3. Compare a block without the skip.

**Formula support:** $y=x+F(x)$; $dy/dx=I+J_F(x)$; for scalar $F(x)=ax$, $dy/dx=1+a$.

**Step-by-step solution**

1. $F(5)=1$, so $y=6$.
2. Residual derivative is $1+0.2=1.2$; input gradient $3(1.2)=3.6$.
3. Without skip, the gradient would be $3(0.2)=0.6$.

### 1.7-L1-Q03 — Normalization choices and residual design

Set F to zero in pre-normalization and post-normalization blocks. Derive both outputs and explain which block becomes exactly the identity.

**Formula support:** Pre-normalization: $y=x+F(\mathrm{LN}(x))$; post-normalization: $y=\mathrm{LN}(x+F(x))$.

**Step-by-step solution**

1. Pre-normalization: $y=x+0=x$.
2. Post-normalization: $y=\mathrm{LN}(x)$, generally different from x.
3. The pre-normalization block retains an exact identity path when its branch is zero.
