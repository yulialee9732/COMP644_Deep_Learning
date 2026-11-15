# 1.7 BatchNorm, LayerNorm, and Residual Connections — Level 2

Sources: [section summary](../../../MT1/1.7%20BatchNorm%2C%20LayerNorm%2C%20Residual%20Connections/1.7_Normalization_and_Residual_Connections.md) and [study notes](../../../MT1/1.7%20BatchNorm%2C%20LayerNorm%2C%20Residual%20Connections/note.md).

Questions with formula support. Answers are in the separate [Level 2 answer sheet](level_2_answers.md).

Show your reasoning and state units, dimensions, and assumptions where relevant.

### 1.7-L2-Q01 — Normalization arithmetic and axes

For X=[[1,3],[5,7]], compute normalized values under dense BatchNorm and LayerNorm. Use population variance, gamma=1, beta=0, epsilon=0.

**Formula support:** $\mu=n^{-1}\sum_i x_i$, $\sigma^2=n^{-1}\sum_i(x_i-\mu)^2$; $y=\gamma(x-\mu)/\sqrt{\sigma^2+\epsilon}+\beta$.

### 1.7-L2-Q02 — Residual forward and backward paths

Let F(x)=-0.5x, x=4, upstream gradient 2. Compute residual output and input gradient. What if F(x)=-x instead?

**Formula support:** $y=x+F(x)$; $dy/dx=I+J_F(x)$; for scalar $F(x)=ax$, $dy/dx=1+a$.

### 1.7-L2-Q03 — Normalization choices and residual design

A residual branch outputs a vector of width 64 but the input has width 32. Describe a compatible skip projection, give its column-vector matrix shape, and count its weights without bias.

**Formula support:** Pre-normalization: $y=x+F(\mathrm{LN}(x))$; post-normalization: $y=\mathrm{LN}(x+F(x))$.

Solutions are in the separate [Level 2 answer sheet](level_2_answers.md).
