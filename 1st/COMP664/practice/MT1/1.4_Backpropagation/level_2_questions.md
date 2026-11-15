# 1.4 Backpropagation — Level 2

Sources: [section summary](../../../MT1/1.4%20Backprop/1.4_Backpropagation.md) and [study notes](../../../MT1/1.4%20Backprop/note.md).

Questions with formula support. Answers are in the separate [Level 2 answer sheet](level_2_answers.md).

Show your reasoning and state units, dimensions, and assumptions where relevant.

### 1.4-L2-Q01 — Multiple paths in a computational graph

Let u=x², v=2x, L=uv. At x=3, compute L and dL/dx by adding the two path contributions.

**Formula support:** $dL/dx=\sum_k(\partial L/\partial u_k)(du_k/dx)$. Multiply along paths, add across paths.

### 1.4-L2-Q02 — Forward and backward passes

Use x=3, w1=2, w2=1, y=2 and no biases. Compute loss, both gradients, and simultaneous updates with eta=0.01.

**Formula support:** $z_1=w_1x$, $h=\mathrm{ReLU}(z_1)$, $s=w_2h$, $L=\tfrac12(s-y)^2$; $\delta_2=s-y$, $g_{w_2}=\delta_2h$, $g_{w_1}=\delta_2w_2\phi^\prime(z_1)x$.

### 1.4-L2-Q03 — Output derivatives and deep gradients

For probabilities (0.6,0.1,0.3) and true class 3, derive all logit gradients. Separately compute the product of six scalar local derivatives each equal to 0.25.

**Formula support:** $\partial p_i/\partial s_j=p_i(\mathbf1_{i=j}-p_j)$; $\partial L/\partial p_i=-y_i/p_i$; repeated local derivatives multiply.

Solutions are in the separate [Level 2 answer sheet](level_2_answers.md).
