# 1.4 Backpropagation — Level 1

Sources: [section summary](../../../MT1/1.4%20Backprop/1.4_Backpropagation.md) and [study notes](../../../MT1/1.4%20Backprop/note.md).

Guided questions with formulas and step-by-step worked solutions.

Show your reasoning and state units, dimensions, and assumptions where relevant.

### 1.4-L1-Q01 — Multiple paths in a computational graph

Let u=x², v=3x, and L=u+v. At x=2, calculate the forward values and the gradient, showing both paths.

**Formula support:** $dL/dx=\sum_k(\partial L/\partial u_k)(du_k/dx)$. Multiply along paths, add across paths.

**Step-by-step solution**

1. $u=4$, $v=6$, $L=10$.
2. The u path contributes $1(2x)=4$.
3. The v path contributes $1(3)=3$.
4. Add paths: $dL/dx=7$.

### 1.4-L1-Q02 — Forward and backward passes

Use x=2, w1=1, w2=3, y=4 and no biases. Compute the forward pass, loss, and both weight gradients.

**Formula support:** $z_1=w_1x$, $h=\mathrm{ReLU}(z_1)$, $s=w_2h$, $L=\tfrac12(s-y)^2$; $\delta_2=s-y$, $g_{w_2}=\delta_2h$, $g_{w_1}=\delta_2w_2\phi^\prime(z_1)x$.

**Step-by-step solution**

1. $z_1=2$, $h=2$, $s=6$.
2. $L=\tfrac12(6-4)^2=2$.
3. $\delta_2=2$; $g_{w_2}=2(2)=4$.
4. ReLU derivative is 1; $g_{w_1}=2(3)(1)(2)=12$.

### 1.4-L1-Q03 — Output derivatives and deep gradients

For softmax probabilities (0.2,0.5,0.3) and true class 2, derive the gradient for logit 1 by explicitly summing all three class paths.

**Formula support:** $\partial p_i/\partial s_j=p_i(\mathbf1_{i=j}-p_j)$; $\partial L/\partial p_i=-y_i/p_i$; repeated local derivatives multiply.

**Step-by-step solution**

1. Target is $(0,1,0)$, so probability gradients are $(0,-2,0)$.
2. The class-2 path to logit 1 uses $\partial p_2/\partial s_1=-p_2p_1=-0.1$.
3. Total is $0+(-2)(-0.1)+0=0.2$. This equals $p_1-y_1$.
