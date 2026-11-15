# 1.6 Convolutional Neural Networks — Level 1

Sources: [section summary](../../../MT1/1.6%20CNN/1.6_Convolutional_Neural_Networks.md) and [study notes](../../../MT1/1.6%20CNN/note.md).

Guided questions with formulas and step-by-step worked solutions.

Show your reasoning and state units, dimensions, and assumptions where relevant.

### 1.6-L1-Q01 — Convolution shapes and parameters

An input is 32×32×3. Apply eight 3×3 filters, stride 1, padding 1, with biases. Compute output shape and parameter count.

**Formula support:** $D_{out}=\lfloor(D_{in}+2P-K)/S\rfloor+1$ for unit dilation; parameters $=K_hK_wC_{in}C_{out}+C_{out}$.

**Step-by-step solution**

1. Spatial size is $\lfloor(32+2-3)/1\rfloor+1=32$.
2. Eight filters produce eight channels: $32\times32\times8$.
3. Weights $3(3)(3)(8)=216$; biases 8; total 224.

### 1.6-L1-Q02 — Pooling and shared-weight gradients

A 2×2 pooling window is [[1,4],[2,3]] and the upstream gradient is 8. Calculate max/average outputs and both input-gradient arrays.

**Formula support:** Max pooling routes gradient to the selected argmax; average pooling divides it equally. For shared filter weight $w$, $\partial L/\partial w=\sum_t\delta_t x_t$.

**Step-by-step solution**

1. Max output is 4 at top right; its gradient array is [[0,8],[0,0]].
2. Average is $(1+4+2+3)/4=2.5$.
3. Each input receives $8/4=2$, yielding [[2,2],[2,2]].

### 1.6-L1-Q03 — Receptive fields and architecture

Two 3×3 stride-1 convolutional layers are stacked. Compute their receptive-field widths, and explain how later layers combine local features.

**Formula support:** For kernel $k_\ell$, stride $s_\ell$, receptive field $r_\ell=r_{\ell-1}+(k_\ell-1)j_{\ell-1}$ and jump $j_\ell=j_{\ell-1}s_\ell$, with $r_0=j_0=1$.

**Step-by-step solution**

1. First layer: $r_1=1+2(1)=3$, $j_1=1$.
2. Second: $r_2=3+2(1)=5$.
3. A later unit combines nearby earlier detectors and can represent a pattern spanning five input positions per axis.
