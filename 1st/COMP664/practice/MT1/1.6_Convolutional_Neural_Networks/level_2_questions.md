# 1.6 Convolutional Neural Networks — Level 2

Sources: [section summary](../../../MT1/1.6%20CNN/1.6_Convolutional_Neural_Networks.md) and [study notes](../../../MT1/1.6%20CNN/note.md).

Questions with formula support. Answers are in the separate [Level 2 answer sheet](level_2_answers.md).

Show your reasoning and state units, dimensions, and assumptions where relevant.

### 1.6-L2-Q01 — Convolution shapes and parameters

An input is 28×28×3. Apply sixteen 5×5 filters, stride 2, padding 1, with biases. Compute output shape and parameter count, explaining the floor.

**Formula support:** $D_{out}=\lfloor(D_{in}+2P-K)/S\rfloor+1$ for unit dilation; parameters $=K_hK_wC_{in}C_{out}+C_{out}$.

### 1.6-L2-Q02 — Pooling and shared-weight gradients

A pooling window is [[5,1],[0,2]] with upstream gradient -4. Give max and average outputs and gradients. A shared scalar filter weight sees patch values 2 and 3 with output gradients 4 and -1; compute its gradient.

**Formula support:** Max pooling routes gradient to the selected argmax; average pooling divides it equally. For shared filter weight $w$, $\partial L/\partial w=\sum_t\delta_t x_t$.

### 1.6-L2-Q03 — Receptive fields and architecture

A 3×3 stride-1 convolution is followed by 2×2 stride-2 pooling and another 3×3 stride-1 convolution. Compute receptive-field width and jump after each layer.

**Formula support:** For kernel $k_\ell$, stride $s_\ell$, receptive field $r_\ell=r_{\ell-1}+(k_\ell-1)j_{\ell-1}$ and jump $j_\ell=j_{\ell-1}s_\ell$, with $r_0=j_0=1$.

Solutions are in the separate [Level 2 answer sheet](level_2_answers.md).
