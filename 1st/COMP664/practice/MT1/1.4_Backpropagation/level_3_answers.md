# 1.4 Backpropagation — Level 3 answer sheet

[Return to questions](level_3_questions.md)

Open after attempting the corresponding questions. For conceptual questions, these are model answers; equivalent well-supported reasoning is acceptable.

## 1.4-L3-Q01 — Multiple paths in a computational graph

Each use creates a path from the parameter to the loss. The chain rule sums all path contributions. Overwriting retains only one use and loses valid contributions; shared parameters in convolutions and RNNs require accumulation.

## 1.4-L3-Q02 — Forward and backward passes

Let $\delta^\ell=\partial L/\partial z^\ell$. Chain rule gives $\delta^\ell=((W^{\ell+1})^\top\delta^{\ell+1})\odot\phi^\prime(z^\ell)$, dimension $n_\ell$. Since $z^\ell=W^\ell a^{\ell-1}+b^\ell$, $
abla_{W^\ell}L=\delta^\ell(a^{\ell-1})^\top$ has shape $n_\ell\times n_{\ell-1}$; bias gradient is $\delta^\ell$.

## 1.4-L3-Q03 — Output derivatives and deep gradients

$\sum_i(-y_i/p_i)p_i(\mathbf1_{i=j}-p_j)=-y_j+p_j\sum_i y_i=p_j-y_j$. Across depth, derivatives multiply through weight matrices and activation Jacobians. Small factors can contract gradients; weight singular values, paths, and normalization also matter.
