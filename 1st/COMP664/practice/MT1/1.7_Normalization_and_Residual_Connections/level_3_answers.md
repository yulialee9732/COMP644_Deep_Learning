# 1.7 BatchNorm, LayerNorm, and Residual Connections — Level 3 answer sheet

[Return to questions](level_3_questions.md)

Open after attempting the corresponding questions. For conceptual questions, these are model answers; equivalent well-supported reasoning is acceptable.

## 1.7-L3-Q01 — Normalization arithmetic and axes

Dense BatchNorm computes per-feature statistics over the batch; convolutional BatchNorm usually also pools spatial axes per channel. Inference generally uses running statistics. LayerNorm computes selected feature-axis statistics separately per example/token in both phases. The grouping determines dependence on other examples and what variation is removed.

## 1.7-L3-Q02 — Residual forward and backward paths

For $y=x+F(x)$, $J_y=I+J_F$. Backward gradient is $(I+J_F)^\top g$, including a direct identity contribution. This often helps features and gradients propagate, though cancellation or other blocks can still impair them. When shapes differ, use a compatible projection skip, giving Jacobian $P+J_F$ for linear projection P.

## 1.7-L3-Q03 — Normalization choices and residual design

An acceptable design uses per-token LayerNorm, which does not require other batch examples. Pre-normalization preserves an unnormalized identity path through a sublayer; post-normalization normalizes the residual sum. Both are valid designs with different optimization behavior. Use projected skips wherever widths change; do not claim either design universally guarantees stability.
