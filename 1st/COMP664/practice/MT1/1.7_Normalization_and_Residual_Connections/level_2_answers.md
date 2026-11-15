# 1.7 BatchNorm, LayerNorm, and Residual Connections — Level 2 answer sheet

[Return to questions](level_2_questions.md)

Open after attempting the corresponding questions. For conceptual questions, these are model answers; equivalent well-supported reasoning is acceptable.

## 1.7-L2-Q01 — Normalization arithmetic and axes

BatchNorm column means (3,5), variances (4,4), giving [[-1,-1],[1,1]]. LayerNorm row means (2,6), variances (1,1), giving [[-1,1],[-1,1]].

## 1.7-L2-Q02 — Residual forward and backward paths

First $y=4-2=2$, input gradient $2(1-0.5)=1$. For $F=-x$, output and local derivative are zero, so the gradient is zero. A skip path does not mathematically prevent cancellation.

## 1.7-L2-Q03 — Normalization choices and residual design

Use $y=Px+F(x)$ with $P\in\mathbb R^{64\times32}$. It has 2048 weights. Direct addition of width-32 and width-64 vectors is undefined.
