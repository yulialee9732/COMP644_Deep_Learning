# 1.1 Linear Regression — Level 2 answer sheet

[Return to questions](level_2_questions.md)

Open after attempting the corresponding questions. For conceptual questions, these are model answers; equivalent well-supported reasoning is acceptable.

## 1.1-L2-Q01 — Prediction and design matrices

$X_a=[[0,1],[2,1],[4,1]]$ has shape $3\times2$; multiplying by $(-0.5,3)^\top$ gives $(3,2,1)^\top$.

## 1.1-L2-Q02 — Loss, gradients, and loss scaling

Residuals are $(-1,-1)$, sum loss 2, gradient $2[1(-1)+2(-1)]=-6$, and updated weight 1.6. Half-sum loss is 1, gradient -3, and updated weight 1.3. Scaling the loss scales the gradient.

## 1.1-L2-Q03 — Minibatches and epochs

Counts are 1, 240, and 8 per complete pass without replacement. Averaging matches mean-loss scaling; summing makes gradient magnitude depend on batch size and changes the effective step size.
